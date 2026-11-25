"""
=============================================================
 RECEPTOR UDP — Radio Definida por Software (SDR)
=============================================================
Asignatura: Redes de Ordenadores / Tratamiento de Señales
Fase 4: Transporte de muestras sin modular (Voz y Música)

¿Qué hace este script?
  1. Abre un socket UDP a la escucha en la IP y puerto especificados.
  2. Desempaqueta y valida la cabecera binaria del protocolo SDR.
  3. Detecta anomalías en tiempo real:
     - Paquetes fuera de orden.
     - Paquetes duplicados.
     - Huecos de secuencia (paquetes perdidos).
  4. Calcula estadísticas de red: retardo de tránsito y jitter (RFC 3550).
  5. Aplica técnicas de ocultación de pérdidas (PLC):
     - 'ceros'      : Relleno con silencio.
     - 'repetir'    : Repetición del bloque anterior (continuidad).
     - 'interpolar' : Transición suave entre bloques adyacentes.
  6. Reconstruye y guarda el audio recuperado en formato WAV (mono o estéreo).
  7. Genera un archivo de log RX con el informe completo de la recepción.
"""

import argparse
import os
import socket
import struct
import sys
import time
import wave
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# Ruta al directorio de modulos (funciona tanto si se ejecuta desde la raiz como desde modulos/)
import os as _os, sys as _sys
_dir = _os.path.dirname(_os.path.abspath(__file__))
if _dir not in _sys.path:
    _sys.path.insert(0, _dir)

from protocolo_sdr import (
    CabeceraSDR,
    FLAG_START,
    FLAG_DATA,
    FLAG_END,
    DTYPE_INT16,
    HEADER_SIZE,
)

# Intentar importar sounddevice para reproducción opcional
try:
    import sounddevice as sd
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False


def guardar_wav_pcm16(ruta: str, pcm_bytes: bytes, fs: int, canales: int):
    """Guarda un buffer de bytes PCM 16-bit como archivo WAV estándar."""
    with wave.open(ruta, "wb") as wf:
        wf.setnchannels(canales)
        wf.setsampwidth(2)  # 16 bits = 2 bytes
        wf.setframerate(fs)
        wf.writeframes(pcm_bytes)


def ocultar_perdida(
    prev_bytes: Optional[bytes],
    sig_bytes: Optional[bytes],
    num_bytes_esperados: int,
    estrategia: str = "repetir"
) -> bytes:
    """Aplica Packet Loss Concealment (PLC) cuando falta un bloque."""
    if estrategia == "ceros" or prev_bytes is None:
        return b"\x00" * num_bytes_esperados

    if estrategia == "repetir":
        # Repetir el bloque anterior (truncando o extendiendo si difiere)
        if len(prev_bytes) >= num_bytes_esperados:
            return prev_bytes[:num_bytes_esperados]
        else:
            return (prev_bytes * ((num_bytes_esperados // len(prev_bytes)) + 1))[:num_bytes_esperados]

    if estrategia == "interpolar" and prev_bytes and sig_bytes:
        # Interpolación lineal entre última muestra anterior y primera muestra siguiente
        try:
            val_ini = struct.unpack_from("<h", prev_bytes, len(prev_bytes) - 2)[0]
            val_fin = struct.unpack_from("<h", sig_bytes, 0)[0]
            num_muestras = num_bytes_esperados // 2
            interpolado = bytearray()
            for i in range(num_muestras):
                alpha = i / max(1, num_muestras - 1)
                v = int(val_ini * (1.0 - alpha) + val_fin * alpha)
                v = max(-32768, min(32767, v))
                interpolado.extend(struct.pack("<h", v))
            return bytes(interpolado)
        except Exception:
            return prev_bytes[:num_bytes_esperados]

    return prev_bytes[:num_bytes_esperados] if prev_bytes else b"\x00" * num_bytes_esperados


def recibir_udp(
    ip_escucha: str = "0.0.0.0",
    puerto_escucha: int = 5005,
    salida_wav: str = "audio_fase4_recuperado.wav",
    estrategia_perdidas: str = "repetir",
    timeout_inactividad: float = 3.0,
    reproducir: bool = False,
    archivo_log: str = "log_udp_rx.txt",
) -> Dict:
    """Ejecuta el bucle del receptor UDP con análisis de flujo y reconstrucción."""
    print("=" * 65)
    print(" RECEPTOR UDP — SDR Fase 4 (Transporte)")
    print("=" * 65)
    print(f"[*] Escuchando en         : {ip_escucha}:{puerto_escucha}")
    print(f"[*] Salida WAV            : {salida_wav}")
    print(f"[*] Estrategia ante fallos: {estrategia_perdidas.upper()}")
    print(f"[*] Timeout inactividad   : {timeout_inactividad} s")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip_escucha, puerto_escucha))
    sock.settimeout(timeout_inactividad)

    bloques_recibidos: Dict[int, Tuple[CabeceraSDR, bytes, float]] = {}
    secuencias_en_orden: List[int] = []
    paquetes_duplicados = 0
    paquetes_desordenados = 0

    metadatos_fs = 48000
    metadatos_canales = 1
    metadatos_tam_bloque = 512
    sesion_iniciada = False
    seq_esperada = 1

    # Cálculo de Jitter según RFC 3550
    jitter = 0.0
    prev_rx_time = 0.0
    prev_tx_time = 0.0

    lineas_log = [
        f"# LOG UDP RECEPTOR (RX) - {datetime.now().isoformat()}",
        f"# Escucha: {ip_escucha}:{puerto_escucha} | Salida: {salida_wav}",
        f"# Estrategia PLC: {estrategia_perdidas}",
        "# " + "-" * 60,
        "# SEQ | RX_TIME | D_BYTES | RETARDO(ms) | JITTER(ms) | ESTADO",
    ]

    print("\n[>] Esperando datagramas UDP...")
    tiempo_primer_paquete = None
    tiempo_ultimo_paquete = None

    try:
        while True:
            try:
                datos, addr = sock.recvfrom(65535)
                tiempo_rx = time.time()
            except socket.timeout:
                if sesion_iniciada:
                    print("\n[!] Fin de transmisión por inactividad (timeout alcanzado).")
                    break
                else:
                    # Sigue esperando el inicio
                    continue

            if tiempo_primer_paquete is None:
                tiempo_primer_paquete = tiempo_rx
            tiempo_ultimo_paquete = tiempo_rx

            # Deserializar cabecera
            try:
                cabecera, payload = CabeceraSDR.unpack(datos)
            except ValueError as e:
                print(f"[!] Paquete descartado: {e}")
                continue

            seq = cabecera.sequence
            retardo_ms = (tiempo_rx - cabecera.timestamp) * 1000.0 if cabecera.timestamp > 0 else 0.0

            # Cálculo de jitter (RFC 3550)
            if prev_rx_time > 0 and cabecera.timestamp > 0:
                d = (tiempo_rx - prev_rx_time) - (cabecera.timestamp - prev_tx_time)
                d = abs(d)
                jitter += (d - jitter) / 16.0
            prev_rx_time = tiempo_rx
            prev_tx_time = cabecera.timestamp

            jitter_ms = jitter * 1000.0

            # Manejo de paquetes según flags
            if cabecera.is_start:
                sesion_iniciada = True
                metadatos_fs = cabecera.sample_rate
                metadatos_canales = cabecera.channels
                print(f"[+] INICIO DE SESIÓN recibido desde {addr[0]}:{addr[1]}")
                print(f"    Fs: {metadatos_fs} Hz | Canales: {cabecera.channels_name} | Modo: {cabecera.modulation_name}")
                lineas_log.append(f"START | {tiempo_rx:.4f} | {len(datos)} | {retardo_ms:.2f} | 0.00 | SESION_INICIADA")
                continue

            if cabecera.is_end:
                print(f"[+] FIN DE SESIÓN recibido desde {addr[0]}:{addr[1]}")
                lineas_log.append(f"END   | {tiempo_rx:.4f} | {len(datos)} | {retardo_ms:.2f} | {jitter_ms:.2f} | SESION_FINALIZADA")
                break

            # Paquete de datos normal
            sesion_iniciada = True
            metadatos_fs = cabecera.sample_rate
            metadatos_canales = cabecera.channels
            metadatos_tam_bloque = cabecera.samples

            estado_str = "OK"

            # Detección de duplicado
            if seq in bloques_recibidos:
                paquetes_duplicados += 1
                estado_str = "DUPLICADO"
                print(f"  [ALERTA] Paquete duplicado detectado: SEQ #{seq:04d}")
            else:
                # Detección de desorden
                if seq != seq_esperada:
                    if seq < seq_esperada:
                        paquetes_desordenados += 1
                        estado_str = "DESORDEN (TARDIO)"
                        print(f"  [ALERTA] Paquete desordenado (tardío): SEQ #{seq:04d} (se esperaba #{seq_esperada:04d})")
                    else:
                        hueco = seq - seq_esperada
                        estado_str = f"HUECO ({hueco} perdidos)"
                        print(f"  [ALERTA] Salto de secuencia: se esperaba #{seq_esperada:04d}, llegó #{seq:04d} (posible pérdida)")
                
                bloques_recibidos[seq] = (cabecera, payload, tiempo_rx)
                secuencias_en_orden.append(seq)
                seq_esperada = max(seq_esperada, seq + 1)

            lineas_log.append(
                f"{seq:05d} | {tiempo_rx:.4f} | {len(datos)} B | {retardo_ms:6.2f} ms | {jitter_ms:5.2f} ms | {estado_str}"
            )

            if len(bloques_recibidos) % 50 == 0:
                print(f"  [RX] Recibidos {len(bloques_recibidos)} bloques | Jitter: {jitter_ms:.2f} ms | Retardo: {retardo_ms:.2f} ms")

    except KeyboardInterrupt:
        print("\n[!] Recepción interrumpida manualmente por el usuario.")
    finally:
        sock.close()

    # ─────────────────────────────────────────────────────────────────────────
    # RECONSTRUCCIÓN DEL AUDIO Y OCULTACIÓN DE PÉRDIDAS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" RECONSTRUCCIÓN Y CONTROL DE PÉRDIDAS (PLC)")
    print("=" * 65)

    if not bloques_recibidos:
        print("[ERROR] No se recibieron paquetes de audio válidos.")
        return {}

    seq_min = min(bloques_recibidos.keys())
    seq_max = max(bloques_recibidos.keys())
    total_esperados = seq_max - seq_min + 1
    total_recibidos_unicos = len(bloques_recibidos)
    total_perdidos = total_esperados - total_recibidos_unicos
    pct_perdida = (total_perdidos / total_esperados) * 100.0 if total_esperados > 0 else 0.0

    print(f"[*] Secuencia detectada : #{seq_min} hasta #{seq_max}")
    print(f"[*] Bloques esperados   : {total_esperados}")
    print(f"[*] Bloques recibidos   : {total_recibidos_unicos}")
    print(f"[*] Bloques perdidos    : {total_perdidos} ({pct_perdida:.2f} %)")
    print(f"[*] Paquetes duplicados : {paquetes_duplicados}")
    print(f"[*] Desórdenes vistos   : {paquetes_desordenados}")

    # Bytes esperados por bloque
    bytes_por_bloque = metadatos_tam_bloque * metadatos_canales * 2

    audio_reconstruido = bytearray()
    bloque_anterior: Optional[bytes] = None

    for s in range(seq_min, seq_max + 1):
        if s in bloques_recibidos:
            _, payload, _ = bloques_recibidos[s]
            audio_reconstruido.extend(payload)
            bloque_anterior = payload
        else:
            # Paquete ausente: aplicar PLC
            siguiente_payload = bloques_recibidos[s + 1][1] if (s + 1) in bloques_recibidos else None
            bloque_reparado = ocultar_perdida(
                bloque_anterior,
                siguiente_payload,
                bytes_por_bloque,
                estrategia=estrategia_perdidas
            )
            audio_reconstruido.extend(bloque_reparado)
            bloque_anterior = bloque_reparado

    # Guardar archivo WAV
    guardar_wav_pcm16(salida_wav, bytes(audio_reconstruido), metadatos_fs, metadatos_canales)
    duracion_recuperada = (len(audio_reconstruido) // (metadatos_canales * 2)) / metadatos_fs
    print(f"[OK] Audio reconstruido guardado en: '{salida_wav}'")
    print(f"     Duración: {duracion_recuperada:.2f} s | Canales: {metadatos_canales} | Fs: {metadatos_fs} Hz")

    # Resumen y métricas finales
    tiempo_transcurrido = (tiempo_ultimo_paquete - tiempo_primer_paquete) if (tiempo_primer_paquete and tiempo_ultimo_paquete) else 0.0
    bytes_totales_recibidos = sum(len(b[1]) + HEADER_SIZE for b in bloques_recibidos.values())
    kbps = (bytes_totales_recibidos * 8) / (tiempo_transcurrido * 1000) if tiempo_transcurrido > 0 else 0

    resumen_lineas = [
        "# " + "=" * 60,
        "# RESUMEN DE RECEPCIÓN (RX)",
        f"# Bloques únicos recibidos   : {total_recibidos_unicos}/{total_esperados}",
        f"# Bloques perdidos detectados: {total_perdidos} ({pct_perdida:.2f} %)",
        f"# Paquetes duplicados        : {paquetes_duplicados}",
        f"# Paquetes desordenados      : {paquetes_desordenados}",
        f"# Jitter medio final         : {jitter*1000:.2f} ms",
        f"# Tasa de datos recibida     : {kbps:.2f} kbps",
        f"# Audio generado             : {salida_wav} ({duracion_recuperada:.3f} s)",
        "# " + "=" * 60,
    ]
    lineas_log.extend(resumen_lineas)

    try:
        with open(archivo_log, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas_log) + "\n")
        print(f"[OK] Log de recepción guardado en: '{archivo_log}'")
    except Exception as e:
        print(f"[AVISO] No se pudo guardar el log: {e}")

    # Reproducción opcional
    if reproducir and AUDIO_OK:
        print("\n[>] Reproduciendo audio recibido...")
        try:
            # Leer WAV para reproducir
            with wave.open(salida_wav, "rb") as wf:
                data = wf.readframes(wf.getnframes())
            import numpy as np
            arr = np.frombuffer(data, dtype=np.int16)
            if metadatos_canales == 2:
                arr = arr.reshape(-1, 2)
            sd.play(arr, metadatos_fs)
            sd.wait()
            print("[OK] Reproducción finalizada.")
        except Exception as e:
            print(f"[AVISO] Error al reproducir: {e}")

    return {
        "esperados": total_esperados,
        "recibidos": total_recibidos_unicos,
        "perdidos": total_perdidos,
        "pct_perdida": pct_perdida,
        "duplicados": paquetes_duplicados,
        "desordenados": paquetes_desordenados,
        "jitter_ms": jitter * 1000.0,
        "kbps": kbps,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receptor UDP SDR — Fase 4")
    parser.add_argument("--ip", default="0.0.0.0", help="IP de escucha (0.0.0.0 para todas las interfaces)")
    parser.add_argument("--puerto", type=int, default=5005, help="Puerto UDP de escucha (def: 5005)")
    parser.add_argument("--salida", default="audio_fase4_recuperado.wav", help="Archivo WAV de salida")
    parser.add_argument("--plc", choices=["repetir", "ceros", "interpolar"], default="repetir", help="Estrategia ante pérdidas")
    parser.add_argument("--timeout", type=float, default=2.5, help="Timeout de inactividad en segundos")
    parser.add_argument("--reproducir", action="store_true", help="Reproducir el audio al terminar")
    parser.add_argument("--log", default="log_udp_rx.txt", help="Archivo de log de salida")

    args = parser.parse_args()

    recibir_udp(
        ip_escucha=args.ip,
        puerto_escucha=args.puerto,
        salida_wav=args.salida,
        estrategia_perdidas=args.plc,
        timeout_inactividad=args.timeout,
        reproducir=args.reproducir,
        archivo_log=args.log,
    )
