"""
=============================================================
 TRANSMISOR UDP — Radio Definida por Software (SDR)
=============================================================
Asignatura: Redes de Ordenadores / Tratamiento de Señales
Fase 4: Transporte de muestras sin modular (Voz y Música)

¿Qué hace este script?
  1. Lee un archivo de audio WAV (voz o música, mono o estéreo).
  2. Fragmenta el flujo de audio en bloques de tamaño controlado.
  3. Empaqueta cada bloque con la cabecera binaria del protocolo SDR.
  4. Envía datagramas UDP al receptor (IP y puerto configurables).
  5. Regula el ritmo de envío (pacing) para simular transmisión en tiempo real.
  6. Registra cada datagrama en consola y en un archivo de log TX.
  7. Permite simular pérdidas o desorden deliberado para pruebas de robustez.
"""

import argparse
import os
import socket
import sys
import time
import wave
from datetime import datetime
from typing import List, Tuple

from protocolo_sdr import (
    CabeceraSDR,
    FLAG_START,
    FLAG_DATA,
    FLAG_END,
    MOD_RAW,
    DTYPE_INT16,
    HEADER_SIZE,
)


def leer_audio_wav(ruta: str) -> Tuple[bytes, int, int, int]:
    """
    Lee un archivo WAV y devuelve (bytes_pcm, sample_rate, num_canales, num_muestras_por_canal).
    Garantiza formato PCM int16.
    """
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No existe el archivo de audio: '{ruta}'")

    with wave.open(ruta, "rb") as wf:
        n_canales = wf.getnchannels()
        ancho = wf.getsampwidth()
        fs = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    # Si es int16 estándar (2 bytes por muestra)
    if ancho == 2:
        pcm_bytes = raw_bytes
    elif ancho == 1:
        # Convertir uint8 a int16
        muestras_int16 = bytearray()
        for b in raw_bytes:
            val = (b - 128) * 256
            muestras_int16.extend(val.to_bytes(2, byteorder="little", signed=True))
        pcm_bytes = bytes(muestras_int16)
    else:
        # Si tiene otro ancho, advertir pero intentar procesar
        print(f"[AVISO] Formato con ancho {ancho} bytes. Se asume PCM lineal.")
        pcm_bytes = raw_bytes

    return pcm_bytes, fs, n_canales, n_frames


def dividir_en_bloques(
    pcm_bytes: bytes,
    tam_bloque_muestras: int,
    num_canales: int,
    bytes_por_muestra: int = 2
) -> List[bytes]:
    """
    Divide los bytes PCM en bloques según el número de muestras por canal.
    bytes_por_bloque = tam_bloque_muestras * num_canales * bytes_por_muestra
    """
    bytes_por_bloque = tam_bloque_muestras * num_canales * bytes_por_muestra
    bloques = []
    total_len = len(pcm_bytes)

    for i in range(0, total_len, bytes_por_bloque):
        bloques.append(pcm_bytes[i : i + bytes_por_bloque])

    return bloques


def transmitir_udp(
    ruta_audio: str,
    ip_destino: str = "127.0.0.1",
    puerto_destino: int = 5005,
    tam_bloque: int = 512,
    tiempo_real: bool = True,
    descartar_seq: int = -1,
    desordenar_seq: int = -1,
    archivo_log: str = "log_udp_tx.txt",
):
    """Ejecuta la transmisión UDP con logging y opciones de simulación de fallos."""
    print("=" * 65)
    print(" TRANSMISOR UDP — SDR Fase 4 (Transporte)")
    print("=" * 65)

    pcm_bytes, fs, canales, total_frames = leer_audio_wav(ruta_audio)
    duracion_s = total_frames / fs
    tipo_audio = "Estéreo (Música)" if canales == 2 else "Mono (Voz/Música)"

    print(f"[*] Archivo fuente      : {ruta_audio}")
    print(f"[*] Formato             : {tipo_audio} | {fs} Hz | 16-bit PCM")
    print(f"[*] Total muestras/ch   : {total_frames:,} ({duracion_s:.2f} s)")
    print(f"[*] Destino             : {ip_destino}:{puerto_destino}")
    print(f"[*] Tamaño bloque       : {tam_bloque} muestras/canal")

    bloques = dividir_en_bloques(pcm_bytes, tam_bloque, canales, bytes_por_muestra=2)
    total_bloques = len(bloques)
    cadencia_bloque_s = tam_bloque / fs

    print(f"[*] Total datagramas    : {total_bloques} bloques (~{cadencia_bloque_s*1000:.2f} ms/bloque)")
    if descartar_seq >= 0:
        print(f"[!] PRUEBA DE ROBUSTEZ  : Se descartará intencionadamente el bloque #{descartar_seq}")
    if desordenar_seq >= 0:
        print(f"[!] PRUEBA DE ROBUSTEZ  : Se desordenará el bloque #{desordenar_seq} con el #{desordenar_seq+1}")

    # Abrir socket UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Preparar archivo de log
    lineas_log = [
        f"# LOG UDP TRANSMISOR (TX) - {datetime.now().isoformat()}",
        f"# Audio: {ruta_audio} | {fs} Hz | {canales} canales | {total_bloques} bloques",
        f"# Destino: {ip_destino}:{puerto_destino} | Tamaño bloque: {tam_bloque} muestras",
        "# " + "-" * 60,
        "# SEQ | TIMESTAMP | BYTES | ESTADO | DETALLE",
    ]

    print("\n[>] Iniciando transmisión UDP...")
    tiempo_inicio = time.time()
    bytes_enviados_totales = 0
    bloques_enviados = 0

    # 1. Enviar paquete START (metadatos iniciales)
    hdr_start = CabeceraSDR(
        sequence=0,
        samples=0,
        sample_rate=fs,
        channels=canales,
        modulation=MOD_RAW,
        dtype=DTYPE_INT16,
        flags=FLAG_START,
    )
    pkt_start = hdr_start.pack(b"")
    sock.sendto(pkt_start, (ip_destino, puerto_destino))
    lineas_log.append(f"START | {time.time():.4f} | {len(pkt_start)} | ENVIADO | Inicio de sesión")

    # 2. Transmisión de bloques de datos
    idx = 0
    while idx < total_bloques:
        # Lógica de prueba de desorden deliberado
        bloque_actual_idx = idx
        if desordenar_seq >= 0 and idx == desordenar_seq and (idx + 1) < total_bloques:
            print(f"  [ROBUSTEZ] Alterando orden: enviando primero #{idx+1} antes de #{idx}")
            orden_envio = [idx + 1, idx]
            idx += 2
        else:
            orden_envio = [idx]
            idx += 1

        for b_idx in orden_envio:
            seq_num = b_idx + 1  # Secuencia 1-indexada tras START
            payload = bloques[b_idx]
            muestras_bloque = len(payload) // (canales * 2)

            # Lógica de descarte deliberado
            if descartar_seq >= 0 and seq_num == descartar_seq:
                print(f"  [ROBUSTEZ] Bloque #{seq_num:04d} DESCARTADO intencionadamente (simulación pérdida)")
                lineas_log.append(f"{seq_num:05d} | {time.time():.4f} | {len(payload)} | DESCARTADO | Simulación de fallo")
                continue

            cabecera = CabeceraSDR(
                sequence=seq_num,
                samples=muestras_bloque,
                sample_rate=fs,
                channels=canales,
                modulation=MOD_RAW,
                dtype=DTYPE_INT16,
                flags=FLAG_DATA,
            )
            paquete = cabecera.pack(payload)
            sock.sendto(paquete, (ip_destino, puerto_destino))

            bytes_enviados_totales += len(paquete)
            bloques_enviados += 1

            linea_txt = (
                f"{seq_num:05d} | {cabecera.timestamp:.4f} | {len(paquete)} B | ENVIADO | "
                f"{muestras_bloque} m/ch ({len(payload)} B payload)"
            )
            lineas_log.append(linea_txt)

            if seq_num % 50 == 0 or seq_num == total_bloques:
                pct = (seq_num / total_bloques) * 100
                print(f"  [TX] Bloque #{seq_num:04d}/{total_bloques} ({pct:5.1f}%) | {len(paquete)} B -> {ip_destino}:{puerto_destino}")

            # Pacing de tiempo real o pequeña pausa técnica para evitar desbordar el socket
            if tiempo_real:
                time.sleep(cadencia_bloque_s * 0.95)  # Ligero margen para no atrasarse
            else:
                time.sleep(0.0005)  # 0.5 ms para dar respiro al buffer UDP del receptor

    # 3. Enviar paquete END
    hdr_end = CabeceraSDR(
        sequence=total_bloques + 1,
        samples=0,
        sample_rate=fs,
        channels=canales,
        modulation=MOD_RAW,
        dtype=DTYPE_INT16,
        flags=FLAG_END,
    )
    pkt_end = hdr_end.pack(b"")
    # Enviar END varias veces para asegurar recepción en UDP no fiable
    for _ in range(3):
        sock.sendto(pkt_end, (ip_destino, puerto_destino))
        time.sleep(0.01)

    lineas_log.append(f"END   | {time.time():.4f} | {len(pkt_end)} | ENVIADO | Fin de sesión")

    tiempo_total = time.time() - tiempo_inicio
    sock.close()

    # Resumen y estadísticas
    kbps = (bytes_enviados_totales * 8) / (tiempo_total * 1000) if tiempo_total > 0 else 0
    resumen_lineas = [
        "# " + "=" * 60,
        "# RESUMEN DE TRANSMISIÓN (TX)",
        f"# Bloques de datos enviados : {bloques_enviados}/{total_bloques}",
        f"# Total bytes transmitidos  : {bytes_enviados_totales:,} B ({bytes_enviados_totales/1024:.2f} KB)",
        f"# Tiempo de transmisión     : {tiempo_total:.3f} s (Audio: {duracion_s:.3f} s)",
        f"# Rendimiento efectivo      : {kbps:.2f} kbps",
        "# " + "=" * 60,
    ]
    lineas_log.extend(resumen_lineas)

    # Guardar archivo de log
    try:
        with open(archivo_log, "w", encoding="utf-8") as f:
            f.write("\n".join(lineas_log) + "\n")
        print(f"[OK] Log de transmisión guardado en: '{archivo_log}'")
    except Exception as e:
        print(f"[AVISO] No se pudo escribir log: {e}")

    print("\n" + "=" * 65)
    print(" RESUMEN TRANSMISOR UDP")
    print(f"  - Datagramas enviados   : {bloques_enviados}")
    print(f"  - Bytes totales         : {bytes_enviados_totales:,} bytes")
    print(f"  - Tiempo total          : {tiempo_total:.2f} segundos")
    print(f"  - Tasa media            : {kbps:.2f} kbps")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transmisor UDP SDR — Fase 4")
    parser.add_argument("--archivo", default="audio_original.wav", help="Ruta del archivo WAV a transmitir")
    parser.add_argument("--ip", default="127.0.0.1", help="Dirección IP del receptor (por defecto 127.0.0.1)")
    parser.add_argument("--puerto", type=int, default=5005, help="Puerto UDP del receptor (por defecto 5005)")
    parser.add_argument("--tam-bloque", type=int, default=512, help="Muestras por canal por datagrama (def: 512)")
    parser.add_argument("--rapido", action="store_true", help="Enviar a máxima velocidad sin simular tiempo real")
    parser.add_argument("--descartar", type=int, default=-1, help="Número de secuencia a descartar (prueba de robustez)")
    parser.add_argument("--desordenar", type=int, default=-1, help="Número de secuencia a desordenar (prueba de robustez)")
    parser.add_argument("--log", default="log_udp_tx.txt", help="Archivo de salida para el log TX")

    args = parser.parse_args()

    # Si se ejecuta sin archivo específico y no existe en carpeta actual, buscar en directorio
    ruta = args.archivo
    if not os.path.isfile(ruta):
        alternativa = os.path.join(os.path.dirname(__file__), ruta)
        if os.path.isfile(alternativa):
            ruta = alternativa

    transmitir_udp(
        ruta_audio=ruta,
        ip_destino=args.ip,
        puerto_destino=args.puerto,
        tam_bloque=args.tam_bloque,
        tiempo_real=not args.rapido,
        descartar_seq=args.descartar,
        desordenar_seq=args.desordenar,
        archivo_log=args.log,
    )
