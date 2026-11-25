"""
=============================================================
 RECEPTOR UDP MODULADO — SDR Fase 5
=============================================================
Asignatura: Redes de Ordenadores / Tratamiento de Señales
Fase 5: Integración Modulación AM/FM + Transporte UDP

¿Qué hace este script?
  1. Escucha datagramas UDP en el puerto configurado.
  2. Detecta automáticamente el modo de modulación leyendo
     el campo `modulation` de la cabecera SDR (AM o FM).
  3. Demodula cada bloque en tiempo real conforme llega.
  4. Detecta pérdidas y aplica PLC antes de demodular.
  5. Reconstruye y guarda el audio recuperado en WAV.
  6. Calcula y reporta métricas de red (jitter, retardo, pérdidas)
     y de señal (SNR AM vs FM).
  7. Genera log RX con evidencias para la entrega.
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

# Añadir vendor al path si existe
_vendor = os.path.join(os.path.dirname(__file__), "vendor")
if os.path.isdir(_vendor) and _vendor not in sys.path:
    sys.path.insert(0, _vendor)

import numpy as np


# Ruta al directorio de modulos (funciona tanto si se ejecuta desde la raiz como desde modulos/)
import os as _os, sys as _sys
_dir = _os.path.dirname(_os.path.abspath(__file__))
if _dir not in _sys.path:
    _sys.path.insert(0, _dir)

from protocolo_sdr import (
    CabeceraSDR,
    FLAG_START, FLAG_DATA, FLAG_END,
    MOD_RAW, MOD_AM, MOD_FM,
    DTYPE_INT16, HEADER_SIZE,
)
from modulador_en_linea import (
    demodular_am_bloque,
    demodular_fm_bloque,
    deserializar_bloque,
    calcular_snr,
    FC_AM_DEFAULT, FC_FM_DEFAULT,
    KA_DEFAULT, KF_DEFAULT,
)


def guardar_wav(ruta: str, señal_float32: np.ndarray, fs: int, canales: int = 1):
    """Guarda un array float32 [-1,1] como WAV PCM int16."""
    datos = np.clip(señal_float32, -1.0, 1.0)
    datos_int16 = (datos * 32767).astype('<i2')
    with wave.open(ruta, "wb") as wf:
        wf.setnchannels(canales)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(datos_int16.tobytes())


def recibir_modulado(
    ip_escucha: str = "0.0.0.0",
    puerto: int = 5006,
    salida_wav: str = "audio_fase5_recuperado.wav",
    fc: Optional[float] = None,
    ka: float = KA_DEFAULT,
    kf: float = KF_DEFAULT,
    timeout: float = 2.5,
    archivo_log: str = "log_fase5_rx.txt",
    ruta_original: Optional[str] = None,
) -> dict:
    """Bucle receptor: recibe, demodula y reconstruye el audio."""
    print("=" * 65)
    print(" RECEPTOR UDP MODULADO — SDR Fase 5")
    print("=" * 65)
    print(f"[*] Escuchando en  : {ip_escucha}:{puerto}")
    print(f"[*] Salida WAV     : {salida_wav}")
    print(f"[*] Timeout        : {timeout} s")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip_escucha, puerto))
    sock.settimeout(timeout)

    # Estado de recepción
    bloques_rx: Dict[int, Tuple[np.ndarray, float]] = {}  # seq → (muestras_float32, t_rx)
    seq_esperada = 1
    perd_detectadas = 0
    desordenados = 0
    duplicados = 0

    # Metadatos extraídos del primer paquete
    fs_rx = 48_000
    canales_rx = 1
    mod_rx = MOD_FM
    tam_bloque_rx = 512

    # Jitter RFC 3550
    jitter = 0.0
    prev_rx = 0.0
    prev_tx = 0.0

    sesion_ok = False
    t_primero = None
    t_ultimo  = None

    lineas_log = [
        f"# LOG RX MODULADO (Fase 5) — {datetime.now().isoformat()}",
        f"# Escucha: {ip_escucha}:{puerto} | Salida: {salida_wav}",
        "# " + "-" * 60,
        "# SEQ | RX_TIME | BYTES | RETARDO_ms | JITTER_ms | ESTADO",
    ]

    print("\n[>] Esperando datagramas UDP modulados...")

    try:
        while True:
            try:
                datos, addr = sock.recvfrom(65535)
                t_rx = time.time()
            except socket.timeout:
                if sesion_ok:
                    print("\n[!] Timeout — fin de transmisión.")
                    break
                continue

            if t_primero is None:
                t_primero = t_rx
            t_ultimo = t_rx

            try:
                cabecera, payload = CabeceraSDR.unpack(datos)
            except ValueError as e:
                print(f"[!] Paquete descartado: {e}")
                continue

            seq      = cabecera.sequence
            retardo  = (t_rx - cabecera.timestamp) * 1000.0
            mod_rx   = cabecera.modulation
            fs_rx    = cabecera.sample_rate
            canales_rx = cabecera.channels

            # Jitter (RFC 3550)
            if prev_rx > 0 and cabecera.timestamp > 0:
                d = abs((t_rx - prev_rx) - (cabecera.timestamp - prev_tx))
                jitter += (d - jitter) / 16.0
            prev_rx = t_rx
            prev_tx = cabecera.timestamp

            if cabecera.is_start:
                sesion_ok = True
                mod_nombre = "FM" if mod_rx == MOD_FM else "AM"
                print(f"[+] INICIO DE SESIÓN desde {addr[0]}:{addr[1]}")
                print(f"    Modo: {mod_nombre} | Fs: {fs_rx} Hz | Canales: {cabecera.channels_name}")
                lineas_log.append(f"START | {t_rx:.4f} | {len(datos)} | {retardo:.2f} | 0.00 | INICIO")
                continue

            if cabecera.is_end:
                print(f"[+] FIN DE SESIÓN recibido.")
                lineas_log.append(f"END   | {t_rx:.4f} | {len(datos)} | {retardo:.2f} | {jitter*1000:.2f} | FIN")
                break

            # Paquete de datos: deserializar muestras moduladas
            sesion_ok = True
            tam_bloque_rx = cabecera.samples
            muestras_mod = deserializar_bloque(payload)

            estado_str = "OK"

            if seq in bloques_rx:
                duplicados += 1
                estado_str = "DUPLICADO"
            else:
                if seq != seq_esperada:
                    if seq < seq_esperada:
                        desordenados += 1
                        estado_str = f"DESORDEN (tardío)"
                    else:
                        gap = seq - seq_esperada
                        perd_detectadas += gap
                        estado_str = f"HUECO ({gap} perdidos)"
                        print(f"  [!] Hueco: esperaba #{seq_esperada:04d}, llegó #{seq:04d}")

                bloques_rx[seq] = (muestras_mod, t_rx)
                seq_esperada = max(seq_esperada, seq + 1)

            lineas_log.append(
                f"{seq:05d} | {t_rx:.4f} | {len(datos)} B | "
                f"{retardo:6.2f} | {jitter*1000:5.2f} | {estado_str}"
            )

            if len(bloques_rx) % 50 == 0:
                mod_str = "FM" if mod_rx == MOD_FM else "AM"
                print(f"  [RX-{mod_str}] {len(bloques_rx)} bloques | "
                      f"Jitter: {jitter*1000:.2f} ms | Retardo: {retardo:.2f} ms")

    except KeyboardInterrupt:
        print("\n[!] Interrupción por teclado.")
    finally:
        sock.close()

    if not bloques_rx:
        print("[ERROR] No se recibieron bloques de audio.")
        return {}

    # ── RECONSTRUCCIÓN Y DEMODULACIÓN ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" DEMODULACIÓN Y RECONSTRUCCIÓN")
    print("=" * 65)

    seq_min = min(bloques_rx.keys())
    seq_max = max(bloques_rx.keys())
    total_esp   = seq_max - seq_min + 1
    total_rx_ok = len(bloques_rx)
    total_perd  = total_esp - total_rx_ok
    pct_perd    = total_perd / total_esp * 100 if total_esp > 0 else 0.0

    mod_nombre = "FM" if mod_rx == MOD_FM else "AM"
    fc_usado   = fc if fc is not None else (FC_FM_DEFAULT if mod_rx == MOD_FM else FC_AM_DEFAULT)

    print(f"[*] Modo detectado       : {mod_nombre}")
    print(f"[*] Frecuencia portadora : {fc_usado} Hz")
    print(f"[*] Bloques esperados    : {total_esp}")
    print(f"[*] Bloques recibidos    : {total_rx_ok}")
    print(f"[*] Pérdidas detectadas  : {total_perd} ({pct_perd:.2f} %)")
    print(f"[*] Desórdenes           : {desordenados}")
    print(f"[*] Duplicados           : {duplicados}")

    # Demodular bloque a bloque
    offset_am = 0
    audio_recuperado = []

    bytes_por_bloque = tam_bloque_rx * 2  # int16

    for s in range(seq_min, seq_max + 1):
        if s in bloques_rx:
            muestras_mod, _ = bloques_rx[s]
        else:
            # PLC: repetir último bloque conocido (silencio si no hay anterior)
            if audio_recuperado:
                muestras_mod = audio_recuperado[-1].copy() * 0.5  # atenuado
            else:
                muestras_mod = np.zeros(tam_bloque_rx, dtype=np.float32)

        if mod_rx == MOD_AM:
            bloque_dem, offset_am = demodular_am_bloque(
                muestras_mod, fc=fc_usado, ka=ka, fs=fs_rx, sample_offset=offset_am
            )
        else:
            bloque_dem = demodular_fm_bloque(
                muestras_mod, fc=fc_usado, kf=kf, fs=fs_rx
            )

        audio_recuperado.append(bloque_dem)

    señal_recuperada = np.concatenate(audio_recuperado)
    duracion_rec = len(señal_recuperada) / fs_rx
    guardar_wav(salida_wav, señal_recuperada, fs_rx, canales_rx)
    print(f"[OK] Audio recuperado guardado: '{salida_wav}'")
    print(f"     Duración: {duracion_rec:.2f} s | Fs: {fs_rx} Hz")

    # SNR si se dispone del original
    snr_info = {}
    if ruta_original and os.path.isfile(ruta_original):
        try:
            with wave.open(ruta_original, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            orig = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            snr_info = calcular_snr(orig, señal_recuperada)
            print(f"[+] SNR {mod_nombre}: {snr_info['SNR_dB']:+.1f} dB | MSE: {snr_info['MSE']:.6f}")
        except Exception as e:
            print(f"[AVISO] No se pudo calcular SNR: {e}")

    # ── LOG Y RESUMEN ──────────────────────────────────────────────────────────
    t_total = (t_ultimo - t_primero) if (t_primero and t_ultimo) else 0.0
    bytes_rx = sum(len(bloques_rx[s][0]) * 2 + HEADER_SIZE for s in bloques_rx)
    kbps_rx  = bytes_rx * 8 / (t_total * 1000) if t_total > 0 else 0

    resumen = [
        "# " + "=" * 60,
        f"# RESUMEN RX MODULADO — {mod_nombre}",
        f"# Bloques recibidos   : {total_rx_ok}/{total_esp}",
        f"# Pérdidas            : {total_perd} ({pct_perd:.2f} %)",
        f"# Duplicados          : {duplicados}",
        f"# Desórdenes          : {desordenados}",
        f"# Jitter final        : {jitter*1000:.2f} ms",
        f"# Tasa datos recibida : {kbps_rx:.1f} kbps",
        f"# Duración recuperada : {duracion_rec:.3f} s",
    ]
    if snr_info:
        resumen.append(f"# SNR {mod_nombre}           : {snr_info['SNR_dB']:+.1f} dB")
    resumen.append("# " + "=" * 60)
    lineas_log.extend(resumen)

    with open(archivo_log, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_log) + "\n")
    print(f"[OK] Log guardado: '{archivo_log}'")

    return {
        "modo": mod_nombre,
        "esperados": total_esp,
        "recibidos": total_rx_ok,
        "perdidos": total_perd,
        "pct_perdida": pct_perd,
        "jitter_ms": jitter * 1000.0,
        "kbps": kbps_rx,
        "snr_db": snr_info.get("SNR_dB", None),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receptor UDP Modulado SDR — Fase 5")
    parser.add_argument("--ip",       default="0.0.0.0")
    parser.add_argument("--puerto",   type=int, default=5006)
    parser.add_argument("--salida",   default="audio_fase5_recuperado.wav")
    parser.add_argument("--fc",       type=float, default=None)
    parser.add_argument("--ka",       type=float, default=KA_DEFAULT)
    parser.add_argument("--kf",       type=float, default=KF_DEFAULT)
    parser.add_argument("--timeout",  type=float, default=2.5)
    parser.add_argument("--log",      default="log_fase5_rx.txt")
    parser.add_argument("--original", default=None, help="WAV original para calcular SNR")
    args = parser.parse_args()

    recibir_modulado(
        ip_escucha=args.ip,
        puerto=args.puerto,
        salida_wav=args.salida,
        fc=args.fc,
        ka=args.ka,
        kf=args.kf,
        timeout=args.timeout,
        archivo_log=args.log,
        ruta_original=args.original,
    )
