"""
=============================================================
 EMISOR UDP MODULADO — SDR Fase 5
=============================================================
Asignatura: Redes de Ordenadores / Tratamiento de Señales
Fase 5: Integración Modulación AM/FM + Transporte UDP

¿Qué hace este script?
  1. Lee un archivo de audio WAV (voz o música, mono).
  2. Divide el audio en bloques de tamaño controlado.
  3. Modula cada bloque en AM o FM EN TIEMPO REAL (bloque a bloque)
     propagando correctamente la fase entre bloques consecutivos (FM).
  4. Serializa las muestras moduladas a PCM int16 y empaqueta
     con la cabecera del protocolo SDR (campo modulation = AM o FM).
  5. Envía los datagramas UDP al receptor al ritmo natural del audio.
  6. Genera log TX con los parámetros de modulación de cada paquete.

Diferencia clave con transmisor_udp.py (Fase 4):
  Fase 4 → modulation = MOD_RAW (muestras tal cual)
  Fase 5 → modulation = MOD_AM o MOD_FM (ya moduladas antes de empaquetar)
"""

import argparse
import os
import socket
import struct
import sys
import time
import wave
from datetime import datetime
from typing import Optional

# Añadir vendor al path si existe (numpy/scipy locales)
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
    DTYPE_INT16,
    HEADER_SIZE,
)
from modulador_en_linea import (
    modular_am_bloque,
    modular_fm_bloque,
    serializar_bloque,
    EstadoFM,
    FC_AM_DEFAULT, FC_FM_DEFAULT,
    KA_DEFAULT, KF_DEFAULT,
    FS_DEFAULT,
)


def leer_wav_float32(ruta: str):
    """Lee WAV y devuelve (señal float32, sample_rate, canales)."""
    with wave.open(ruta, "rb") as wf:
        canales = wf.getnchannels()
        ancho   = wf.getsampwidth()
        fs      = wf.getframerate()
        n       = wf.getnframes()
        raw     = wf.readframes(n)

    tipos = {1: np.int8, 2: np.int16, 4: np.int32}
    muestras = np.frombuffer(raw, dtype=tipos.get(ancho, np.int16))
    if canales == 2:
        muestras = muestras.reshape(-1, 2).mean(axis=1).astype(muestras.dtype)
        canales = 1

    señal = muestras.astype(np.float32) / np.iinfo(tipos.get(ancho, np.int16)).max
    señal = señal - np.mean(señal)
    pico = np.max(np.abs(señal))
    if pico > 0:
        señal = señal / pico * 0.95

    return señal, fs, canales


def emitir_modulado(
    ruta_audio: str,
    modo: str = "fm",               # "am" o "fm"
    ip_destino: str = "127.0.0.1",
    puerto: int = 5006,
    tam_bloque: int = 512,
    fc: Optional[float] = None,
    ka: float = KA_DEFAULT,
    kf: float = KF_DEFAULT,
    tiempo_real: bool = True,
    descartar_seq: int = -1,
    archivo_log: str = "log_fase5_tx.txt",
):
    modo = modo.lower().strip()
    assert modo in ("am", "fm"), f"Modo '{modo}' inválido. Usa 'am' o 'fm'."

    if fc is None:
        fc = FC_FM_DEFAULT if modo == "fm" else FC_AM_DEFAULT

    mod_code = MOD_FM if modo == "fm" else MOD_AM
    mod_nombre = "FM" if modo == "fm" else "AM"

    print("=" * 65)
    print(f" EMISOR UDP MODULADO — SDR Fase 5 ({mod_nombre})")
    print("=" * 65)

    señal, fs, canales = leer_wav_float32(ruta_audio)
    duracion = len(señal) / fs

    print(f"[*] Archivo         : {ruta_audio}")
    print(f"[*] Fs / Canales    : {fs} Hz / {canales} canal(es)")
    print(f"[*] Duración        : {duracion:.2f} s ({len(señal):,} muestras)")
    print(f"[*] Modo            : {mod_nombre} | fc={fc} Hz" +
          (f" | ka={ka}" if modo == "am" else f" | kf={kf} Hz"))
    print(f"[*] Destino         : {ip_destino}:{puerto}")
    print(f"[*] Bloque          : {tam_bloque} muestras (~{tam_bloque/fs*1000:.1f} ms)")

    # Dividir en bloques
    bloques = [señal[i:i+tam_bloque] for i in range(0, len(señal), tam_bloque)]
    cadencia_s = tam_bloque / fs
    print(f"[*] Total bloques   : {len(bloques)}")
    if descartar_seq >= 0:
        print(f"[!] ROBUSTEZ: se descartará intencionadamente el bloque #{descartar_seq}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    lineas_log = [
        f"# LOG TX MODULADO (Fase 5) — {datetime.now().isoformat()}",
        f"# Audio: {ruta_audio} | Modo: {mod_nombre} | fc={fc} Hz | fs={fs} Hz",
        f"# Destino: {ip_destino}:{puerto} | Bloque: {tam_bloque} muestras",
        "# " + "-" * 60,
        f"# SEQ | TIMESTAMP | BYTES | ESTADO | DETALLE",
    ]

    # Paquete START
    hdr_start = CabeceraSDR(sequence=0, samples=0, sample_rate=fs,
                             channels=canales, modulation=mod_code,
                             dtype=DTYPE_INT16, flags=FLAG_START)
    sock.sendto(hdr_start.pack(b""), (ip_destino, puerto))
    lineas_log.append(f"START | {time.time():.4f} | {HEADER_SIZE} | ENVIADO | Inicio {mod_nombre}")

    print(f"\n[>] Transmitiendo {mod_nombre} en tiempo real...")
    t_inicio = time.time()
    bytes_total = 0
    enviados = 0

    # Estado de fase FM (solo se usa si modo == "fm")
    estado_fm = EstadoFM()
    offset_am = 0  # Contador de muestras para coherencia AM

    for idx, bloque in enumerate(bloques):
        seq = idx + 1

        # ── MODULACIÓN DEL BLOQUE ──────────────────────────────
        if modo == "am":
            bloque_mod, offset_am = modular_am_bloque(
                bloque, fc=fc, ka=ka, fs=fs, sample_offset=offset_am
            )
        else:
            bloque_mod, estado_fm = modular_fm_bloque(
                bloque, fc=fc, kf=kf, fs=fs, estado=estado_fm
            )

        payload = serializar_bloque(bloque_mod)

        # ── PRUEBA DE ROBUSTEZ ──────────────────────────────────
        if descartar_seq >= 0 and seq == descartar_seq:
            print(f"  [ROBUSTEZ] Bloque #{seq:04d} DESCARTADO (simulación pérdida)")
            lineas_log.append(f"{seq:05d} | {time.time():.4f} | {len(payload)} | DESCARTADO | Simulación de fallo")
            if tiempo_real:
                time.sleep(cadencia_s * 0.95)
            else:
                time.sleep(0.0005)
            continue

        # ── ENVÍO ───────────────────────────────────────────────
        cabecera = CabeceraSDR(
            sequence=seq, samples=len(bloque_mod),
            sample_rate=fs, channels=canales,
            modulation=mod_code, dtype=DTYPE_INT16, flags=FLAG_DATA,
        )
        paquete = cabecera.pack(payload)
        sock.sendto(paquete, (ip_destino, puerto))

        bytes_total += len(paquete)
        enviados += 1

        lineas_log.append(
            f"{seq:05d} | {cabecera.timestamp:.4f} | {len(paquete)} B | ENVIADO | "
            f"{mod_nombre} {len(bloque_mod)} muestras"
        )

        if seq % 50 == 0 or seq == len(bloques):
            pct = seq / len(bloques) * 100
            print(f"  [TX-{mod_nombre}] #{seq:04d}/{len(bloques)} ({pct:5.1f}%) "
                  f"| {len(paquete)} B -> {ip_destino}:{puerto}")

        if tiempo_real:
            time.sleep(cadencia_s * 0.95)
        else:
            time.sleep(0.0005)

    # Paquete END (enviado varias veces para fiabilidad)
    hdr_end = CabeceraSDR(sequence=len(bloques)+1, samples=0, sample_rate=fs,
                           channels=canales, modulation=mod_code,
                           dtype=DTYPE_INT16, flags=FLAG_END)
    for _ in range(3):
        sock.sendto(hdr_end.pack(b""), (ip_destino, puerto))
        time.sleep(0.01)
    lineas_log.append(f"END   | {time.time():.4f} | {HEADER_SIZE} | ENVIADO | Fin de sesión")

    t_total = time.time() - t_inicio
    sock.close()

    kbps = bytes_total * 8 / (t_total * 1000) if t_total > 0 else 0
    resumen = [
        "# " + "=" * 60,
        f"# RESUMEN TX {mod_nombre}",
        f"# Bloques enviados : {enviados}/{len(bloques)}",
        f"# Bytes totales    : {bytes_total:,} B",
        f"# Tiempo           : {t_total:.3f} s",
        f"# Tasa efectiva    : {kbps:.1f} kbps",
        "# " + "=" * 60,
    ]
    lineas_log.extend(resumen)

    with open(archivo_log, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_log) + "\n")

    print("\n" + "=" * 65)
    print(f" RESUMEN EMISOR {mod_nombre}")
    print(f"  - Bloques enviados : {enviados}")
    print(f"  - Bytes totales    : {bytes_total:,}")
    print(f"  - Tiempo total     : {t_total:.2f} s")
    print(f"  - Tasa efectiva    : {kbps:.1f} kbps")
    print(f"  - Log guardado     : {archivo_log}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emisor UDP Modulado SDR — Fase 5")
    parser.add_argument("--archivo",   default="audio_original.wav")
    parser.add_argument("--modo",      choices=["am", "fm"], default="fm")
    parser.add_argument("--ip",        default="127.0.0.1")
    parser.add_argument("--puerto",    type=int, default=5006)
    parser.add_argument("--fc",        type=float, default=None)
    parser.add_argument("--ka",        type=float, default=KA_DEFAULT)
    parser.add_argument("--kf",        type=float, default=KF_DEFAULT)
    parser.add_argument("--tam-bloque",type=int, default=512)
    parser.add_argument("--rapido",    action="store_true")
    parser.add_argument("--descartar", type=int, default=-1)
    parser.add_argument("--log",       default="log_fase5_tx.txt")
    args = parser.parse_args()

    ruta = args.archivo
    if not os.path.isfile(ruta):
        alt = os.path.join(os.path.dirname(__file__), ruta)
        if os.path.isfile(alt):
            ruta = alt

    emitir_modulado(
        ruta_audio=ruta,
        modo=args.modo,
        ip_destino=args.ip,
        puerto=args.puerto,
        tam_bloque=args.tam_bloque,
        fc=args.fc,
        ka=args.ka,
        kf=args.kf,
        tiempo_real=not args.rapido,
        descartar_seq=args.descartar,
        archivo_log=args.log,
    )
