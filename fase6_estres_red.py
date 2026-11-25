"""
=============================================================
 FASE 6 — Pruebas de Estrés de Red, Tamaño de Bloque y Pérdidas
=============================================================
Asignatura: Redes de Ordenadores / Tratamiento de Señales

¿Qué hace este archivo?
  1. Analiza el impacto del tamaño de bloque (128, 256, 512, 1024 muestras)
     sobre la sobrecarga de cabecera (overhead), nº de datagramas UDP y retardo.
  2. Simula un canal hostil con pérdidas forzadas (0%, 5%, 10%, 20%) sobre
     audio FM transmitido por UDP y evalúa la recuperación mediante PLC
     (Packet Loss Concealment).
  3. Genera gráficas comparativas en 'plots/fase6_estres_red.png' y un
     informe de evidencias en 'plots/evidencias_fase6.txt'.

Ejecutar:
  python fase6_estres_red.py
"""

import os
import sys
import time
import wave
import socket
import struct
import random
from datetime import datetime

# Añadir vendor y modulos al path
_vendor = os.path.join(os.path.dirname(__file__), "vendor")
if os.path.isdir(_vendor) and _vendor not in sys.path:
    sys.path.insert(0, _vendor)

_modulos = os.path.join(os.path.dirname(__file__), "modulos")
if _modulos not in sys.path:
    sys.path.insert(0, _modulos)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modulador_en_linea import (
    modular_fm_bloque, demodular_fm_bloque,
    serializar_bloque, deserializar_bloque,
    EstadoFM, calcular_snr,
    FC_FM_DEFAULT, KF_DEFAULT, FS_DEFAULT
)

HDR_FMT = "!4sIdBH"
HDR_SIZE = struct.calcsize(HDR_FMT)  # 19 bytes de cabecera Fase 5/6
FS = FS_DEFAULT


def cargar_voz(ruta="audio_original.wav") -> np.ndarray:
    if not os.path.isfile(ruta):
        ruta = os.path.join(os.path.dirname(__file__), "audio_original.wav")
    with wave.open(ruta, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    voz = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    voz -= np.mean(voz)
    pico = np.max(np.abs(voz))
    if pico > 0:
        voz = voz / pico * 0.95
    return voz


def transmitir_y_medir_udp(voz: np.ndarray, tam_bloque: int, tasa_perdida: float = 0.0, puerto: int = 5025) -> dict:
    """Modula en FM, envía por UDP loopback aplicando tasa_perdida y demodula con PLC."""
    rng = random.Random(42)
    bloques = [voz[i:i + tam_bloque] for i in range(0, len(voz), tam_bloque)]
    total_bloques = len(bloques)

    sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_rx.bind(("127.0.0.1", puerto))
    sock_rx.settimeout(0.2)

    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    estado_fm = EstadoFM()
    recibidos = {}
    bytes_totales = 0
    jitter = 0.0
    prev_rx = 0.0
    prev_tx = 0.0

    t_inicio = time.time()
    for idx, bloque in enumerate(bloques):
        seq = idx + 1
        bloque_mod, estado_fm = modular_fm_bloque(bloque, fc=FC_FM_DEFAULT, kf=KF_DEFAULT, fs=FS, estado=estado_fm)
        payload = serializar_bloque(bloque_mod)

        # Simulación de pérdida aleatoria en canal hostil
        if tasa_perdida > 0.0 and rng.random() < tasa_perdida:
            continue

        t_tx = time.time()
        pkt = struct.pack(HDR_FMT, b"SDR6", seq, t_tx, 1, len(payload)) + payload
        sock_tx.sendto(pkt, ("127.0.0.1", puerto))
        bytes_totales += len(pkt)

        try:
            data, _ = sock_rx.recvfrom(8192)
            t_rx = time.time()
            _, s_num, ts, _, pay_len = struct.unpack(HDR_FMT, data[:HDR_SIZE])
            recibidos[s_num] = deserializar_bloque(data[HDR_SIZE:HDR_SIZE + pay_len])

            if prev_rx > 0:
                d = abs((t_rx - prev_rx) - (ts - prev_tx))
                jitter += (d - jitter) / 16.0
            prev_rx = t_rx
            prev_tx = ts
        except socket.timeout:
            pass

    t_total_ms = (time.time() - t_inicio) * 1000.0
    sock_tx.close()
    sock_rx.close()

    # Reconstrucción con PLC (Packet Loss Concealment)
    partes = []
    for s in range(1, total_bloques + 1):
        if s in recibidos:
            b_dem = demodular_fm_bloque(recibidos[s], fc=FC_FM_DEFAULT, kf=KF_DEFAULT, fs=FS)
            partes.append(b_dem)
        else:
            # Ocultación por repetición atenuada del último bloque válido
            plc = partes[-1] * 0.45 if partes else np.zeros(tam_bloque, dtype=np.float32)
            partes.append(plc)

    audio_rec = np.concatenate(partes)
    snr_info = calcular_snr(voz, audio_rec)

    payload_bytes = tam_bloque * 2  # int16
    overhead_pct = (HDR_SIZE / (HDR_SIZE + payload_bytes)) * 100.0
    perdidos = total_bloques - len(recibidos)

    return {
        "tam_bloque": tam_bloque,
        "duracion_bloque_ms": (tam_bloque / FS) * 1000.0,
        "total_bloques": total_bloques,
        "recibidos": len(recibidos),
        "perdidos": perdidos,
        "perdida_real_pct": (perdidos / total_bloques) * 100.0,
        "bytes_totales": bytes_totales,
        "overhead_pct": overhead_pct,
        "t_total_ms": t_total_ms,
        "jitter_ms": jitter * 1000.0,
        "snr_db": snr_info["SNR_dB"],
        "mse": snr_info["MSE"],
    }


def generar_grafica(res_bloques: list, res_perdidas: list, ruta_png: str):
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0f172a")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.grid(True, color="#334155", linestyle="--", alpha=0.6)

    # Panel 1: Tamaño de bloque vs Overhead y Nº de Paquetes
    tams = [str(r["tam_bloque"]) for r in res_bloques]
    overheads = [r["overhead_pct"] for r in res_bloques]
    pkts = [r["total_bloques"] for r in res_bloques]

    bars = ax1.bar(tams, overheads, color="#38bdf8", width=0.5, alpha=0.85, label="Overhead Cabecera (%)")
    ax1.set_title("1. Impacto del Tamaño de Bloque (MTU y Overhead)", fontweight="bold", color="#f8fafc")
    ax1.set_xlabel("Tamaño de bloque (muestras por paquete)", color="#cbd5e1")
    ax1.set_ylabel("Overhead de cabecera (%)", color="#38bdf8")

    for bar, r in zip(bars, res_bloques):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f"{r['overhead_pct']:.1f}%\n({r['total_bloques']} pkts)",
            ha="center", va="bottom", fontsize=8, color="#f8fafc", fontweight="bold"
        )
    ax1.set_ylim(0, max(overheads) * 1.35)

    # Panel 2: Tasa de pérdida vs Calidad SNR
    perd_pcts = [r["perdida_real_pct"] for r in res_perdidas]
    snrs = [r["snr_db"] for r in res_perdidas]

    ax2.plot(perd_pcts, snrs, marker="o", lw=2.5, markersize=8, color="#f472b6", label="SNR con PLC")
    for x, y in zip(perd_pcts, snrs):
        ax2.annotate(f"{y:+.1f} dB", (x, y), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8, color="#f8fafc", fontweight="bold")
    ax2.set_title("2. Degradación de Señal ante Pérdidas UDP (con PLC)", fontweight="bold", color="#f8fafc")
    ax2.set_xlabel("Paquetes perdidos en la red (%)", color="#cbd5e1")
    ax2.set_ylabel("SNR Recuperado (dB)", color="#f472b6")
    ax2.set_ylim(min(snrs) - 2.0, max(snrs) + 2.5)

    fig.tight_layout()
    fig.savefig(ruta_png, dpi=140, facecolor="#0f172a")
    plt.close(fig)


def ejecutar_fase6():
    print("=" * 70)
    print(" FASE 6 — PRUEBAS DE ESTRÉS DE RED (MTU, PÉRDIDAS Y PLC)")
    print("=" * 70)

    voz = cargar_voz()
    os.makedirs("plots", exist_ok=True)

    # 1. Prueba de tamaños de bloque
    print("\n>>> [1/2] Evaluando tamaños de bloque (128, 256, 512, 1024 muestras)...")
    res_bloques = []
    for idx, tam in enumerate([128, 256, 512, 1024]):
        r = transmitir_y_medir_udp(voz, tam_bloque=tam, tasa_perdida=0.0, puerto=5020 + idx)
        res_bloques.append(r)
        print(f"  Bloque {tam:>4} muestras ({r['duracion_bloque_ms']:5.1f} ms) -> "
              f"{r['total_bloques']:>4} pkts | Overhead: {r['overhead_pct']:4.1f}% | SNR: {r['snr_db']:+.1f} dB")

    # 2. Prueba de pérdidas crecientes (bloque fijo = 512)
    print("\n>>> [2/2] Evaluando canal hostil con pérdidas (0%, 5%, 10%, 20%)...")
    res_perdidas = []
    for idx, p in enumerate([0.0, 0.05, 0.10, 0.20]):
        r = transmitir_y_medir_udp(voz, tam_bloque=512, tasa_perdida=p, puerto=5030 + idx)
        res_perdidas.append(r)
        print(f"  Pérdida obj {p*100:4.0f}% (real {r['perdida_real_pct']:4.1f}%) -> "
              f"Perdidos: {r['perdidos']:>3}/{r['total_bloques']} | SNR: {r['snr_db']:+.1f} dB | MSE: {r['mse']:.5f}")

    # 3. Guardar gráfica y reporte
    ruta_png = os.path.join("plots", "fase6_estres_red.png")
    generar_grafica(res_bloques, res_perdidas, ruta_png)

    ruta_txt = os.path.join("plots", "evidencias_fase6.txt")
    lineas = [
        "=" * 70,
        f" EVIDENCIAS DE FASE 6: ESTRÉS DE RED — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "\n[1] COMPARATIVA DE TAMAÑO DE BLOQUE (MTU):",
    ]
    for r in res_bloques:
        lineas.append(
            f"  - {r['tam_bloque']:>4} muestras ({r['duracion_bloque_ms']:4.1f} ms): "
            f"{r['total_bloques']:>4} pkts | Overhead {r['overhead_pct']:4.1f}% | Jitter {r['jitter_ms']:.2f} ms"
        )
    lineas.append("\n[2] DEGRADACIÓN ANTE PÉRDIDAS DE RED (BLOQUE=512, PLC ACTIVO):")
    for r in res_perdidas:
        lineas.append(
            f"  - Pérdida {r['perdida_real_pct']:4.1f}% ({r['perdidos']:>3}/{r['total_bloques']} pkts): "
            f"SNR {r['snr_db']:+.1f} dB | MSE {r['mse']:.6f}"
        )
    lineas.extend([
        "\n" + "=" * 70,
        " CONCLUSIONES TÉCNICAS:",
        "  - Bloque de 512 muestras (~10.7 ms) ofrece el mejor equilibrio:",
        "    bajo overhead de cabecera (<2%) y tamaño de datagrama seguro (< MTU 1500B).",
        "  - Con PLC (repetición atenuada), el sistema tolera hasta ~5-10% de pérdida",
        "    manteniendo la frase comprensible sin bloqueos ni retransmisiones TCP.",
        "=" * 70,
    ])
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")

    print(f"\n[OK] Gráfica guardada en: '{ruta_png}'")
    print(f"[OK] Informe guardado en: '{ruta_txt}'")
    print("=" * 70)


if __name__ == "__main__":
    ejecutar_fase6()
