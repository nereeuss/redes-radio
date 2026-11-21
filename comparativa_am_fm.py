"""
=============================================================
 COMPARATIVA AM vs FM — Evidencia visual para la memoria
=============================================================

¿Qué hace este archivo?
  Genera 3 figuras comparativas entre AM y FM:

  Figura 1 — Espectro: cómo se ve cada modulación en frecuencia
  Figura 2 — Efecto del ruido: por qué FM resiste mejor
  Figura 3 — Tabla de métricas: números que justifican la elección

Ejecutar:
  python comparativa_am_fm.py

Autor: Equipo SDR — Comparativa Fase 2
"""

import numpy as np
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    raise SystemExit("Instala matplotlib: pip install matplotlib")

try:
    from scipy.signal import butter, filtfilt, hilbert
except ImportError:
    raise SystemExit("Instala scipy: pip install scipy")


# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

FS      = 48_000   # Hz
FC_AM   =  8_000   # Hz — portadora AM
FC_FM   = 10_000   # Hz — portadora FM (más alta porque FM ocupa más BW)
KA      = 0.9      # índice modulación AM
KF      = 2_000    # desviación FM en Hz
DURACION = 3.0     # segundos de señal de prueba
FREQ_VOZ = 300.0   # Hz — fundamental del tono sintético


# ─────────────────────────────────────────────────────────────────────────────
# GENERAR SEÑAL DE VOZ SINTÉTICA
# ─────────────────────────────────────────────────────────────────────────────

def generar_voz(duracion=DURACION, freq=FREQ_VOZ, fs=FS) -> np.ndarray:
    t = np.linspace(0, duracion, int(duracion * fs), endpoint=False)
    amplitudes = [1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]
    señal = sum(a * np.sin(2 * np.pi * freq * (i+1) * t)
                for i, a in enumerate(amplitudes))
    n_fade = int(0.05 * fs)
    señal[:n_fade]  *= np.linspace(0, 1, n_fade)
    señal[-n_fade:] *= np.linspace(1, 0, n_fade)
    señal -= np.mean(señal)
    pico = np.max(np.abs(señal))
    return (señal / pico * 0.95).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODULACIÓN Y DEMODULACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def modular_am(señal, fs=FS, fc=FC_AM, ka=KA):
    t = np.arange(len(señal)) / fs
    s = (1.0 + ka * señal) * np.cos(2 * np.pi * fc * t)
    return (s / np.max(np.abs(s)) * 0.95).astype(np.float32)

def demodular_am(s_am, fs=FS, fc=FC_AM, corte=4000.0):
    t = np.arange(len(s_am)) / fs
    mezclado = s_am * np.cos(2 * np.pi * fc * t)
    b, a = butter(5, corte / (fs/2), btype='low')
    demod = filtfilt(b, a, mezclado)
    demod -= np.mean(demod)
    pico = np.max(np.abs(demod))
    return (demod / pico * 0.95).astype(np.float32) if pico > 0 else demod

def modular_fm(señal, fs=FS, fc=FC_FM, kf=KF):
    t = np.arange(len(señal)) / fs
    integral = np.cumsum(señal) / fs
    fase = 2 * np.pi * fc * t + 2 * np.pi * kf * integral
    return np.cos(fase).astype(np.float32)

def demodular_fm(s_fm, fs=FS, fc=FC_FM, kf=KF, corte=4000.0):
    analitica = hilbert(s_fm)
    fase_inst = np.unwrap(np.angle(analitica))
    freq_inst = np.diff(fase_inst) * fs / (2 * np.pi)
    demod = (freq_inst - fc) / kf
    demod = np.append(demod, 0.0)
    b, a = butter(5, corte / (fs/2), btype='low')
    demod = filtfilt(b, a, demod)
    demod -= np.mean(demod)
    pico = np.max(np.abs(demod))
    return (demod / pico * 0.95).astype(np.float32) if pico > 0 else demod

def añadir_ruido(señal, snr_db):
    """Añade ruido gaussiano para simular un canal imperfecto."""
    potencia_señal = np.mean(señal ** 2)
    potencia_ruido = potencia_señal / (10 ** (snr_db / 10))
    ruido = np.random.normal(0, np.sqrt(potencia_ruido), len(señal))
    return (señal + ruido).astype(np.float32)

def snr_db(original, recuperada):
    n = min(len(original), len(recuperada))
    pot_señal = np.mean(original[:n] ** 2)
    pot_error = np.mean((original[:n] - recuperada[:n]) ** 2)
    return 10 * np.log10(pot_señal / (pot_error + 1e-12))

def espectro(señal, fs=FS):
    N = len(señal)
    mags = 20 * np.log10(np.abs(np.fft.rfft(señal)) / N + 1e-12)
    freqs = np.fft.rfftfreq(N, 1/fs) / 1000
    return freqs, mags


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 1 — Comparativa de espectros AM vs FM
# ─────────────────────────────────────────────────────────────────────────────

def figura_espectros(voz, s_am, s_fm):
    """
    Muestra 3 espectros en columnas:
    - Voz original
    - Señal AM modulada
    - Señal FM modulada

    ¿Qué justifica?
      Que AM y FM trasladan la voz a zonas distintas del espectro,
      y que FM ocupa más ancho de banda (efecto Regla de Carson).
    """
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Figura 1 — Comparativa de espectros AM vs FM",
                 fontsize=13, fontweight='bold')

    datos = [
        (voz,  "Voz ORIGINAL",    'steelblue',   None,    None),
        (s_am, "Señal AM modulada", 'darkorange', FC_AM,   "fc AM"),
        (s_fm, "Señal FM modulada", 'crimson',    FC_FM,   "fc FM"),
    ]

    for ax, (sig, titulo, color, fc, fc_label) in zip(ejes, datos):
        freqs, mags = espectro(sig)
        ax.plot(freqs, mags, color=color, linewidth=0.8, label=titulo)
        ax.axvline(4, color='gray', linestyle=':', linewidth=1, label='4 kHz (máx voz)')
        if fc:
            ax.axvline(fc/1000, color='black', linestyle='--',
                       linewidth=1.2, label=fc_label)
            # Sombrear ancho de banda
            if 'AM' in titulo:
                bw_low  = (fc - 4000) / 1000
                bw_high = (fc + 4000) / 1000
                label_bw = f"BW AM = {2*4000} Hz"
            else:
                bw_low  = (fc - (KF+4000)) / 1000
                bw_high = (fc + (KF+4000)) / 1000
                label_bw = f"BW FM = {2*(KF+4000)} Hz"
            ax.axvspan(bw_low, bw_high, alpha=0.12, color=color, label=label_bw)

        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel("Frecuencia [kHz]")
        ax.set_ylabel("Magnitud [dB]")
        ax.set_xlim(0, FS/2/1000)
        ax.set_ylim(-90, 5)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.35)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/comparativa_espectros.png", dpi=150, bbox_inches='tight')
    print("[OK] Figura 1 guardada: plots/comparativa_espectros.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 2 — Efecto del ruido sobre AM y FM
# ─────────────────────────────────────────────────────────────────────────────

def figura_ruido(voz):
    """
    Añade el mismo nivel de ruido a la señal AM y FM,
    demodula ambas y compara la calidad de la recuperación.

    ¿Qué justifica?
      Que FM resiste mejor el ruido porque la información
      está en la frecuencia (que el ruido no distorsiona fácilmente)
      y no en la amplitud (que el ruido sí distorsiona directamente).
    """
    niveles_ruido = [30, 20, 15, 10]   # SNR del canal en dB

    fig, ejes = plt.subplots(len(niveles_ruido), 2,
                             figsize=(14, 3.5 * len(niveles_ruido)))
    fig.suptitle("Figura 2 — Efecto del ruido: AM (izq.) vs FM (der.)",
                 fontsize=13, fontweight='bold')

    s_am_limpio = modular_am(voz)
    s_fm_limpio = modular_fm(voz)

    for fila, snr_canal in enumerate(niveles_ruido):
        # Añadir el mismo ruido a ambas señales moduladas
        s_am_ruidosa = añadir_ruido(s_am_limpio, snr_canal)
        s_fm_ruidosa = añadir_ruido(s_fm_limpio, snr_canal)

        # Demodular con ruido
        rec_am = demodular_am(s_am_ruidosa)
        rec_fm = demodular_fm(s_fm_ruidosa)

        # Calcular SNR de la voz recuperada
        snr_am = snr_db(voz, rec_am)
        snr_fm = snr_db(voz, rec_fm)

        n_frag = int(0.05 * FS)
        t_ms   = np.arange(n_frag) / FS * 1000

        for col, (rec, mod, color, snr_rec) in enumerate([
            (rec_am, "AM", 'darkorange', snr_am),
            (rec_fm, "FM", 'crimson',    snr_fm),
        ]):
            ax = ejes[fila, col]
            ax.plot(t_ms, voz[:n_frag],  color='steelblue',
                    linewidth=1.2, alpha=0.6, label='Original')
            ax.plot(t_ms, rec[:n_frag],  color=color,
                    linewidth=0.8, alpha=0.9, label=f'{mod} recuperada')
            ax.set_title(
                f"Canal con ruido SNR={snr_canal} dB — {mod} recuperada "
                f"(SNR voz={snr_rec:.1f} dB)",
                fontsize=9
            )
            ax.set_xlabel("Tiempo [ms]")
            ax.set_ylabel("Amplitud")
            ax.set_ylim(-1.1, 1.1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/comparativa_ruido.png", dpi=150, bbox_inches='tight')
    print("[OK] Figura 2 guardada: plots/comparativa_ruido.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 3 — Curva SNR recuperado vs SNR canal (AM vs FM)
# ─────────────────────────────────────────────────────────────────────────────

def figura_curva_snr(voz):
    """
    Barre niveles de ruido del canal de 5 a 40 dB y mide el SNR
    de la voz recuperada para AM y FM.

    ¿Qué justifica?
      Que a igualdad de condiciones de canal, FM siempre
      recupera la voz con más calidad (mayor SNR) que AM.
      La diferencia se hace mayor cuanto peor es el canal.
    """
    s_am = modular_am(voz)
    s_fm = modular_fm(voz)

    snrs_canal = np.arange(5, 41, 2)
    snrs_am, snrs_fm = [], []

    for snr_canal in snrs_canal:
        rec_am = demodular_am(añadir_ruido(s_am, snr_canal))
        rec_fm = demodular_fm(añadir_ruido(s_fm, snr_canal))
        snrs_am.append(snr_db(voz, rec_am))
        snrs_fm.append(snr_db(voz, rec_fm))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(snrs_canal, snrs_am, 'o-', color='darkorange',
            linewidth=2, markersize=5, label='AM recuperada')
    ax.plot(snrs_canal, snrs_fm, 's-', color='crimson',
            linewidth=2, markersize=5, label='FM recuperada')
    ax.axhline(20, color='gray', linestyle='--', linewidth=1,
               label='SNR mínimo aceptable (20 dB)')
    ax.fill_between(snrs_canal, snrs_am, snrs_fm,
                    alpha=0.1, color='crimson', label='Ventaja de FM')
    ax.set_title("Figura 3 — SNR de voz recuperada vs calidad del canal",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("SNR del canal [dB]  (mayor = canal más limpio)")
    ax.set_ylabel("SNR de la voz recuperada [dB]")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("plots/comparativa_snr_curva.png", dpi=150, bbox_inches='tight')
    print("[OK] Figura 3 guardada: plots/comparativa_snr_curva.png")
    plt.show()

    # Imprimir tabla de resultados
    print("\n── Tabla comparativa AM vs FM ──────────────────────────────")
    print(f"{'SNR canal':>12} {'SNR AM rec.':>14} {'SNR FM rec.':>14} {'Ventaja FM':>12}")
    print("─" * 56)
    for snr_c, snr_a, snr_f in zip(snrs_canal, snrs_am, snrs_fm):
        ventaja = snr_f - snr_a
        print(f"{snr_c:>10} dB {snr_a:>12.1f} dB {snr_f:>12.1f} dB {ventaja:>+10.1f} dB")
    print("─" * 56)


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  COMPARATIVA AM vs FM")
    print("=" * 60)

    print("\n[1/4] Generando señal de voz sintética...")
    voz = generar_voz()
    print(f"      {len(voz)} muestras, {DURACION} s, fs={FS} Hz")

    print("\n[2/4] Modulando AM y FM...")
    s_am = modular_am(voz)
    s_fm = modular_fm(voz)

    print("\n[3/4] Generando Figura 1 — Espectros...")
    figura_espectros(voz, s_am, s_fm)

    print("\n[4/4] Generando Figura 2 — Efecto del ruido...")
    figura_ruido(voz)

    print("\n[5/4] Generando Figura 3 — Curva SNR...")
    figura_curva_snr(voz)

    print("\n[OK] Comparativa completa.")
    print("     Gráficas guardadas en la carpeta 'plots/'")
    print("     Úsalas directamente en la memoria del proyecto.")
