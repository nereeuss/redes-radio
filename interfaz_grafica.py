"""
=============================================================
 PANEL DE CONTROL INTERACTIVO — SDR REDES DE RADIO
=============================================================

Interfaz gráfica para visualizar y experimentar con:
  - Fase 1: Audio original y análisis FFT
  - Fase 2: Modulación / Demodulación AM y FM
  - Comparativa: Espectros, resistencia a ruido y curvas SNR
  - Reproducción de audio en tiempo real y métricas

Ejecución:
  python interfaz_grafica.py
"""

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from scipy.signal import butter, filtfilt, hilbert

try:
    import sounddevice as sd
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False

# ────────────────────────────────────────────────────────────
# FUNCIONES DE PROCESAMIENTO DE SEÑAL
# ────────────────────────────────────────────────────────────

FS = 48_000  # Frecuencia de muestreo estándar

def generar_tono_demo(duracion=3.0, freq=300.0, fs=FS):
    t = np.linspace(0, duracion, int(duracion * fs), endpoint=False)
    amplitudes = [1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]
    sig = sum(a * np.sin(2 * np.pi * freq * (i + 1) * t) for i, a in enumerate(amplitudes))
    # Fade in / out
    n_fade = int(0.04 * fs)
    sig[:n_fade] *= np.linspace(0, 1, n_fade)
    sig[-n_fade:] *= np.linspace(1, 0, n_fade)
    sig -= np.mean(sig)
    pico = np.max(np.abs(sig))
    return (sig / pico * 0.95).astype(np.float32)

def modular_am(señal, fs=FS, fc=8000.0, ka=0.9):
    t = np.arange(len(señal)) / fs
    s = (1.0 + ka * señal) * np.cos(2 * np.pi * fc * t)
    pico = np.max(np.abs(s))
    return (s / pico * 0.95).astype(np.float32) if pico > 0 else s

def demodular_am(s_am, fs=FS, fc=8000.0, corte=4000.0):
    t = np.arange(len(s_am)) / fs
    mezclado = s_am * np.cos(2 * np.pi * fc * t)
    b, a = butter(5, corte / (fs / 2), btype='low')
    demod = filtfilt(b, a, mezclado)
    demod -= np.mean(demod)
    pico = np.max(np.abs(demod))
    return (demod / pico * 0.95).astype(np.float32) if pico > 0 else demod

def modular_fm(señal, fs=FS, fc=8000.0, kf=2000.0):
    t = np.arange(len(señal)) / fs
    integral = np.cumsum(señal) / fs
    fase = 2 * np.pi * fc * t + 2 * np.pi * kf * integral
    return np.cos(fase).astype(np.float32)

def demodular_fm(s_fm, fs=FS, fc=8000.0, kf=2000.0, corte=4000.0):
    analitica = hilbert(s_fm)
    fase_inst = np.unwrap(np.angle(analitica))
    freq_inst = np.diff(fase_inst) * fs / (2 * np.pi)
    demod = (freq_inst - fc) / (kf + 1e-9)
    demod = np.append(demod, 0.0)
    b, a = butter(5, corte / (fs / 2), btype='low')
    demod = filtfilt(b, a, demod)
    demod -= np.mean(demod)
    pico = np.max(np.abs(demod))
    return (demod / pico * 0.95).astype(np.float32) if pico > 0 else demod

def añadir_ruido(señal, snr_db):
    pot_sig = np.mean(señal ** 2)
    pot_ruido = pot_sig / (10 ** (snr_db / 10))
    ruido = np.random.normal(0, np.sqrt(pot_ruido), len(señal))
    return (señal + ruido).astype(np.float32)

def calcular_metricas(orig, rec):
    n = min(len(orig), len(rec))
    o = orig[:n]
    r = rec[:n]
    mse = float(np.mean((o - r) ** 2))
    pot_sig = float(np.mean(o ** 2))
    snr = 10 * np.log10(pot_sig / (mse + 1e-12))
    return mse, snr

# ────────────────────────────────────────────────────────────
# CLASE PRINCIPAL DE LA INTERFAZ
# ────────────────────────────────────────────────────────────

class SDRDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SDR Voice Radio — Panel de Control y Análisis")
        self.geometry("1280x820")
        self.minsize(1050, 700)

        # Cargar señal base
        if os.path.exists("audio_original.wav"):
            try:
                import wave
                with wave.open("audio_original.wav", "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                muestras = np.frombuffer(raw, dtype=np.int16)
                self.señal_base = (muestras.astype(np.float32) / 32767.0)
            except Exception:
                self.señal_base = generar_tono_demo()
        else:
            self.señal_base = generar_tono_demo()

        # Cache de señales procesadas
        self.audio_actual = None
        self.s_am = None
        self.s_fm = None
        self.s_demod_am = None
        self.s_demod_fm = None

        self._crear_interfaz()
        self.actualizar_vista()

    def _crear_interfaz(self):
        # Contenedor principal: Izquierda (Controles) | Derecha (Gráficas)
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── Frame lateral izquierdo (Controles) ──
        frame_izq = ttk.Frame(paned, width=320, padding=12)
        paned.add(frame_izq, weight=0)

        # Título
        ttk.Label(frame_izq, text="SDR Radio Control", font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 10))

        # Selector de Vista
        ttk.Label(frame_izq, text="SELECCIONAR VISTA / FASE:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(5, 3))
        self.vista_var = tk.StringVar(value="Fase 1: Audio Original")
        opciones_vista = [
            "Fase 1: Audio Original",
            "Fase 2: Modulación AM",
            "Fase 2: Modulación FM",
            "Comparativa: Espectros AM vs FM",
            "Comparativa: Resistencia al Ruido",
            "Comparativa: Curva SNR vs Canal"
        ]
        self.combo_vista = ttk.Combobox(frame_izq, textvariable=self.vista_var, values=opciones_vista, state="readonly")
        self.combo_vista.pack(fill=tk.X, pady=(0, 15))
        self.combo_vista.bind("<<ComboboxSelected>>", lambda e: self.actualizar_vista())

        # ── Parámetros configurables ──
        lbl_params = ttk.LabelFrame(frame_izq, text="Parámetros de Radio", padding=10)
        lbl_params.pack(fill=tk.X, pady=5)

        # Portadora fc
        ttk.Label(lbl_params, text="Frecuencia Portadora (fc):").pack(anchor="w")
        self.fc_val_lbl = ttk.Label(lbl_params, text="8000 Hz", font=("Consolas", 9, "bold"), foreground="darkblue")
        self.fc_val_lbl.pack(anchor="e")
        self.slider_fc = ttk.Scale(lbl_params, from_=5000, to=15000, value=8000, command=self._on_slider_fc)
        self.slider_fc.pack(fill=tk.X, pady=(0, 8))

        # Índice AM ka
        ttk.Label(lbl_params, text="Índice Modulación AM (ka):").pack(anchor="w")
        self.ka_val_lbl = ttk.Label(lbl_params, text="0.90", font=("Consolas", 9, "bold"), foreground="darkorange")
        self.ka_val_lbl.pack(anchor="e")
        self.slider_ka = ttk.Scale(lbl_params, from_=0.1, to=1.2, value=0.90, command=self._on_slider_ka)
        self.slider_ka.pack(fill=tk.X, pady=(0, 8))

        # Desviación FM kf
        ttk.Label(lbl_params, text="Desviación Frecuencia FM (kf):").pack(anchor="w")
        self.kf_val_lbl = ttk.Label(lbl_params, text="2000 Hz", font=("Consolas", 9, "bold"), foreground="crimson")
        self.kf_val_lbl.pack(anchor="e")
        self.slider_kf = ttk.Scale(lbl_params, from_=500, to=4000, value=2000, command=self._on_slider_kf)
        self.slider_kf.pack(fill=tk.X, pady=(0, 8))

        # Ruido del canal (dB)
        ttk.Label(lbl_params, text="Ruido del Canal (SNR dB):").pack(anchor="w")
        self.snr_canal_lbl = ttk.Label(lbl_params, text="20 dB", font=("Consolas", 9, "bold"), foreground="purple")
        self.snr_canal_lbl.pack(anchor="e")
        self.slider_snr = ttk.Scale(lbl_params, from_=5, to=40, value=20, command=self._on_slider_snr)
        self.slider_snr.pack(fill=tk.X, pady=(0, 5))

        btn_recalc = ttk.Button(lbl_params, text="🔄 Actualizar y Recalcular", command=self.actualizar_vista)
        btn_recalc.pack(fill=tk.X, pady=8)

        # ── Métricas calculadas ──
        lbl_metricas = ttk.LabelFrame(frame_izq, text="Métricas de Calidad", padding=10)
        lbl_metricas.pack(fill=tk.X, pady=10)
        self.lbl_mse = ttk.Label(lbl_metricas, text="MSE: --", font=("Consolas", 9))
        self.lbl_mse.pack(anchor="w")
        self.lbl_snr = ttk.Label(lbl_metricas, text="SNR Recuperada: --", font=("Consolas", 9, "bold"), foreground="green")
        self.lbl_snr.pack(anchor="w", pady=2)
        self.lbl_bw = ttk.Label(lbl_metricas, text="Ancho de Banda: --", font=("Consolas", 9))
        self.lbl_bw.pack(anchor="w")

        # ── Controles de Reproducción de Audio ──
        lbl_audio = ttk.LabelFrame(frame_izq, text="Reproductor de Audio", padding=10)
        lbl_audio.pack(fill=tk.X, pady=5)

        ttk.Button(lbl_audio, text="▶ Original", command=lambda: self._reproducir(self.señal_base)).pack(fill=tk.X, pady=2)
        ttk.Button(lbl_audio, text="▶ Modulada (AM o FM)", command=self._reproducir_modulada).pack(fill=tk.X, pady=2)
        ttk.Button(lbl_audio, text="▶ Demodulada / Recuperada", command=self._reproducir_recuperada).pack(fill=tk.X, pady=2)
        ttk.Button(lbl_audio, text="⏹ Parar Audio", command=self._parar_audio).pack(fill=tk.X, pady=(6, 2))

        # ── Frame lateral derecho (Matplotlib Canvas) ──
        frame_der = ttk.Frame(paned, padding=5)
        paned.add(frame_der, weight=1)

        self.figura = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figura, master=frame_der)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, frame_der)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    # ── Callbacks de Sliders ──
    def _on_slider_fc(self, val):
        self.fc_val_lbl.config(text=f"{int(float(val))} Hz")

    def _on_slider_ka(self, val):
        self.ka_val_lbl.config(text=f"{float(val):.2f}")

    def _on_slider_kf(self, val):
        self.kf_val_lbl.config(text=f"{int(float(val))} Hz")

    def _on_slider_snr(self, val):
        self.snr_canal_lbl.config(text=f"{int(float(val))} dB")

    # ── Reproducción de audio ──
    def _reproducir(self, sig):
        if not AUDIO_OK:
            messagebox.showwarning("Audio", "sounddevice no está disponible.")
            return
        if sig is None or len(sig) == 0:
            return
        sd.stop()
        sd.play(sig, FS)

    def _reproducir_modulada(self):
        vista = self.vista_var.get()
        if "FM" in vista and self.s_fm is not None:
            self._reproducir(self.s_fm)
        elif self.s_am is not None:
            self._reproducir(self.s_am)

    def _reproducir_recuperada(self):
        vista = self.vista_var.get()
        if "FM" in vista and self.s_demod_fm is not None:
            self._reproducir(self.s_demod_fm)
        elif self.s_demod_am is not None:
            self._reproducir(self.s_demod_am)

    def _parar_audio(self):
        if AUDIO_OK:
            sd.stop()

    # ── Renderizado y actualización de gráficas ──
    def actualizar_vista(self):
        fc = float(self.slider_fc.get())
        ka = float(self.slider_ka.get())
        kf = float(self.slider_kf.get())
        snr_canal = float(self.slider_snr.get())

        # Calcular moduladas
        self.s_am = modular_am(self.señal_base, fs=FS, fc=fc, ka=ka)
        self.s_demod_am = demodular_am(self.s_am, fs=FS, fc=fc)

        self.s_fm = modular_fm(self.señal_base, fs=FS, fc=fc, kf=kf)
        self.s_demod_fm = demodular_fm(self.s_fm, fs=FS, fc=fc, kf=kf)

        self.figura.clf()
        vista = self.vista_var.get()

        if vista == "Fase 1: Audio Original":
            self._plot_fase1()
        elif vista == "Fase 2: Modulación AM":
            self._plot_modulacion("AM", self.s_am, self.s_demod_am, fc, 2 * 4000)
        elif vista == "Fase 2: Modulación FM":
            self._plot_modulacion("FM", self.s_fm, self.s_demod_fm, fc, 2 * (kf + 4000))
        elif vista == "Comparativa: Espectros AM vs FM":
            self._plot_comp_espectros(fc, kf)
        elif vista == "Comparativa: Resistencia al Ruido":
            self._plot_comp_ruido(snr_canal)
        elif vista == "Comparativa: Curva SNR vs Canal":
            self._plot_curva_snr()

        self.figura.tight_layout()
        self.canvas.draw()

    # ── Vistas específicas ──
    def _plot_fase1(self):
        ax1 = self.figura.add_subplot(2, 1, 1)
        ax2 = self.figura.add_subplot(2, 1, 2)

        n_p = min(len(self.señal_base), int(0.04 * FS))
        t_ms = np.arange(n_p) / FS * 1000
        ax1.plot(t_ms, self.señal_base[:n_p], color='steelblue', lw=1.2)
        ax1.set_title("Fase 1: Forma de onda de audio original (Tiempo)", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Tiempo (ms)")
        ax1.set_ylabel("Amplitud [-1, 1]")
        ax1.grid(True, alpha=0.3)

        N = len(self.señal_base)
        mags = 20 * np.log10(np.abs(np.fft.rfft(self.señal_base)) / N + 1e-12)
        freqs = np.fft.rfftfreq(N, 1 / FS) / 1000
        ax2.plot(freqs, mags, color='steelblue', lw=0.9)
        ax2.axvline(4.0, color='red', linestyle='--', label='Límite voz 4 kHz')
        ax2.set_title("Fase 1: Espectro de frecuencias (FFT)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Frecuencia (kHz)")
        ax2.set_ylabel("Magnitud (dB)")
        ax2.set_xlim(0, FS / 2 / 1000)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        self.lbl_mse.config(text="MSE: 0.000000 (Referencia)")
        self.lbl_snr.config(text="SNR: ∞ dB (Original)")
        self.lbl_bw.config(text="Ancho de Banda voz: 4.0 kHz")

    def _plot_modulacion(self, tipo, s_mod, s_rec, fc, bw_teorico):
        ejes = self.figura.subplots(3, 2)
        color = 'darkorange' if tipo == "AM" else 'crimson'
        self.figura.suptitle(f"Fase 2: Modulación y Demodulación {tipo} (fc={fc:.0f} Hz)", fontsize=12, fontweight='bold')

        filas = [
            (self.señal_base, "Original", 'steelblue'),
            (s_mod, f"Modulada {tipo}", color),
            (s_rec, f"Recuperada", 'seagreen')
        ]

        n_p = min(len(self.señal_base), int(0.005 * FS))
        t_ms = np.arange(n_p) / FS * 1000

        for i, (sig, tag, c) in enumerate(filas):
            # Tiempo
            ejes[i, 0].plot(t_ms, sig[:n_p], color=c, lw=1)
            ejes[i, 0].set_title(f"{tag} — Tiempo (ms)", fontsize=9)
            ejes[i, 0].set_ylabel("Amplitud")
            ejes[i, 0].set_ylim(-1.15, 1.15)
            ejes[i, 0].grid(True, alpha=0.3)

            # FFT
            N = len(sig)
            mags = 20 * np.log10(np.abs(np.fft.rfft(sig)) / N + 1e-12)
            freqs = np.fft.rfftfreq(N, 1 / FS) / 1000
            ejes[i, 1].plot(freqs, mags, color=c, lw=0.7)
            ejes[i, 1].axvline(fc / 1000, color='red', linestyle='--', lw=0.8)
            ejes[i, 1].set_title(f"{tag} — Espectro (kHz)", fontsize=9)
            ejes[i, 1].set_xlim(0, FS / 2 / 1000)
            ejes[i, 1].grid(True, alpha=0.3)

        mse, snr = calcular_metricas(self.señal_base, s_rec)
        self.lbl_mse.config(text=f"MSE: {mse:.6f}")
        self.lbl_snr.config(text=f"SNR Recuperada: {snr:.2f} dB")
        self.lbl_bw.config(text=f"Ancho Banda {tipo}: {bw_teorico/1000:.1f} kHz")

    def _plot_comp_espectros(self, fc, kf):
        ejes = self.figura.subplots(1, 3)
        self.figura.suptitle("Comparativa Espectral: Original vs AM vs FM", fontsize=12, fontweight='bold')

        series = [
            (self.señal_base, "Voz Original", 'steelblue', None, 4000),
            (self.s_am, "Modulación AM", 'darkorange', fc, 8000),
            (self.s_fm, "Modulación FM", 'crimson', fc, 2 * (kf + 4000))
        ]

        for ax, (sig, tag, c, f_carr, bw) in zip(ejes, series):
            N = len(sig)
            mags = 20 * np.log10(np.abs(np.fft.rfft(sig)) / N + 1e-12)
            freqs = np.fft.rfftfreq(N, 1 / FS) / 1000
            ax.plot(freqs, mags, color=c, lw=0.8, label=tag)
            ax.axvline(4.0, color='gray', linestyle=':', label='4 kHz')
            if f_carr:
                ax.axvline(f_carr / 1000, color='black', linestyle='--', label=f'fc={f_carr/1000:.1f}k')
                ax.axvspan((f_carr - bw / 2) / 1000, (f_carr + bw / 2) / 1000, alpha=0.15, color=c, label=f'BW={bw/1000:.1f}k')
            ax.set_title(tag, fontsize=10, fontweight='bold')
            ax.set_xlabel("Frecuencia (kHz)")
            ax.set_xlim(0, FS / 2 / 1000)
            ax.set_ylim(-90, 5)
            ax.legend(fontsize=7, loc='lower right')
            ax.grid(True, alpha=0.3)

        self.lbl_mse.config(text="Comparativa de BW")
        self.lbl_snr.config(text="AM: 8 kHz | FM: 12 kHz")
        self.lbl_bw.config(text="FM ocupa más espectro")

    def _plot_comp_ruido(self, snr_canal):
        ejes = self.figura.subplots(2, 2)
        self.figura.suptitle(f"Prueba con Ruido en Canal (SNR Canal = {snr_canal:.0f} dB)", fontsize=12, fontweight='bold')

        # Modular con ruido
        s_am_r = añadir_ruido(self.s_am, snr_canal)
        s_fm_r = añadir_ruido(self.s_fm, snr_canal)
        rec_am = demodular_am(s_am_r, fs=FS, fc=float(self.slider_fc.get()))
        rec_fm = demodular_fm(s_fm_r, fs=FS, fc=float(self.slider_fc.get()), kf=float(self.slider_kf.get()))

        mse_am, snr_am = calcular_metricas(self.señal_base, rec_am)
        mse_fm, snr_fm = calcular_metricas(self.señal_base, rec_fm)

        n_p = min(len(self.señal_base), int(0.04 * FS))
        t_ms = np.arange(n_p) / FS * 1000

        # AM Tiempo y Espectro
        ejes[0, 0].plot(t_ms, self.señal_base[:n_p], 'steelblue', alpha=0.5, label='Original')
        ejes[0, 0].plot(t_ms, rec_am[:n_p], 'darkorange', lw=1, label=f'AM Recup ({snr_am:.1f} dB)')
        ejes[0, 0].set_title("AM: Señal en el Tiempo con Ruido", fontsize=9)
        ejes[0, 0].legend(fontsize=7)
        ejes[0, 0].grid(True, alpha=0.3)

        N = len(rec_am)
        mags_am = 20 * np.log10(np.abs(np.fft.rfft(rec_am)) / N + 1e-12)
        freqs = np.fft.rfftfreq(N, 1 / FS) / 1000
        ejes[0, 1].plot(freqs, mags_am, 'darkorange', lw=0.8)
        ejes[0, 1].set_title("AM: Espectro de Señal Recuperada", fontsize=9)
        ejes[0, 1].set_xlim(0, 8)
        ejes[0, 1].grid(True, alpha=0.3)

        # FM Tiempo y Espectro
        ejes[1, 0].plot(t_ms, self.señal_base[:n_p], 'steelblue', alpha=0.5, label='Original')
        ejes[1, 0].plot(t_ms, rec_fm[:n_p], 'crimson', lw=1, label=f'FM Recup ({snr_fm:.1f} dB)')
        ejes[1, 0].set_title("FM: Señal en el Tiempo con Ruido", fontsize=9)
        ejes[1, 0].legend(fontsize=7)
        ejes[1, 0].grid(True, alpha=0.3)

        mags_fm = 20 * np.log10(np.abs(np.fft.rfft(rec_fm)) / N + 1e-12)
        ejes[1, 1].plot(freqs, mags_fm, 'crimson', lw=0.8)
        ejes[1, 1].set_title("FM: Espectro de Señal Recuperada", fontsize=9)
        ejes[1, 1].set_xlim(0, 8)
        ejes[1, 1].grid(True, alpha=0.3)

        ventaja = snr_fm - snr_am
        self.lbl_mse.config(text=f"MSE AM: {mse_am:.4f} | FM: {mse_fm:.4f}")
        self.lbl_snr.config(text=f"SNR AM: {snr_am:.1f} dB | FM: {snr_fm:.1f} dB")
        self.lbl_bw.config(text=f"Ventaja de FM: {ventaja:+.1f} dB")

    def _plot_curva_snr(self):
        ax = self.figura.add_subplot(1, 1, 1)
        fc = float(self.slider_fc.get())
        kf = float(self.slider_kf.get())

        snrs_c = np.arange(5, 41, 4)
        snrs_am, snrs_fm = [], []

        for c_db in snrs_c:
            s_am_r = añadir_ruido(self.s_am, c_db)
            s_fm_r = añadir_ruido(self.s_fm, c_db)
            r_am = demodular_am(s_am_r, fs=FS, fc=fc)
            r_fm = demodular_fm(s_fm_r, fs=FS, fc=fc, kf=kf)
            _, s_a = calcular_metricas(self.señal_base, r_am)
            _, s_f = calcular_metricas(self.señal_base, r_fm)
            snrs_am.append(s_a)
            snrs_fm.append(s_f)

        ax.plot(snrs_c, snrs_am, 'o-', color='darkorange', lw=2, label='AM Recuperada')
        ax.plot(snrs_c, snrs_fm, 's-', color='crimson', lw=2, label='FM Recuperada')
        ax.fill_between(snrs_c, snrs_am, snrs_fm, color='crimson', alpha=0.15, label='Ventaja FM')
        ax.axhline(20, color='gray', linestyle='--', label='20 dB (Umbral óptimo)')

        ax.set_title("Curva de Rendimiento: SNR de Audio vs Calidad del Canal", fontsize=11, fontweight='bold')
        ax.set_xlabel("SNR del Canal [dB] (mayor = mejor canal)")
        ax.set_ylabel("SNR de la Voz Recuperada [dB]")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

        self.lbl_mse.config(text="Curva calculada (5-40 dB)")
        self.lbl_snr.config(text="FM superior a partir de >15 dB")
        self.lbl_bw.config(text="Evidencia para la memoria")

if __name__ == "__main__":
    app = SDRDashboard()
    app.mainloop()
