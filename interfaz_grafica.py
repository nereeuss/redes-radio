"""
=============================================================
 PANEL EDUCATIVO SDR — Modulación y Demodulación de Radio
=============================================================

Diseño intuitivo paso a paso:
  - Pestaña 1: ¿Cómo funciona AM? (Paso a paso con audio y gráficas)
  - Pestaña 2: ¿Cómo funciona FM? (Paso a paso con audio y gráficas)
  - Pestaña 3: AM vs FM con Ruido (Demostración de por qué FM es mejor)

Ejecución:
  python interfaz_grafica.py
"""

import os
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
# PARÁMETROS FIJOS Y SEÑALES
# ────────────────────────────────────────────────────────────

FS = 48_000        # Frecuencia de muestreo (48 kHz)
FC = 8_000.0       # Portadora a 8 kHz
KA = 0.90          # Índice AM
KF = 2_000.0       # Desviación FM
CORTE_LP = 4_000.0 # Filtro paso-bajo a 4 kHz

def generar_tono_demo():
    duracion = 3.0
    t = np.linspace(0, duracion, int(duracion * FS), endpoint=False)
    # Tono con armónicos que simula una vocal
    armonicos = [1.0, 0.7, 0.5, 0.3, 0.2]
    sig = sum(a * np.sin(2 * np.pi * 350 * (i + 1) * t) for i, a in enumerate(armonicos))
    n_fade = int(0.05 * FS)
    sig[:n_fade] *= np.linspace(0, 1, n_fade)
    sig[-n_fade:] *= np.linspace(1, 0, n_fade)
    sig -= np.mean(sig)
    pico = np.max(np.abs(sig))
    return (sig / pico * 0.95).astype(np.float32)

def cargar_o_generar_audio():
    if os.path.exists("audio_original.wav"):
        try:
            import wave
            with wave.open("audio_original.wav", "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                n_ch = wf.getnchannels()
                ancho = wf.getsampwidth()
            tipos = {1: np.int8, 2: np.int16, 4: np.int32}
            muestras = np.frombuffer(raw, dtype=tipos.get(ancho, np.int16))
            if n_ch == 2:
                muestras = muestras.reshape(-1, 2).mean(axis=1)
            sig = muestras.astype(np.float32) / 32767.0
            sig -= np.mean(sig)
            pico = np.max(np.abs(sig))
            if pico > 0:
                sig = sig / pico * 0.95
            return sig.astype(np.float32)
        except Exception:
            return generar_tono_demo()
    return generar_tono_demo()

# Procesamiento AM
def modular_am(señal):
    t = np.arange(len(señal)) / FS
    s = (1.0 + KA * señal) * np.cos(2 * np.pi * FC * t)
    return (s / np.max(np.abs(s)) * 0.95).astype(np.float32)

def demodular_am(s_am):
    t = np.arange(len(s_am)) / FS
    mezclado = s_am * np.cos(2 * np.pi * FC * t)
    b, a = butter(5, CORTE_LP / (FS / 2), btype='low')
    demod = filtfilt(b, a, mezclado)
    demod -= np.mean(demod)
    pico = np.max(np.abs(demod))
    return (demod / pico * 0.95).astype(np.float32) if pico > 0 else demod

# Procesamiento FM
def modular_fm(señal):
    t = np.arange(len(señal)) / FS
    integral = np.cumsum(señal) / FS
    fase = 2 * np.pi * FC * t + 2 * np.pi * KF * integral
    return np.cos(fase).astype(np.float32)

def demodular_fm(s_fm):
    analitica = hilbert(s_fm)
    fase_inst = np.unwrap(np.angle(analitica))
    freq_inst = np.diff(fase_inst) * FS / (2 * np.pi)
    demod = (freq_inst - FC) / KF
    demod = np.append(demod, 0.0)
    b, a = butter(5, CORTE_LP / (FS / 2), btype='low')
    demod = filtfilt(b, a, demod)
    demod -= np.mean(demod)
    pico = np.max(np.abs(demod))
    return (demod / pico * 0.95).astype(np.float32) if pico > 0 else demod

def añadir_ruido(señal, snr_db=15.0):
    pot_sig = np.mean(señal ** 2)
    pot_ruido = pot_sig / (10 ** (snr_db / 10))
    ruido = np.random.normal(0, np.sqrt(pot_ruido), len(señal))
    return (señal + ruido).astype(np.float32)


# ────────────────────────────────────────────────────────────
# INTERFAZ GRÁFICA
# ────────────────────────────────────────────────────────────

class SDRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Laboratorio SDR: Radio AM y FM Explicada Paso a Paso")
        self.geometry("1200x750")
        self.minsize(980, 650)

        # Cargar señales precalculadas
        self.voz_original = cargar_o_generar_audio()
        self.am_mod = modular_am(self.voz_original)
        self.am_rec = demodular_am(self.am_mod)

        self.fm_mod = modular_fm(self.voz_original)
        self.fm_rec = demodular_fm(self.fm_mod)

        self._construir_ui()

    def _construir_ui(self):
        # Cabecera
        header = ttk.Frame(self, padding=10)
        header.pack(fill=tk.X)
        ttk.Label(header, text="📻 Demostrador de Radio: Cómo viaja la voz",
                  font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT)
        ttk.Button(header, text="⏹ Parar Audio", command=self.parar_audio).pack(side=tk.RIGHT)

        # Pestañas
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tab_am = ttk.Frame(self.notebook, padding=10)
        self.tab_fm = ttk.Frame(self.notebook, padding=10)
        self.tab_ruido = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_am, text="  1. Modulación AM (Paso a Paso)  ")
        self.notebook.add(self.tab_fm, text="  2. Modulación FM (Paso a Paso)  ")
        self.notebook.add(self.tab_ruido, text="  3. AM vs FM con Ruido (¿Por qué FM?)  ")

        self._crear_pestana_am()
        self._crear_pestana_fm()
        self._crear_pestana_ruido()

    # ────────────────────────────────────────────────────────
    # PESTAÑA 1: AM PASO A PASO
    # ────────────────────────────────────────────────────────
    def _crear_pestana_am(self):
        paned = ttk.PanedWindow(self.tab_am, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: Los 3 Pasos
        izq = ttk.Frame(paned, width=380, padding=10)
        paned.add(izq, weight=0)

        ttk.Label(izq, text="Cadena de Transmisión AM", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 10))

        # Paso 1
        f1 = ttk.LabelFrame(izq, text="Paso 1: Tu Voz Original", padding=8)
        f1.pack(fill=tk.X, pady=6)
        ttk.Label(f1, text="La voz natural tiene frecuencias bajas (0 a 4 kHz).", wraplength=340).pack(anchor="w", pady=2)
        ttk.Button(f1, text="🔊 Ver y Escuchar Voz Original",
                   command=lambda: self.mostrar_etapa(
                       self.canvas_am, self.fig_am, self.voz_original,
                       "Paso 1: Voz Original (Bajas Frecuencias)",
                       "steelblue", self.lbl_info_am,
                       "✅ ESTA ES TU VOZ: Frecuencias bajas (0 a 4 kHz). Suena natural.",
                       reproducir=self.voz_original
                   )).pack(fill=tk.X, pady=4)

        # Paso 2
        f2 = ttk.LabelFrame(izq, text="Paso 2: Modulación AM (La Portadora)", padding=8)
        f2.pack(fill=tk.X, pady=6)
        ttk.Label(f2, text="Montamos la voz sobre una onda de 8.000 Hz para que pueda viajar.", wraplength=340).pack(anchor="w", pady=2)
        ttk.Button(f2, text="🔊 Ver y Escuchar Señal Modulada",
                   command=lambda: self.mostrar_etapa(
                       self.canvas_am, self.fig_am, self.am_mod,
                       "Paso 2: Señal AM Modulada (Viajando a 8.000 Hz)",
                       "darkorange", self.lbl_info_am,
                       "⚠️ ATENCIÓN: Escuchas un pitido agudo (8 kHz). La voz está escondida dentro de la amplitud de la portadora. ¡Así viaja la radio por el aire!",
                       reproducir=self.am_mod
                   )).pack(fill=tk.X, pady=4)

        # Paso 3
        f3 = ttk.LabelFrame(izq, text="Paso 3: Demodulación (El Receptor)", padding=8)
        f3.pack(fill=tk.X, pady=6)
        ttk.Label(f3, text="El receptor multiplica por 8 kHz y filtra los restos para extraer tu voz.", wraplength=340).pack(anchor="w", pady=2)
        ttk.Button(f3, text="🔊 Ver y Escuchar Voz Recuperada",
                   command=lambda: self.mostrar_etapa(
                       self.canvas_am, self.fig_am, self.am_rec,
                       "Paso 3: Voz Recuperada por el Receptor",
                       "seagreen", self.lbl_info_am,
                       "🎉 VOZ RECUPERADA: El receptor eliminó la portadora de 8 kHz. ¡Vuelve a sonar tu voz limpia!",
                       reproducir=self.am_rec
                   )).pack(fill=tk.X, pady=4)

        # Cuadro de explicación abajo a la izquierda
        self.lbl_info_am = ttk.Label(izq, text="👉 Pulsa en el Paso 1 para empezar.",
                                     font=("Segoe UI", 9, "italic"), foreground="navy", wraplength=340)
        self.lbl_info_am.pack(fill=tk.X, pady=15)

        # Panel derecho: Gráfica
        der = ttk.Frame(paned, padding=5)
        paned.add(der, weight=1)

        self.fig_am = Figure(figsize=(7, 5), dpi=100)
        self.canvas_am = FigureCanvasTkAgg(self.fig_am, master=der)
        self.canvas_am.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas_am, der)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        # Mostrar Paso 1 por defecto
        self.mostrar_etapa(self.canvas_am, self.fig_am, self.voz_original,
                           "Paso 1: Voz Original (Bajas Frecuencias)", "steelblue",
                           self.lbl_info_am, "👉 Pulsa en los pasos para ver y escuchar qué hace cada parte.", reproducir=None)

    # ────────────────────────────────────────────────────────
    # PESTAÑA 2: FM PASO A PASO
    # ────────────────────────────────────────────────────────
    def _crear_pestana_fm(self):
        paned = ttk.PanedWindow(self.tab_fm, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo
        izq = ttk.Frame(paned, width=380, padding=10)
        paned.add(izq, weight=0)

        ttk.Label(izq, text="Cadena de Transmisión FM", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 10))

        # Paso 1
        f1 = ttk.LabelFrame(izq, text="Paso 1: Tu Voz Original", padding=8)
        f1.pack(fill=tk.X, pady=6)
        ttk.Label(f1, text="La misma señal de voz de partida (0 a 4 kHz).", wraplength=340).pack(anchor="w", pady=2)
        ttk.Button(f1, text="🔊 Ver y Escuchar Voz Original",
                   command=lambda: self.mostrar_etapa(
                       self.canvas_fm, self.fig_fm, self.voz_original,
                       "Paso 1: Voz Original", "steelblue", self.lbl_info_fm,
                       "✅ Señal de voz original.", reproducir=self.voz_original
                   )).pack(fill=tk.X, pady=4)

        # Paso 2
        f2 = ttk.LabelFrame(izq, text="Paso 2: Modulación FM (Variación de Frecuencia)", padding=8)
        f2.pack(fill=tk.X, pady=6)
        ttk.Label(f2, text="La amplitud NO cambia. La portadora oscila más rápido o más lento según la voz.", wraplength=340).pack(anchor="w", pady=2)
        ttk.Button(f2, text="🔊 Ver y Escuchar Señal FM Modulada",
                   command=lambda: self.mostrar_etapa(
                       self.canvas_fm, self.fig_fm, self.fm_mod,
                       "Paso 2: Señal FM Modulada (Frecuencia Variable)",
                       "crimson", self.lbl_info_fm,
                       "⚠️ ATENCIÓN: Escuchas la portadora oscilando alrededor de 8 kHz. La amplitud es fija; la voz está en la velocidad de oscilación.",
                       reproducir=self.fm_mod
                   )).pack(fill=tk.X, pady=4)

        # Paso 3
        f3 = ttk.LabelFrame(izq, text="Paso 3: Demodulación FM (Discriminador)", padding=8)
        f3.pack(fill=tk.X, pady=6)
        ttk.Label(f3, text="El receptor mide los cambios de frecuencia y reconstruye la voz.", wraplength=340).pack(anchor="w", pady=2)
        ttk.Button(f3, text="🔊 Ver y Escuchar Voz Recuperada FM",
                   command=lambda: self.mostrar_etapa(
                       self.canvas_fm, self.fig_fm, self.fm_rec,
                       "Paso 3: Voz Recuperada de FM",
                       "seagreen", self.lbl_info_fm,
                       "🎉 VOZ RECUPERADA FM: Reconstruida a partir de la frecuencia instantánea.",
                       reproducir=self.fm_rec
                   )).pack(fill=tk.X, pady=4)

        self.lbl_info_fm = ttk.Label(izq, text="👉 Pulsa en los pasos para ver y escuchar.",
                                     font=("Segoe UI", 9, "italic"), foreground="navy", wraplength=340)
        self.lbl_info_fm.pack(fill=tk.X, pady=15)

        # Panel derecho: Gráfica
        der = ttk.Frame(paned, padding=5)
        paned.add(der, weight=1)

        self.fig_fm = Figure(figsize=(7, 5), dpi=100)
        self.canvas_fm = FigureCanvasTkAgg(self.fig_fm, master=der)
        self.canvas_fm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas_fm, der)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        self.mostrar_etapa(self.canvas_fm, self.fig_fm, self.voz_original,
                           "Paso 1: Voz Original", "steelblue", self.lbl_info_fm,
                           "👉 Pulsa en los pasos para explorar.", reproducir=None)

    # ────────────────────────────────────────────────────────
    # PESTAÑA 3: COMPARATIVA CON RUIDO
    # ────────────────────────────────────────────────────────
    def _crear_pestana_ruido(self):
        paned = ttk.PanedWindow(self.tab_ruido, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo
        izq = ttk.Frame(paned, width=380, padding=10)
        paned.add(izq, weight=0)

        ttk.Label(izq, text="Prueba de Fuego: ¿Por qué FM?", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        ttk.Label(izq, text="Inyectamos el mismo ruido gaussiano en el canal para ver cuál aguanta mejor.",
                  wraplength=340).pack(anchor="w", pady=(0, 12))

        # AM con Ruido
        f_am_r = ttk.LabelFrame(izq, text="1. Recepción AM con Ruido", padding=8)
        f_am_r.pack(fill=tk.X, pady=6)
        ttk.Label(f_am_r, text="AM guarda la voz en la amplitud. El ruido modifica la amplitud.", wraplength=340).pack(anchor="w")
        ttk.Button(f_am_r, text="🔊 Escuchar AM con Ruido",
                   command=self._demo_am_ruido).pack(fill=tk.X, pady=4)

        # FM con Ruido
        f_fm_r = ttk.LabelFrame(izq, text="2. Recepción FM con Ruido", padding=8)
        f_fm_r.pack(fill=tk.X, pady=6)
        ttk.Label(f_fm_r, text="FM guarda la voz en la frecuencia. El ruido de amplitud casi no le afecta.", wraplength=340).pack(anchor="w")
        ttk.Button(f_fm_r, text="🔊 Escuchar FM con Ruido",
                   command=self._demo_fm_ruido).pack(fill=tk.X, pady=4)

        # Explicación
        self.lbl_info_ruido = ttk.Label(izq, text="👉 Pulsa 'Escuchar AM' y luego 'Escuchar FM' para notar la diferencia de calidad.",
                                        font=("Segoe UI", 9, "bold"), foreground="navy", wraplength=340)
        self.lbl_info_ruido.pack(fill=tk.X, pady=15)

        # Panel derecho: Gráfica comparativa
        der = ttk.Frame(paned, padding=5)
        paned.add(der, weight=1)

        self.fig_ruido = Figure(figsize=(7, 5), dpi=100)
        self.canvas_ruido = FigureCanvasTkAgg(self.fig_ruido, master=der)
        self.canvas_ruido.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas_ruido, der)
        toolbar.update()
        toolbar.pack(fill=tk.X)

        self._plot_comparativa_inicial()

    def _demo_am_ruido(self):
        s_ruidosa = añadir_ruido(self.am_mod, snr_db=14.0)
        rec = demodular_am(s_ruidosa)
        self.lbl_info_ruido.config(
            text="❌ RESULTADO AM: ¿Oyes el siseo constante? El ruido sumado al canal alteró directamente la envolvente de la voz.",
            foreground="darkred"
        )
        self._plot_ruido_comparado(rec, "AM Recuperada con Ruido (Siseo notable)", "darkorange")
        self.reproducir_audio(rec)

    def _demo_fm_ruido(self):
        s_ruidosa = añadir_ruido(self.fm_mod, snr_db=14.0)
        rec = demodular_fm(s_ruidosa)
        self.lbl_info_ruido.config(
            text="✅ RESULTADO FM: ¿Ves cómo suena mucho más limpia? Como la voz viaja en la frecuencia, el receptor descarta el ruido de amplitud.",
            foreground="darkgreen"
        )
        self._plot_ruido_comparado(rec, "FM Recuperada con Ruido (Mucho más limpia)", "crimson")
        self.reproducir_audio(rec)

    def _plot_ruido_comparado(self, rec, titulo, color):
        self.fig_ruido.clf()
        ax1 = self.fig_ruido.add_subplot(2, 1, 1)
        ax2 = self.fig_ruido.add_subplot(2, 1, 2)

        n_p = min(len(self.voz_original), int(0.03 * FS))
        t_ms = np.arange(n_p) / FS * 1000

        ax1.plot(t_ms, self.voz_original[:n_p], color='steelblue', alpha=0.5, label='Original Limpia')
        ax1.plot(t_ms, rec[:n_p], color=color, lw=1.2, label=titulo)
        ax1.set_title(f"{titulo} — Forma de onda", fontsize=10, fontweight='bold')
        ax1.set_xlabel("Tiempo (ms)")
        ax1.set_ylabel("Amplitud")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        N = len(rec)
        mags = 20 * np.log10(np.abs(np.fft.rfft(rec)) / N + 1e-12)
        freqs = np.fft.rfftfreq(N, 1 / FS) / 1000
        ax2.plot(freqs, mags, color=color, lw=0.9)
        ax2.set_title("Espectro de Frecuencias (FFT)", fontsize=10, fontweight='bold')
        ax2.set_xlabel("Frecuencia (kHz)")
        ax2.set_ylabel("Magnitud (dB)")
        ax2.set_xlim(0, 8)
        ax2.grid(True, alpha=0.3)

        self.fig_ruido.tight_layout()
        self.canvas_ruido.draw()

    def _plot_comparativa_inicial(self):
        self.fig_ruido.clf()
        ax = self.fig_ruido.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "Pulsa en 'Escuchar AM con Ruido' o 'Escuchar FM con Ruido'\npara ver la comparación gráfica en directo.",
                ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')
        self.canvas_ruido.draw()

    # ────────────────────────────────────────────────────────
    # UTILIDADES DE GRÁFICA Y AUDIO
    # ────────────────────────────────────────────────────────
    def mostrar_etapa(self, canvas, fig, señal, titulo, color, lbl_info, texto_info, reproducir=None):
        lbl_info.config(text=texto_info)

        fig.clf()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # Gráfica 1: Tiempo
        # Mostramos 4 ms para ver las ondas individuales con claridad
        n_p = min(len(señal), int(0.005 * FS))
        t_ms = np.arange(n_p) / FS * 1000
        ax1.plot(t_ms, señal[:n_p], color=color, lw=1.2)
        ax1.set_title(f"{titulo} — Dominio Temporal (5 ms)", fontsize=10, fontweight='bold')
        ax1.set_xlabel("Tiempo (ms)")
        ax1.set_ylabel("Amplitud")
        ax1.set_ylim(-1.15, 1.15)
        ax1.grid(True, alpha=0.3)

        # Gráfica 2: Espectro FFT
        N = len(señal)
        mags = 20 * np.log10(np.abs(np.fft.rfft(señal)) / N + 1e-12)
        freqs = np.fft.rfftfreq(N, 1 / FS) / 1000  # kHz
        ax2.plot(freqs, mags, color=color, lw=0.9)
        ax2.axvline(4.0, color='gray', linestyle=':', label='4 kHz (Límite voz)')
        ax2.axvline(FC / 1000, color='red', linestyle='--', label=f'Portadora fc={FC/1000:.0f} kHz')
        ax2.set_title(f"{titulo} — Espectro de Frecuencias (FFT)", fontsize=10, fontweight='bold')
        ax2.set_xlabel("Frecuencia (kHz)")
        ax2.set_ylabel("Magnitud (dB)")
        ax2.set_xlim(0, FS / 2 / 1000)
        ax2.legend(loc='lower right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        canvas.draw()

        if reproducir is not None:
            self.reproducir_audio(reproducir)

    def reproducir_audio(self, señal):
        if not AUDIO_OK:
            return
        sd.stop()
        sd.play(señal, FS)

    def parar_audio(self):
        if AUDIO_OK:
            sd.stop()


if __name__ == "__main__":
    app = SDRApp()
    app.mainloop()
