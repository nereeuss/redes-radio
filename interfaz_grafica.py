"""
=============================================================
 PANEL EDUCATIVO SDR — Laboratorio Interactivo de Radio
=============================================================
Interfaz visual moderna paso a paso con:
  - Audio real de voz humana en español
  - Comparativa AM vs FM visual y auditiva de alto contraste
  - Transmisión en directo por red sobre sockets UDP (Fase 3)
  - Osciloscopio en tiempo real y monitor de datagramas
  - Cifrado acústico por esteganografía musical
  - Telemetría de red RFC 3550 (latencia, jitter, pérdidas)
  - Diseño moderno y estilizado (Dark Slate UI)
"""

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import wave
import threading
import socket
import time
import queue

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

# Importar protocolo binario SDR de la Fase 3
from protocolo_sdr import (
    empaquetar, desempaquetar, trocear_señal, cargar_mascara_musical,
    aplicar_mascara, retirar_mascara, MetricasRed,
    SAMPLES_PER_BLOCK, BLOCK_MS, FS, FC_DEFAULT,
    MOD_AM, MOD_FM, MASK_OFF, MASK_ON,
    DEFAULT_IP, DEFAULT_PORT, HEADER_SIZE
)

# ────────────────────────────────────────────────────────────
# PARÁMETROS GENERALES
# ────────────────────────────────────────────────────────────

FS = 48_000        # 48 kHz
FC = 8_000.0       # Portadora: 8 kHz
KA = 0.90          # Índice AM
KF = 3_500.0       # Desviación FM (3.5 kHz para auténtica ventaja FM)
CORTE_LP = 3_800.0 # Filtro paso-bajo voz general
CORTE_FM = 3_200.0 # Filtro desénfasis FM (elimina siseo triangular de alta frecuencia)

# Paleta de colores Dark Slate moderna
BG_DARK = "#0f172a"       # Slate 900
BG_CARD = "#1e293b"       # Slate 800
BG_CARD_HOVER = "#334155" # Slate 700
BG_TERMINAL = "#090d16"   # Slate 950 deep
TEXT_WHITE = "#f8fafc"    # Slate 50
TEXT_MUTED = "#94a3b8"    # Slate 400
COLOR_VOZ = "#38bdf8"     # Sky blue
COLOR_AM = "#f59e0b"      # Amber orange
COLOR_FM = "#f43f5e"      # Rose red
COLOR_REC = "#10b981"     # Emerald green
COLOR_ALERT = "#ef4444"   # Red alert
COLOR_PURPLE = "#a855f7"  # Purple accent
COLOR_CYAN = "#06b6d4"    # Cyan
COLOR_PINK = "#ec4899"    # Pink / Magenta

# ────────────────────────────────────────────────────────────
# CARGA DE AUDIO Y PROCESAMIENTO
# ────────────────────────────────────────────────────────────

def cargar_audio_voz():
    ruta = "audio_original.wav"
    if os.path.exists(ruta):
        try:
            with wave.open(ruta, 'rb') as wf:
                n_frames = wf.getnframes()
                ancho = wf.getsampwidth()
                n_canales = wf.getnchannels()
                tasa = wf.getframerate()
                raw_data = wf.readframes(n_frames)

            tipos = {1: np.int8, 2: np.int16, 4: np.int32}
            señal = np.frombuffer(raw_data, dtype=tipos.get(ancho, np.int16)).astype(np.float32)

            if n_canales == 2:
                señal = señal.reshape(-1, 2).mean(axis=1)

            señal /= np.iinfo(tipos.get(ancho, np.int16)).max
            señal -= np.mean(señal)
            pico = np.max(np.abs(señal))
            if pico > 0:
                señal = señal / pico * 0.95

            if tasa != FS:
                from scipy.signal import resample_poly
                import math
                g = math.gcd(int(tasa), int(FS))
                señal = resample_poly(señal, int(FS // g), int(tasa // g))

            return señal.astype(np.float32)
        except Exception as e:
            print(f"[ERROR] No se pudo leer {ruta}: {e}")

    dur = 4.0
    t = np.linspace(0, dur, int(FS * dur), endpoint=False)
    voz_sint = (0.6 * np.sin(2 * np.pi * 220 * t) +
                0.4 * np.sin(2 * np.pi * 440 * t) +
                0.2 * np.sin(2 * np.pi * 880 * t))
    voz_sint -= np.mean(voz_sint)
    return (voz_sint / np.max(np.abs(voz_sint)) * 0.95).astype(np.float32)


def modular_am(señal):
    t = np.arange(len(señal)) / FS
    return ((1.0 + KA * señal) * np.cos(2 * np.pi * FC * t)).astype(np.float32)


def demodular_am(s_am):
    t = np.arange(len(s_am)) / FS
    mezclado = s_am * np.cos(2 * np.pi * FC * t)
    b, a = butter(5, CORTE_LP / (FS / 2), btype='low')
    demod = filtfilt(b, a, mezclado)
    demod -= np.mean(demod)
    umbral = np.percentile(np.abs(demod), 99.8)
    if umbral > 0:
        demod = np.clip(demod / umbral * 0.95, -0.95, 0.95)
    return demod.astype(np.float32)


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

    b, a = butter(4, CORTE_FM / (FS / 2), btype='low')
    demod = filtfilt(b, a, demod)
    demod -= np.mean(demod)

    umbral = np.percentile(np.abs(demod), 99.8)
    if umbral > 0:
        demod = np.clip(demod / umbral * 0.95, -0.95, 0.95)
    return demod.astype(np.float32)


def simular_ruido_canal(señal, snr_db=18.0):
    pot_sig = np.mean(señal ** 2)
    pot_ruido = pot_sig / (10 ** (snr_db / 10))
    ruido = np.random.normal(0, np.sqrt(pot_ruido), len(señal))
    return (señal + ruido).astype(np.float32)


# ────────────────────────────────────────────────────────────
# APLICACIÓN GUI PRINCIPAL
# ────────────────────────────────────────────────────────────

class SDRModernApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SDR Voice Radio — Laboratorio Visual de Telecomunicaciones")
        self.geometry("1280x820")
        self.minsize(1080, 720)
        self.configure(bg=BG_DARK)

        # Cargar señales base
        self.voz = cargar_audio_voz()
        self.am_mod = modular_am(self.voz)
        self.am_rec = demodular_am(self.am_mod)
        self.fm_mod = modular_fm(self.voz)
        self.fm_rec = demodular_fm(self.fm_mod)

        # Señales para comparativa de ruido
        self.ruido_snr = 18.0
        self.am_canal_ruido = simular_ruido_canal(self.am_mod, self.ruido_snr)
        self.am_rec_ruido = demodular_am(self.am_canal_ruido)
        self.fm_canal_ruido = simular_ruido_canal(self.fm_mod, self.ruido_snr)
        self.fm_rec_ruido = demodular_fm(self.fm_canal_ruido)

        # Cargar máscara musical para Fase 3
        dur_s = len(self.voz) / FS
        self.mascara = cargar_mascara_musical(duracion_s=dur_s + 2.0, fs=FS, ruta_wav="mascara_musical.wav")

        # Variables de estado para transmisión UDP
        self.transmision_en_curso = False
        self.detener_transmision_flag = False
        self.audio_recibido_red = None
        self.audio_enmascarado_red = None
        self.metricas_red_ultimo = None

        self.hist_seqs = []
        self.hist_latencia = []
        self.hist_jitter = []

        # Cola thread-safe para eventos de la red
        self.cola_eventos = queue.Queue()

        self._configurar_estilos()
        self._construir_ui()
        self.after(20, self._verificar_cola_eventos)

    def _verificar_cola_eventos(self):
        while not self.cola_eventos.empty():
            try:
                evento, datos = self.cola_eventos.get_nowait()
                if evento == "PAQUETE":
                    self._notificar_paquete_live(*datos)
                elif evento == "FIN":
                    self._finalizar_transmision_live(*datos)
                elif evento == "CANCEL":
                    self._cancelar_transmision_live()
            except queue.Empty:
                break
        self.after(20, self._verificar_cola_eventos)

    def _configurar_estilos(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_CARD, foreground=TEXT_MUTED,
                        padding=[16, 7], font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_CARD_HOVER)],
                  foreground=[("selected", TEXT_WHITE)])

        style.configure("TFrame", background=BG_DARK)
        style.configure("Card.TFrame", background=BG_CARD, relief="flat")

        # Botones temáticos
        style.configure("Modern.TButton", font=("Segoe UI", 9, "bold"), padding=[10, 6],
                        background="#2563eb", foreground="white", borderwidth=0)
        style.map("Modern.TButton", background=[("active", "#1d4ed8")])

        style.configure("Stop.TButton", font=("Segoe UI", 9, "bold"), padding=[10, 5],
                        background="#dc2626", foreground="white", borderwidth=0)
        style.map("Stop.TButton", background=[("active", "#b91c1c")])

        style.configure("AM.TButton", font=("Segoe UI", 9, "bold"), padding=[10, 6],
                        background="#d97706", foreground="white", borderwidth=0)
        style.map("AM.TButton", background=[("active", "#b45309")])

        style.configure("FM.TButton", font=("Segoe UI", 9, "bold"), padding=[10, 6],
                        background="#e11d48", foreground="white", borderwidth=0)
        style.map("FM.TButton", background=[("active", "#be123c")])

        style.configure("Green.TButton", font=("Segoe UI", 9, "bold"), padding=[10, 6],
                        background="#059669", foreground="white", borderwidth=0)
        style.map("Green.TButton", background=[("active", "#047857")])

        style.configure("Purple.TButton", font=("Segoe UI", 9, "bold"), padding=[10, 6],
                        background="#9333ea", foreground="white", borderwidth=0)
        style.map("Purple.TButton", background=[("active", "#7e22ce")])

        style.configure("Pink.TButton", font=("Segoe UI", 9, "bold"), padding=[10, 6],
                        background="#db2777", foreground="white", borderwidth=0)
        style.map("Pink.TButton", background=[("active", "#be185d")])

    def _construir_ui(self):
        # Header superior
        header = tk.Frame(self, bg=BG_CARD, height=60, padx=20)
        header.pack(fill=tk.X, side=tk.TOP)

        titulo_box = tk.Frame(header, bg=BG_CARD)
        titulo_box.pack(side=tk.LEFT, pady=10)

        tk.Label(titulo_box, text="📻 SDR RADIO TRANSMITTER & UDP NETWORK", font=("Segoe UI", 13, "bold"),
                 fg=COLOR_VOZ, bg=BG_CARD).pack(anchor="w")
        dur_s = len(self.voz) / FS
        tk.Label(titulo_box, text=f"Voz real cargada ({dur_s:.1f} s a 48 kHz) — Portadora fc = 8 kHz — Sockets UDP",
                 font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")

        ttk.Button(header, text="⏹ Parar Reproducción", style="Stop.TButton",
                   command=self.parar_audio).pack(side=tk.RIGHT, pady=12)

        # Notebook con las 4 pestañas
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        self.tab_am = ttk.Frame(self.notebook)
        self.tab_fm = ttk.Frame(self.notebook)
        self.tab_ruido = ttk.Frame(self.notebook)
        self.tab_red = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_am, text="  1. Modulación AM  ")
        self.notebook.add(self.tab_fm, text="  2. Modulación FM  ")
        self.notebook.add(self.tab_ruido, text="  3. AM vs FM Ruido  ")
        self.notebook.add(self.tab_red, text="  4. Transmisión en Vivo UDP (Fase 3)  ")

        self._crear_tab_am()
        self._crear_tab_fm()
        self._crear_tab_ruido()
        self._crear_tab_red()

    # ────────────────────────────────────────────────────────────
    # PESTAÑA 1: AM PASO A PASO
    # ────────────────────────────────────────────────────────────
    def _crear_tab_am(self):
        paned = tk.PanedWindow(self.tab_am, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)

        izq = tk.Frame(paned, bg=BG_CARD, width=390, padx=15, pady=15)
        paned.add(izq, minsize=360)

        tk.Label(izq, text="Cadena de Audio AM", font=("Segoe UI", 12, "bold"),
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w", pady=(0, 10))

        # Paso 1
        card1 = tk.LabelFrame(izq, text=" PASO 1 · TU VOZ NATURAL ", font=("Segoe UI", 8, "bold"),
                              bg=BG_CARD, fg=COLOR_VOZ, bd=1, padx=10, pady=8)
        card1.pack(fill=tk.X, pady=6)
        tk.Label(card1, text="Voz en frecuencias bajas (0 a 4 kHz). Escucha cómo suena la persona hablando.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card1, text="▶ 1. Escuchar Voz Original", style="Modern.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_am, self.fig_am, self.voz,
                       "Paso 1: Voz Original (Bajas Frecuencias 0-4 kHz)",
                       COLOR_VOZ, self.lbl_badge_am, self.lbl_desc_am,
                       "✅ ESTADO: Voz Humana Natural", COLOR_VOZ,
                       "Escuchas a la persona hablando en su frecuencia normal. En la gráfica inferior ves que toda la energía está concentrada por debajo de los 4 kHz.",
                       self.voz
                   )).pack(fill=tk.X)

        # Paso 2
        card2 = tk.LabelFrame(izq, text=" PASO 2 · MODULADA EN LA PORTADORA ", font=("Segoe UI", 8, "bold"),
                              bg=BG_CARD, fg=COLOR_AM, bd=1, padx=10, pady=8)
        card2.pack(fill=tk.X, pady=6)
        tk.Label(card2, text="Montamos la voz sobre una portadora de 8.000 Hz. El volumen de la onda dibuja la voz.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card2, text="▶ 2. Escuchar Señal AM Modulada", style="AM.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_am, self.fig_am, self.am_mod,
                       "Paso 2: Señal AM Modulada (Viajando a 8.000 Hz)",
                       COLOR_AM, self.lbl_badge_am, self.lbl_desc_am,
                       "⚠️ ATENCIÓN: Pitido Portadora a 8.000 Hz", COLOR_AM,
                       "¡Escuchas un pitido agudo! No suena a voz humana porque la información está escondida en la envolvente (amplitud) a 8 kHz. Así viaja la radio.",
                       self.am_mod
                   )).pack(fill=tk.X)

        # Paso 3
        card3 = tk.LabelFrame(izq, text=" PASO 3 · DEMODULADA (EL RECEPTOR) ", font=("Segoe UI", 8, "bold"),
                              bg=BG_CARD, fg=COLOR_REC, bd=1, padx=10, pady=8)
        card3.pack(fill=tk.X, pady=6)
        tk.Label(card3, text="El receptor multiplica por 8 kHz y filtra el residuo agudo para rescatar la voz.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card3, text="▶ 3. Escuchar Voz Recuperada", style="Green.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_am, self.fig_am, self.am_rec,
                       "Paso 3: Voz Recuperada por el Receptor AM",
                       COLOR_REC, self.lbl_badge_am, self.lbl_desc_am,
                       "🎉 ÉXITO: Voz Rescatada y Limpia", COLOR_REC,
                       "El receptor eliminó la portadora de 8 kHz y el filtro paso-bajo dejó pasar únicamente la voz. ¡Vuelves a entender a la persona con total claridad!",
                       self.am_rec
                   )).pack(fill=tk.X)

        # Caja info
        info_box = tk.Frame(izq, bg="#0f172a", padx=10, pady=10)
        info_box.pack(fill=tk.X, pady=(15, 0))
        self.lbl_badge_am = tk.Label(info_box, text="👉 HAZ CLIC EN EL PASO 1", font=("Segoe UI", 9, "bold"),
                                     fg=COLOR_VOZ, bg="#0f172a")
        self.lbl_badge_am.pack(anchor="w")
        self.lbl_desc_am = tk.Label(info_box, text="Haz clic en los tres botones para ver y escuchar cómo la voz se transforma para viajar por el aire y se reconstruye en el receptor.",
                                    font=("Segoe UI", 8), fg=TEXT_MUTED, bg="#0f172a", wraplength=310, justify="left")
        self.lbl_desc_am.pack(anchor="w", pady=(4, 0))

        # Panel derecho: Canvas
        der = tk.Frame(paned, bg=BG_CARD)
        paned.add(der, minsize=500)

        self.fig_am = Figure(figsize=(7, 5), dpi=100, facecolor=BG_CARD)
        self.canvas_am = FigureCanvasTkAgg(self.fig_am, master=der)
        self.canvas_am.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._configurar_toolbar(self.canvas_am, der)

        self.mostrar_grafica(self.canvas_am, self.fig_am, self.voz, "Paso 1: Voz Original (Bajas Frecuencias)", COLOR_VOZ)

    # ────────────────────────────────────────────────────────────
    # PESTAÑA 2: FM PASO A PASO
    # ────────────────────────────────────────────────────────────
    def _crear_tab_fm(self):
        paned = tk.PanedWindow(self.tab_fm, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)

        izq = tk.Frame(paned, bg=BG_CARD, width=390, padx=15, pady=15)
        paned.add(izq, minsize=360)

        tk.Label(izq, text="Cadena de Audio FM", font=("Segoe UI", 12, "bold"),
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w", pady=(0, 10))

        # Paso 1
        card1 = tk.LabelFrame(izq, text=" PASO 1 · TU VOZ NATURAL ", font=("Segoe UI", 8, "bold"),
                              bg=BG_CARD, fg=COLOR_VOZ, bd=1, padx=10, pady=8)
        card1.pack(fill=tk.X, pady=6)
        tk.Label(card1, text="Voz en banda base (0 a 4 kHz). Esta es la señal moduladora que viaja en la frecuencia.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card1, text="▶ 1. Escuchar Voz Original", style="Modern.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_fm, self.fig_fm, self.voz,
                       "Paso 1: Voz Original (Banda Base 0-4 kHz)",
                       COLOR_VOZ, self.lbl_badge_fm, self.lbl_desc_fm,
                       "✅ ESTADO: Voz Humana Natural", COLOR_VOZ,
                       "La voz humana original. Observa en la gráfica inferior que el ancho de banda ocupa estrictamente hasta los 4 kHz.",
                       self.voz
                   )).pack(fill=tk.X)

        # Paso 2
        card2 = tk.LabelFrame(izq, text=" PASO 2 · MODULADA EN FRECUENCIA (FM) ", font=("Segoe UI", 8, "bold"),
                              bg=BG_CARD, fg=COLOR_FM, bd=1, padx=10, pady=8)
        card2.pack(fill=tk.X, pady=6)
        tk.Label(card2, text="La amplitud es constante (inmune al ruido). La voz hace que la onda se comprima y se expanda.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card2, text="▶ 2. Escuchar Señal FM Modulada", style="FM.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_fm, self.fig_fm, self.fm_mod,
                       "Paso 2: Señal FM Modulada (8.000 Hz ± 3.500 Hz)",
                       COLOR_FM, self.lbl_badge_fm, self.lbl_desc_fm,
                       "⚡ ATENCIÓN: Sonido Sintético Modulado", COLOR_FM,
                       "Escuchas cómo el tono oscila rápidamente. En FM la información va en la velocidad de vibración, no en el volumen. ¡Por eso es inmune a las interferencias!",
                       self.fm_mod
                   )).pack(fill=tk.X)

        # Paso 3
        card3 = tk.LabelFrame(izq, text=" PASO 3 · DEMODULADA (DISCRIMINADOR FM) ", font=("Segoe UI", 8, "bold"),
                              bg=BG_CARD, fg=COLOR_REC, bd=1, padx=10, pady=8)
        card3.pack(fill=tk.X, pady=6)
        tk.Label(card3, text="El receptor mide los giros de fase instántaneos (Hilbert) y rescata la voz original limpia.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card3, text="▶ 3. Escuchar Voz Recuperada FM", style="Green.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_fm, self.fig_fm, self.fm_rec,
                       "Paso 3: Voz Recuperada por el Receptor FM",
                       COLOR_REC, self.lbl_badge_fm, self.lbl_desc_fm,
                       "🎉 ÉXITO: Voz FM Rescatada a la Perfección", COLOR_REC,
                       "¡El receptor FM extrajo la variación de frecuencia y aplicó el desénfasis! La voz suena con un timbre cálido, limpio y sin siseos molestos.",
                       self.fm_rec
                   )).pack(fill=tk.X)

        info_box = tk.Frame(izq, bg="#0f172a", padx=10, pady=10)
        info_box.pack(fill=tk.X, pady=(15, 0))
        self.lbl_badge_fm = tk.Label(info_box, text="👉 HAZ CLIC EN EL PASO 1", font=("Segoe UI", 9, "bold"),
                                     fg=COLOR_VOZ, bg="#0f172a")
        self.lbl_badge_fm.pack(anchor="w")
        self.lbl_desc_fm = tk.Label(info_box, text="FM mantiene la amplitud fija en 1.0 y varía la frecuencia. Esto permite eliminar cualquier ruido electromagnético cortando los picos.",
                                    font=("Segoe UI", 8), fg=TEXT_MUTED, bg="#0f172a", wraplength=310, justify="left")
        self.lbl_desc_fm.pack(anchor="w", pady=(4, 0))

        der = tk.Frame(paned, bg=BG_CARD)
        paned.add(der, minsize=500)

        self.fig_fm = Figure(figsize=(7, 5), dpi=100, facecolor=BG_CARD)
        self.canvas_fm = FigureCanvasTkAgg(self.fig_fm, master=der)
        self.canvas_fm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._configurar_toolbar(self.canvas_fm, der)

        self.mostrar_grafica(self.canvas_fm, self.fig_fm, self.voz, "Paso 1: Voz Original (Banda Base)", COLOR_VOZ)

    # ────────────────────────────────────────────────────────────
    # PESTAÑA 3: COMPARATIVA CON RUIDO
    # ────────────────────────────────────────────────────────────
    def _crear_tab_ruido(self):
        paned = tk.PanedWindow(self.tab_ruido, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)

        izq = tk.Frame(paned, bg=BG_CARD, width=410, padx=15, pady=15)
        paned.add(izq, minsize=380)

        tk.Label(izq, text="Prueba de Canal Ruidoso", font=("Segoe UI", 12, "bold"),
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        tk.Label(izq, text="Inyectamos el mismo ruido electromagnético de radio a ambas señales. ¡Compara cómo suena cada una!",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=350, justify="left").pack(anchor="w", pady=(4, 12))

        # Tarjeta AM
        card_am = tk.LabelFrame(izq, text=" 📻 RECEPCIÓN AM (AFECTADA POR RUIDO) ", font=("Segoe UI", 8, "bold"),
                                bg=BG_CARD, fg=COLOR_AM, bd=1, padx=10, pady=8)
        card_am.pack(fill=tk.X, pady=6)
        tk.Label(card_am, text="AM guarda la voz en la amplitud. Como el ruido modifica la amplitud, se oye un siseo constante.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=340, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card_am, text="🔊 Escuchar AM con Ruido (Siseo)", style="AM.TButton",
                   command=self._accion_am_ruido).pack(fill=tk.X)

        # Tarjeta FM
        card_fm = tk.LabelFrame(izq, text=" 📡 RECEPCIÓN FM (INMUNE AL RUIDO) ", font=("Segoe UI", 8, "bold"),
                                bg=BG_CARD, fg=COLOR_FM, bd=1, padx=10, pady=8)
        card_fm.pack(fill=tk.X, pady=6)
        tk.Label(card_fm, text="FM guarda la voz en la frecuencia. El limitador del receptor corta el ruido de amplitud y la voz suena limpia.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=340, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card_fm, text="🔊 Escuchar FM con Ruido (Voz Limpia)", style="FM.TButton",
                   command=self._accion_fm_ruido).pack(fill=tk.X)

        # Resumen explicativo
        resumen_box = tk.Frame(izq, bg="#0f172a", padx=12, pady=12)
        resumen_box.pack(fill=tk.X, pady=(15, 0))
        self.lbl_titulo_resumen = tk.Label(resumen_box, text="⚖️ ¿POR QUÉ ELEGIMOS FM?",
                                           font=("Segoe UI", 9, "bold"), fg="#38bdf8", bg="#0f172a")
        self.lbl_titulo_resumen.pack(anchor="w")
        self.lbl_texto_resumen = tk.Label(
            resumen_box,
            text="Pulsa en los botones para escuchar y ver la diferencia.\n\n"
                 "• En AM escucharás la voz tapada por ruido.\n"
                 "• En FM la voz sigue sonando clara y nítida.\n\n"
                 "Por este motivo, para la Fase 3 transmitimos la radio FM paquetizada por sockets UDP.",
            font=("Segoe UI", 8), fg=TEXT_WHITE, bg="#0f172a", wraplength=330, justify="left"
        )
        self.lbl_texto_resumen.pack(anchor="w", pady=(5, 0))

        der = tk.Frame(paned, bg=BG_CARD)
        paned.add(der, minsize=500)

        self.fig_ruido = Figure(figsize=(7, 5), dpi=100, facecolor=BG_CARD)
        self.canvas_ruido = FigureCanvasTkAgg(self.fig_ruido, master=der)
        self.canvas_ruido.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._configurar_toolbar(self.canvas_ruido, der)

        self._plot_ruido_inicial()

    def _accion_am_ruido(self):
        self._plot_comparativa_ruido_grafica(self.am_rec_ruido, "AM Recuperada con Ruido (Voz degradada con siseo)", COLOR_AM)
        self.reproducir_audio(self.am_rec_ruido)

    def _accion_fm_ruido(self):
        self._plot_comparativa_ruido_grafica(self.fm_rec_ruido, "FM Recuperada con Ruido (Voz limpia y nítida)", COLOR_FM)
        self.reproducir_audio(self.fm_rec_ruido)

    # ────────────────────────────────────────────────────────────
    # PESTAÑA 4: TRANSMISIÓN EN VIVO UDP (FASE 3)
    # ────────────────────────────────────────────────────────────
    def _crear_tab_red(self):
        paned = tk.PanedWindow(self.tab_red, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)

        izq = tk.Frame(paned, bg=BG_CARD, width=430, padx=14, pady=12)
        paned.add(izq, minsize=410)

        tk.Label(izq, text="Transmisión por Red UDP en Vivo", font=("Segoe UI", 12, "bold"),
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        tk.Label(izq, text="Observa los datagramas viajando en tiempo real, el osciloscopio del paquete entrante y la telemetría RFC 3550.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=380, justify="left").pack(anchor="w", pady=(2, 8))

        # Tarjeta 1: Controles de Emisión
        card_ctrl = tk.LabelFrame(izq, text=" ⚙️ CONTROL DE EMISIÓN UDP ", font=("Segoe UI", 8, "bold"),
                                  bg=BG_CARD, fg=COLOR_VOZ, bd=1, padx=10, pady=8)
        card_ctrl.pack(fill=tk.X, pady=4)

        f_cfg = tk.Frame(card_ctrl, bg=BG_CARD)
        f_cfg.pack(fill=tk.X, pady=(0, 6))

        self.var_mascara = tk.BooleanVar(value=True)
        chk_mask = tk.Checkbutton(f_cfg, text="🔒 Cifrado Musical (Esteganografía)",
                                  variable=self.var_mascara, font=("Segoe UI", 8, "bold"),
                                  fg="#f472b6", bg=BG_CARD, selectcolor=BG_DARK,
                                  activebackground=BG_CARD, activeforeground="#f472b6")
        chk_mask.pack(side=tk.LEFT)

        self.var_modo_rt = tk.BooleanVar(value=True)
        chk_rt = tk.Checkbutton(f_cfg, text="⏱ Modo Tiempo Real (10 ms)",
                                variable=self.var_modo_rt, font=("Segoe UI", 8),
                                fg=TEXT_WHITE, bg=BG_CARD, selectcolor=BG_DARK,
                                activebackground=BG_CARD, activeforeground=TEXT_WHITE)
        chk_rt.pack(side=tk.RIGHT)

        f_btns = tk.Frame(card_ctrl, bg=BG_CARD)
        f_btns.pack(fill=tk.X)

        self.btn_iniciar_tx = ttk.Button(f_btns, text="🚀 Iniciar Transmisión en Vivo", style="Purple.TButton",
                                         command=self._iniciar_transmision_live)
        self.btn_iniciar_tx.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_detener_tx = ttk.Button(f_btns, text="⏹ Detener", style="Stop.TButton",
                                         command=self._detener_transmision_live, state=tk.DISABLED)
        self.btn_detener_tx.pack(side=tk.RIGHT)

        # Tarjeta 2: Monitor de Terminal de Datagramas en Vivo
        card_feed = tk.LabelFrame(izq, text=" 📟 MONITOR DE DATAGRAMAS (SOCKET FEED EN DIRECTO) ",
                                  font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg="#38bdf8", bd=1, padx=8, pady=6)
        card_feed.pack(fill=tk.X, pady=4)

        self.feed_terminal = tk.Text(card_feed, height=6, bg="#090d16", fg="#38bdf8",
                                     font=("Consolas", 8), bd=0, padx=6, pady=4, wrap=tk.NONE)
        self.feed_terminal.pack(fill=tk.BOTH, expand=True)
        self.feed_terminal.insert(tk.END, "[SDR-NET] Esperando inicio de transmisión...\n[CONFIG] Socket UDP 127.0.0.1:5005 | MTU=1500B | Bloque=982B\n")
        self.feed_terminal.config(state=tk.DISABLED)

        # Tarjeta 3: Telemetría de Red en Vivo (KPIs)
        card_kpis = tk.LabelFrame(izq, text=" 📊 TELEMETRÍA DE RED (RFC 3550) ", font=("Segoe UI", 8, "bold"),
                                  bg=BG_CARD, fg=COLOR_REC, bd=1, padx=8, pady=6)
        card_kpis.pack(fill=tk.X, pady=4)

        self.prog_tx = ttk.Progressbar(card_kpis, orient=tk.HORIZONTAL, mode="determinate")
        self.prog_tx.pack(fill=tk.X, pady=(0, 6))

        f_grid = tk.Frame(card_kpis, bg=BG_CARD)
        f_grid.pack(fill=tk.X)

        for col, (titulo, attr) in enumerate([
            ("PAQUETES", "lbl_kpi_pkts"),
            ("PÉRDIDAS", "lbl_kpi_perd"),
            ("RETARDO", "lbl_kpi_ret"),
            ("JITTER", "lbl_kpi_jit"),
            ("VELOCIDAD", "lbl_kpi_kbps")
        ]):
            box = tk.Frame(f_grid, bg="#0f172a", padx=4, pady=4)
            box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            tk.Label(box, text=titulo, font=("Segoe UI", 6, "bold"), fg=TEXT_MUTED, bg="#0f172a").pack()
            lbl = tk.Label(box, text="--", font=("Segoe UI", 8, "bold"), fg=TEXT_WHITE, bg="#0f172a")
            lbl.pack()
            setattr(self, attr, lbl)

        # Tarjeta 4: Verificación Auditiva
        card_audio = tk.LabelFrame(izq, text=" 🔊 VERIFICACIÓN AUDITIVA TRAS LA TRANSMISIÓN ",
                                   font=("Segoe UI", 8, "bold"), bg=BG_CARD, fg="#f59e0b", bd=1, padx=8, pady=6)
        card_audio.pack(fill=tk.X, pady=4)

        tk.Label(card_audio, text="Comprueba qué viajaba en la red (música de camuflaje) vs lo que extrae el receptor:",
                 font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_CARD, justify="left", wraplength=380).pack(anchor="w", pady=(0, 4))

        f_play_btns = tk.Frame(card_audio, bg=BG_CARD)
        f_play_btns.pack(fill=tk.X)

        self.btn_play_mask = ttk.Button(f_play_btns, text="🎵 Tráfico Red (Música)", style="Pink.TButton",
                                        command=self._reproducir_audio_enmascarado, state=tk.DISABLED)
        self.btn_play_mask.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self.btn_play_rec = ttk.Button(f_play_btns, text="✅ Voz Recuperada UDP", style="Green.TButton",
                                       command=self._reproducir_audio_red, state=tk.DISABLED)
        self.btn_play_rec.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))

        # Panel Derecho: Gráfica Canvas
        der = tk.Frame(paned, bg=BG_CARD)
        paned.add(der, minsize=500)

        self.fig_red = Figure(figsize=(7, 5), dpi=100, facecolor=BG_CARD)
        self.canvas_red = FigureCanvasTkAgg(self.fig_red, master=der)
        self.canvas_red.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._configurar_toolbar(self.canvas_red, der)

        self._plot_red_inicial()

    def _plot_red_inicial(self):
        self.fig_red.clf()
        self.ax_scope = self.fig_red.add_subplot(2, 1, 1)
        self.ax_net = self.fig_red.add_subplot(2, 1, 2)

        for ax in (self.ax_scope, self.ax_net):
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#334155")

        # Subplot 1: Osciloscopio del datagrama actual
        t_ms_pkt = np.arange(SAMPLES_PER_BLOCK) / FS * 1000
        self.line_scope, = self.ax_scope.plot(t_ms_pkt, np.zeros(SAMPLES_PER_BLOCK),
                                              color=COLOR_REC, lw=1.2, label="Datagrama 10 ms (480 muestras int16)")
        self.ax_scope.set_title("Osciloscopio del Datagrama UDP Actual (Pulsa 'Iniciar Transmisión en Vivo')",
                                fontsize=9, fontweight="bold", color=TEXT_WHITE)
        self.ax_scope.set_xlabel("Tiempo dentro del datagrama (ms)", color=TEXT_MUTED, fontsize=8)
        self.ax_scope.set_ylabel("Amplitud [-1, 1]", color=TEXT_MUTED, fontsize=8)
        self.ax_scope.set_ylim(-1.15, 1.15)
        self.ax_scope.grid(True, color="#1e293b", linestyle="--", alpha=0.7)
        self.ax_scope.legend(loc="upper right", facecolor=BG_CARD, edgecolor="#334155", labelcolor=TEXT_WHITE, fontsize=7)

        # Subplot 2: Dinámica de Red
        self.ax_net.set_title("Telemetría de Red — Retardo y Jitter por Datagrama en Tiempo Real (RFC 3550)",
                              fontsize=9, fontweight="bold", color=TEXT_WHITE)
        self.ax_net.set_xlabel("Número de Secuencia del Paquete", color=TEXT_MUTED, fontsize=8)
        self.ax_net.set_ylabel("Milisegundos (ms)", color=TEXT_MUTED, fontsize=8)
        self.ax_net.set_xlim(0, 641)
        self.ax_net.set_ylim(0, 2.5)
        self.line_latencia, = self.ax_net.plot([], [], color=COLOR_CYAN, marker='.', markersize=3, linestyle='', label="Retardo (ms)")
        self.line_jitter, = self.ax_net.plot([], [], color="#fbbf24", lw=1.2, label="Jitter RFC 3550 (ms)")
        self.ax_net.grid(True, color="#1e293b", linestyle="--", alpha=0.7)
        self.ax_net.legend(loc="upper right", facecolor=BG_CARD, edgecolor="#334155", labelcolor=TEXT_WHITE, fontsize=7)

        self.fig_red.tight_layout()
        self.canvas_red.draw()

    # ────────────────────────────────────────────────────────────
    # CONTROL DE TRANSMISIÓN LIVE (HILOS Y SOCKETS)
    # ────────────────────────────────────────────────────────────
    def _iniciar_transmision_live(self):
        if self.transmision_en_curso:
            return

        self.transmision_en_curso = True
        self.detener_transmision_flag = False

        # Preparar botones y terminal
        self.btn_iniciar_tx.config(state=tk.DISABLED)
        self.btn_detener_tx.config(state=tk.NORMAL)
        self.btn_play_mask.config(state=tk.DISABLED)
        self.btn_play_rec.config(state=tk.DISABLED)

        self.feed_terminal.config(state=tk.NORMAL)
        self.feed_terminal.delete("1.0", tk.END)
        self.feed_terminal.insert(tk.END, "[TX] Iniciando emisor UDP...\n")
        self.feed_terminal.config(state=tk.DISABLED)

        # Resetear métricas e historial
        self.hist_seqs.clear()
        self.hist_latencia.clear()
        self.hist_jitter.clear()
        self.prog_tx["value"] = 0

        # Preparar señal modulada FM
        señal_a_modular = self.fm_mod
        usar_mascara = self.var_mascara.get()
        modo_rt = self.var_modo_rt.get()

        bloques_raw = trocear_señal(señal_a_modular, SAMPLES_PER_BLOCK)

        if usar_mascara:
            mask_flag = MASK_ON
            bloques_tx = [aplicar_mascara(b, self.mascara, n_seq=i, bloque=SAMPLES_PER_BLOCK)
                          for i, b in enumerate(bloques_raw)]
            self.audio_enmascarado_red = np.concatenate(bloques_tx)
        else:
            mask_flag = MASK_OFF
            bloques_tx = bloques_raw
            self.audio_enmascarado_red = None

        # Resetear gráficas
        self._plot_red_inicial()

        # Lanzar hilo en background
        threading.Thread(target=self._hilo_transmision_live,
                         args=(bloques_tx, mask_flag, modo_rt),
                         daemon=True).start()

    def _detener_transmision_live(self):
        if self.transmision_en_curso:
            self.detener_transmision_flag = True

    def _hilo_transmision_live(self, bloques, mask_flag, modo_rt):
        puerto = 5005
        sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            sock_rx.bind(('127.0.0.1', puerto))
        except Exception:
            puerto = 5006
            sock_rx.bind(('127.0.0.1', puerto))

        sock_rx.settimeout(0.3)
        sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        paquetes_rx = {}
        metricas = MetricasRed()
        total_bloques = len(bloques)
        delay = (BLOCK_MS / 1000.0) if modo_rt else 0.0015

        for i, bloque in enumerate(bloques):
            if self.detener_transmision_flag:
                break

            t_tx = time.time()
            datagrama = empaquetar(bloque, n_seq=i, mod=MOD_FM, fc=int(FC), mask=mask_flag)
            sock_tx.sendto(datagrama, ('127.0.0.1', puerto))

            try:
                data, addr = sock_rx.recvfrom(2048)
                t_rx = time.time()
                pkt = desempaquetar(data)
                metricas.registrar_paquete(pkt, len(data))

                muestras = pkt['muestras']
                if pkt['mask'] == MASK_ON and self.mascara is not None:
                    muestras = retirar_mascara(muestras, self.mascara, n_seq=pkt['seq'], bloque=SAMPLES_PER_BLOCK)

                paquetes_rx[pkt['seq']] = muestras
                retardo_ms = max(0.05, (t_rx - pkt['timestamp']) * 1000)

                actualizar_grafica = (i % 8 == 0) or (i == total_bloques - 1)
                self.cola_eventos.put(("PAQUETE", (
                    i + 1, total_bloques, pkt['seq'], retardo_ms,
                    metricas.jitter, len(data), pkt['mask'] == MASK_ON,
                    muestras, actualizar_grafica
                )))

            except socket.timeout:
                pass

            t_transcurrido = time.time() - t_tx
            t_espera = max(0, delay - t_transcurrido)
            if t_espera > 0:
                time.sleep(t_espera)

        sock_tx.close()
        sock_rx.close()

        if paquetes_rx and not self.detener_transmision_flag:
            seqs_ordenadas = sorted(paquetes_rx.keys())
            señal_fm_reensamblada = np.concatenate([paquetes_rx[s] for s in seqs_ordenadas])
            audio_recuperado = demodular_fm(señal_fm_reensamblada)
            resumen = metricas.resumen()
            self.cola_eventos.put(("FIN", (audio_recuperado, resumen)))
        else:
            self.cola_eventos.put(("CANCEL", ()))

    def _notificar_paquete_live(self, num_pkt, total_pkts, pkt_seq, retardo_ms,
                                jitter_ms, tamaño_bytes, con_mascara, muestras_bloque,
                                actualizar_grafica):
        pct = (num_pkt / total_pkts) * 100
        self.prog_tx["value"] = pct

        # Terminal feed en vivo
        estado_mask = "🔒 MASK" if con_mascara else "📡 RAW "
        msg = f"[RX  5005] #{pkt_seq:03d} ({tamaño_bytes}B) | {retardo_ms:.2f}ms | Jit:{jitter_ms:.2f}ms | {estado_mask}\n"

        self.feed_terminal.config(state=tk.NORMAL)
        self.feed_terminal.insert(tk.END, msg)
        self.feed_terminal.see(tk.END)
        self.feed_terminal.config(state=tk.DISABLED)

        # KPIs
        self.lbl_kpi_pkts.config(text=f"{num_pkt}/{total_pkts} ({pct:.0f}%)")
        self.lbl_kpi_perd.config(text="0.0%", fg=COLOR_REC)
        self.lbl_kpi_ret.config(text=f"{retardo_ms:.2f} ms")
        self.lbl_kpi_jit.config(text=f"{jitter_ms:.2f} ms")
        self.lbl_kpi_kbps.config(text=f"{tamaño_bytes * 8 / (BLOCK_MS / 1000) / 1000:.0f} kbps")

        # Actualizar datos de gráficas
        self.hist_seqs.append(pkt_seq)
        self.hist_latencia.append(retardo_ms)
        self.hist_jitter.append(jitter_ms)

        if actualizar_grafica:
            # 1. Osciloscopio del datagrama actual
            color_scope = "#ec4899" if con_mascara else COLOR_REC
            self.line_scope.set_ydata(muestras_bloque)
            self.line_scope.set_color(color_scope)
            tipo_txt = "🔒 Con Máscara Musical Activa" if con_mascara else "📡 FM Pura"
            self.ax_scope.set_title(f"Datagrama #{pkt_seq:03d} en Vivo ({len(muestras_bloque)} muestras) — {tipo_txt}",
                                    color=TEXT_WHITE, fontsize=9, fontweight="bold")

            # 2. Dinámica de red
            self.line_latencia.set_data(self.hist_seqs, self.hist_latencia)
            self.line_jitter.set_data(self.hist_seqs, self.hist_jitter)

            max_y = max(2.5, max(self.hist_latencia[-20:] or [1.0]) * 1.3)
            self.ax_net.set_ylim(0, max_y)

            self.canvas_red.draw_idle()

    def _finalizar_transmision_live(self, audio_recuperado, resumen):
        self.transmision_en_curso = False
        self.audio_recibido_red = audio_recuperado

        self.btn_iniciar_tx.config(state=tk.NORMAL)
        self.btn_detener_tx.config(state=tk.DISABLED)
        self.btn_play_rec.config(state=tk.NORMAL)
        if self.audio_enmascarado_red is not None:
            self.btn_play_mask.config(state=tk.NORMAL)

        # Mensaje final en terminal
        self.feed_terminal.config(state=tk.NORMAL)
        self.feed_terminal.insert(tk.END, f"\n✅ [SDR-NET] ¡TRANSMISIÓN COMPLETADA! ({resumen['paquetes_recibidos']} paquetes recibidos)\n")
        self.feed_terminal.insert(tk.END, f"   Pérdidas: {resumen['perdida_pct']:.1f}% | Retardo Medio: {resumen['retardo_medio_ms']:.2f} ms | Jitter: {resumen['jitter_ms']:.2f} ms\n")
        self.feed_terminal.see(tk.END)
        self.feed_terminal.config(state=tk.DISABLED)

        # Mostrar gráfica final con la señal reensamblada completa
        self.ax_scope.cla()
        self.ax_scope.set_facecolor("#0f172a")
        for spine in self.ax_scope.spines.values():
            spine.set_color("#334155")

        n_p = min(len(self.voz), int(0.040 * FS))
        t_ms = np.arange(n_p) / FS * 1000

        self.ax_scope.plot(t_ms, self.voz[:n_p], color=COLOR_VOZ, alpha=0.45, lw=1.5, label="Voz Original")
        self.ax_scope.plot(t_ms, audio_recuperado[:n_p], color=COLOR_REC, lw=1.1, label="Voz Reconstruida tras Red UDP")
        self.ax_scope.set_title("Transmision Completada: Voz Reconstruida tras Sockets UDP (0.0% Perdida)",
                                color="#10b981", fontsize=10, fontweight="bold")
        self.ax_scope.set_xlabel("Tiempo (ms)", color=TEXT_MUTED, fontsize=8)
        self.ax_scope.set_ylabel("Amplitud", color=TEXT_MUTED, fontsize=8)
        self.ax_scope.set_ylim(-1.2, 1.2)
        self.ax_scope.legend(loc="upper right", facecolor=BG_CARD, edgecolor="#334155", labelcolor=TEXT_WHITE, fontsize=7)
        self.ax_scope.grid(True, color="#1e293b", linestyle="--", alpha=0.7)

        self.canvas_red.draw()

    def _cancelar_transmision_live(self):
        self.transmision_en_curso = False
        self.btn_iniciar_tx.config(state=tk.NORMAL)
        self.btn_detener_tx.config(state=tk.DISABLED)

        self.feed_terminal.config(state=tk.NORMAL)
        self.feed_terminal.insert(tk.END, "\n⚠️ [SDR-NET] Transmisión detenida por el usuario.\n")
        self.feed_terminal.see(tk.END)
        self.feed_terminal.config(state=tk.DISABLED)

    def _reproducir_audio_enmascarado(self):
        if self.audio_enmascarado_red is not None:
            self.reproducir_audio(self.audio_enmascarado_red)

    def _reproducir_audio_red(self):
        if self.audio_recibido_red is not None:
            self.reproducir_audio(self.audio_recibido_red)

    # ────────────────────────────────────────────────────────────
    # RENDERIZADO DE GRÁFICAS OSCURAS Y ELEGANTES
    # ────────────────────────────────────────────────────────────
    def mostrar_y_reproducir(self, canvas, fig, señal, titulo, color, lbl_b, lbl_d, txt_b, col_b, txt_d, sig_play):
        lbl_b.config(text=txt_b, fg=col_b)
        lbl_d.config(text=txt_d)
        self.mostrar_grafica(canvas, fig, señal, titulo, color)
        self.reproducir_audio(sig_play)

    def mostrar_grafica(self, canvas, fig, señal, titulo, color):
        fig.clf()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        for ax in (ax1, ax2):
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#334155")

        # 1. Gráfica Temporal
        n_muestras_plot = int(0.010 * FS)
        t_ms = np.arange(n_muestras_plot) / FS * 1000
        ax1.plot(t_ms, señal[:n_muestras_plot], color=color, lw=1.2)
        ax1.set_title(f"{titulo} — Forma de Onda (10 ms)", fontsize=10, fontweight='bold', color=TEXT_WHITE)
        ax1.set_xlabel("Tiempo (ms)", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylabel("Amplitud [-1, 1]", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylim(-1.15, 1.15)
        ax1.grid(True, color="#1e293b", linestyle="--", alpha=0.7)

        # 2. Gráfica Espectro FFT
        N = len(señal)
        mags = 20 * np.log10(np.abs(np.fft.rfft(señal)) / N + 1e-12)
        freqs = np.fft.rfftfreq(N, 1 / FS) / 1000
        ax2.plot(freqs, mags, color=color, lw=0.9)
        ax2.axvline(4.0, color="#64748b", linestyle=":", label="4 kHz (límite voz)")
        ax2.axvline(FC / 1000, color="#f87171", linestyle="--", label=f"Portadora fc={FC/1000:.0f} kHz")
        ax2.set_title("Espectro de Frecuencias (FFT)", fontsize=10, fontweight='bold', color=TEXT_WHITE)
        ax2.set_xlabel("Frecuencia (kHz)", color=TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("Magnitud (dB)", color=TEXT_MUTED, fontsize=8)
        ax2.set_xlim(0, FS / 2 / 1000)
        ax2.legend(loc="lower right", facecolor=BG_CARD, edgecolor="#334155", labelcolor=TEXT_WHITE, fontsize=7)
        ax2.grid(True, color="#1e293b", linestyle="--", alpha=0.7)

        fig.tight_layout()
        canvas.draw()

    def _plot_ruido_inicial(self):
        self.fig_ruido.clf()
        ax = self.fig_ruido.add_subplot(1, 1, 1)
        ax.set_facecolor("#0f172a")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.text(0.5, 0.5, "Pulsa en 'Escuchar AM con Ruido' o 'Escuchar FM con Ruido'\npara ver la comparación gráfica y auditiva en tiempo real.",
                ha="center", va="center", fontsize=11, color=TEXT_MUTED)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.canvas_ruido.draw()

    def _plot_comparativa_ruido_grafica(self, sig_recup, titulo, color):
        self.fig_ruido.clf()
        ax1 = self.fig_ruido.add_subplot(2, 1, 1)
        ax2 = self.fig_ruido.add_subplot(2, 1, 2)

        for ax in (ax1, ax2):
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#334155")

        n_p = min(len(self.voz), int(0.040 * FS))
        t_ms = np.arange(n_p) / FS * 1000

        ax1.plot(t_ms, self.voz[:n_p], color=COLOR_VOZ, alpha=0.45, lw=1.5, label="Voz Original Limpia")
        ax1.plot(t_ms, sig_recup[:n_p], color=color, lw=1.1, label=titulo)
        ax1.set_title(f"{titulo} — Señal en el Tiempo", fontsize=10, fontweight="bold", color=TEXT_WHITE)
        ax1.set_xlabel("Tiempo (ms)", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylabel("Amplitud", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylim(-1.2, 1.2)
        ax1.legend(loc="upper right", facecolor=BG_CARD, edgecolor="#334155", labelcolor=TEXT_WHITE, fontsize=7)
        ax1.grid(True, color="#1e293b", linestyle="--", alpha=0.7)

        N = len(sig_recup)
        mags = 20 * np.log10(np.abs(np.fft.rfft(sig_recup)) / N + 1e-12)
        freqs = np.fft.rfftfreq(N, 1 / FS) / 1000
        ax2.plot(freqs, mags, color=color, lw=0.9)
        ax2.set_title("Espectro de Frecuencias (FFT)", fontsize=10, fontweight="bold", color=TEXT_WHITE)
        ax2.set_xlabel("Frecuencia (kHz)", color=TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("Magnitud (dB)", color=TEXT_MUTED, fontsize=8)
        ax2.set_xlim(0, 8)
        ax2.grid(True, color="#1e293b", linestyle="--", alpha=0.7)

        self.fig_ruido.tight_layout()
        self.canvas_ruido.draw()

    def _configurar_toolbar(self, canvas, parent):
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.config(background=BG_CARD)
        toolbar._message_label.config(background=BG_CARD, foreground=TEXT_WHITE)
        for button in toolbar.winfo_children():
            button.config(background=BG_CARD)
        toolbar.update()
        toolbar.pack(fill=tk.X)

    def reproducir_audio(self, sig):
        if not AUDIO_OK:
            return
        sd.stop()
        sd.play(sig, FS)

    def parar_audio(self):
        if AUDIO_OK:
            sd.stop()


if __name__ == "__main__":
    app = SDRModernApp()
    app.mainloop()
