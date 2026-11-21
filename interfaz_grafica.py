"""
=============================================================
 PANEL EDUCATIVO SDR — Laboratorio Interactivo de Radio
=============================================================
Interfaz visual moderna paso a paso con:
  - Audio real de voz humana en español
  - Comparativa AM vs FM visual y auditiva de alto contraste
  - Diseño moderno y estilizado (Dark Slate UI)
"""

import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import wave

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
# PARÁMETROS GENERALES
# ────────────────────────────────────────────────────────────

FS = 48_000        # 48 kHz
FC = 8_000.0       # Portadora: 8 kHz
KA = 0.90          # Índice AM
KF = 2_000.0       # Desviación FM (Hz)
CORTE_LP = 4_000.0 # Filtro paso-bajo voz (4 kHz)

# Paleta de colores Dark Slate moderna
BG_DARK = "#0f172a"       # Slate 900
BG_CARD = "#1e293b"       # Slate 800
BG_CARD_HOVER = "#334155" # Slate 700
TEXT_WHITE = "#f8fafc"    # Slate 50
TEXT_MUTED = "#94a3b8"    # Slate 400
COLOR_VOZ = "#38bdf8"     # Sky blue
COLOR_AM = "#f59e0b"      # Amber orange
COLOR_FM = "#f43f5e"      # Rose red
COLOR_REC = "#10b981"     # Emerald green
COLOR_ALERT = "#ef4444"   # Red alert

# ────────────────────────────────────────────────────────────
# CARGA DE AUDIO Y PROCESAMIENTO
# ────────────────────────────────────────────────────────────

def cargar_audio_voz():
    ruta = "audio_original.wav"
    if os.path.exists(ruta):
        try:
            with wave.open(ruta, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                n_ch = wf.getnchannels()
                ancho = wf.getsampwidth()
            tipos = {1: np.int8, 2: np.int16, 4: np.int32}
            muestras = np.frombuffer(raw, dtype=tipos.get(ancho, np.int16)).astype(np.float32)
            if n_ch == 2:
                muestras = muestras.reshape(-1, 2).mean(axis=1)
            muestras -= np.mean(muestras)
            pico = np.max(np.abs(muestras))
            if pico > 0:
                muestras = muestras / pico * 0.95
            return muestras.astype(np.float32)
        except Exception as e:
            print("Error cargando WAV:", e)
    
    # Fallback sintético si no hubiera archivo
    duracion = 4.0
    t = np.linspace(0, duracion, int(duracion * FS), endpoint=False)
    sig = (0.7 * np.sin(2 * np.pi * 300 * t) + 0.4 * np.sin(2 * np.pi * 600 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t))
    sig -= np.mean(sig)
    return (sig / np.max(np.abs(sig)) * 0.95).astype(np.float32)

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

def modular_fm(señal):
    t = np.arange(len(señal)) / FS
    integral = np.cumsum(señal) / FS
    fase = 2 * np.pi * FC * t + 2 * np.pi * KF * integral
    return np.cos(fase).astype(np.float32)

def demodular_fm(s_fm):
    # Demodulador FM con limitador de amplitud (suprime ruido en la envolvente)
    s_lim = np.sign(s_fm)  # Limitador duro de amplitud
    analitica = hilbert(s_lim)
    fase_inst = np.unwrap(np.angle(analitica))
    freq_inst = np.diff(fase_inst) * FS / (2 * np.pi)
    demod = (freq_inst - FC) / KF
    demod = np.append(demod, 0.0)
    b, a = butter(5, CORTE_LP / (FS / 2), btype='low')
    demod = filtfilt(b, a, demod)
    demod -= np.mean(demod)
    pico = np.max(np.abs(demod))
    return (demod / pico * 0.95).astype(np.float32) if pico > 0 else demod

def simular_ruido_canal(señal, snr_db=12.0):
    pot_sig = np.mean(señal ** 2)
    pot_ruido = pot_sig / (10 ** (snr_db / 10))
    ruido = np.random.normal(0, np.sqrt(pot_ruido), len(señal))
    return (señal + ruido).astype(np.float32)


# ────────────────────────────────────────────────────────────
# APLICACIÓN GUI MODERNA
# ────────────────────────────────────────────────────────────

class SDRModernApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SDR Voice Radio — Laboratorio Visual de Telecomunicaciones")
        self.geometry("1260x800")
        self.minsize(1050, 700)
        self.configure(bg=BG_DARK)

        # Cargar señales
        self.voz = cargar_audio_voz()
        self.am_mod = modular_am(self.voz)
        self.am_rec = demodular_am(self.am_mod)
        self.fm_mod = modular_fm(self.voz)
        self.fm_rec = demodular_fm(self.fm_mod)

        # Señales ruidosas cacheadas para demo instantánea
        self.ruido_snr = 12.0  # Canal ruidoso realista
        self.am_canal_ruido = simular_ruido_canal(self.am_mod, self.ruido_snr)
        self.am_rec_ruido = demodular_am(self.am_canal_ruido)
        self.fm_canal_ruido = simular_ruido_canal(self.fm_mod, self.ruido_snr)
        self.fm_rec_ruido = demodular_fm(self.fm_canal_ruido)

        self._configurar_estilos()
        self._construir_ui()

    def _configurar_estilos(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Configurar estilos de notebook y frames
        style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_CARD, foreground=TEXT_MUTED,
                        padding=[18, 8], font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_CARD_HOVER)],
                  foreground=[("selected", TEXT_WHITE)])

        style.configure("TFrame", background=BG_DARK)
        style.configure("Card.TFrame", background=BG_CARD, relief="flat")
        
        # Botones modernos estilizados
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

    def _construir_ui(self):
        # ── Barra Superior (Header) ──
        header = tk.Frame(self, bg=BG_CARD, height=65, padx=20)
        header.pack(fill=tk.X, side=tk.TOP)

        titulo_box = tk.Frame(header, bg=BG_CARD)
        titulo_box.pack(side=tk.LEFT, pady=12)

        tk.Label(titulo_box, text="📻 SDR RADIO TRANSMITTER", font=("Segoe UI", 14, "bold"),
                 fg=COLOR_VOZ, bg=BG_CARD).pack(anchor="w")
        dur_s = len(self.voz) / FS
        tk.Label(titulo_box, text=f"Voz humana real cargada ({dur_s:.1f} s a 48 kHz) — Portadora fc = 8 kHz",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")

        # Botón de parar audio
        ttk.Button(header, text="⏹ Parar Reproducción", style="Stop.TButton",
                   command=self.parar_audio).pack(side=tk.RIGHT, pady=15)

        # ── Notebook Pestañas ──
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.tab_am = ttk.Frame(self.notebook)
        self.tab_fm = ttk.Frame(self.notebook)
        self.tab_ruido = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_am, text="  1. Modulación AM (Paso a Paso)  ")
        self.notebook.add(self.tab_fm, text="  2. Modulación FM (Paso a Paso)  ")
        self.notebook.add(self.tab_ruido, text="  3. AM vs FM con Ruido (¿Por qué FM?)  ")

        self._crear_tab_am()
        self._crear_tab_fm()
        self._crear_tab_ruido()

    # ────────────────────────────────────────────────────────────
    # PESTAÑA 1: AM PASO A PASO
    # ────────────────────────────────────────────────────────────
    def _crear_tab_am(self):
        paned = tk.PanedWindow(self.tab_am, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)

        # Panel izquierdo: Guía interactiva
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

        # Caja de información didáctica
        info_box = tk.Frame(izq, bg="#0f172a", padx=10, pady=10)
        info_box.pack(fill=tk.X, pady=(15, 0))
        self.lbl_badge_am = tk.Label(info_box, text="👉 HAZ CLIC EN EL PASO 1", font=("Segoe UI", 9, "bold"),
                                     fg=COLOR_VOZ, bg="#0f172a")
        self.lbl_badge_am.pack(anchor="w")
        self.lbl_desc_am = tk.Label(info_box, text="Haz clic en los tres botones para ver y escuchar cómo la voz se transforma para viajar por el aire y se reconstruye en el receptor.",
                                    font=("Segoe UI", 8), fg=TEXT_MUTED, bg="#0f172a", wraplength=310, justify="left")
        self.lbl_desc_am.pack(anchor="w", pady=(4, 0))

        # Panel derecho: Canvas Matplotlib oscuro
        der = tk.Frame(paned, bg=BG_CARD)
        paned.add(der, minsize=500)

        self.fig_am = Figure(figsize=(7, 5), dpi=100, facecolor=BG_CARD)
        self.canvas_am = FigureCanvasTkAgg(self.fig_am, master=der)
        self.canvas_am.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._configurar_toolbar(self.canvas_am, der)

        # Estado inicial
        self.mostrar_grafica(self.canvas_am, self.fig_am, self.voz,
                             "Paso 1: Voz Original (Bajas Frecuencias 0-4 kHz)", COLOR_VOZ)

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
        tk.Label(card1, text="La misma voz de partida hablando en español.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card1, text="▶ 1. Escuchar Voz Original", style="Modern.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_fm, self.fig_fm, self.voz,
                       "Paso 1: Voz Original (0-4 kHz)",
                       COLOR_VOZ, self.lbl_badge_fm, self.lbl_desc_fm,
                       "✅ ESTADO: Voz Humana Natural", COLOR_VOZ,
                       "La señal de voz sin modular.",
                       self.voz
                   )).pack(fill=tk.X)

        # Paso 2
        card2 = tk.LabelFrame(izq, text=" PASO 2 · MODULADA EN FRECUENCIA ", font=("Segoe UI", 8, "bold"),
                              bg=BG_CARD, fg=COLOR_FM, bd=1, padx=10, pady=8)
        card2.pack(fill=tk.X, pady=6)
        tk.Label(card2, text="La amplitud es CONSTANTE. La onda se aprieta o se ensancha según hables.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card2, text="▶ 2. Escuchar Señal FM Modulada", style="FM.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_fm, self.fig_fm, self.fm_mod,
                       "Paso 2: Señal FM Modulada (Frecuencia Variable a 8 kHz)",
                       COLOR_FM, self.lbl_badge_fm, self.lbl_desc_fm,
                       "⚠️ ATENCIÓN: Portadora FM oscilando a 8.000 Hz", COLOR_FM,
                       "La amplitud no cambia nunca. Lo que escuchas es la portadora cambiando de velocidad de oscilación en tiempo real.",
                       self.fm_mod
                   )).pack(fill=tk.X)

        # Paso 3
        card3 = tk.LabelFrame(izq, text=" PASO 3 · DEMODULADA FM ", font=("Segoe UI", 8, "bold"),
                              bg=BG_CARD, fg=COLOR_REC, bd=1, padx=10, pady=8)
        card3.pack(fill=tk.X, pady=6)
        tk.Label(card3, text="El discriminador mide la velocidad de giro (frecuencia) y reconstruye la voz.",
                 font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card3, text="▶ 3. Escuchar Voz Recuperada FM", style="Green.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_fm, self.fig_fm, self.fm_rec,
                       "Paso 3: Voz Reconstruida por Receptor FM",
                       COLOR_REC, self.lbl_badge_fm, self.lbl_desc_fm,
                       "🎉 ÉXITO: Voz Reconstruida Perfectamente", COLOR_REC,
                       "El receptor midió la derivada de la fase para sacar la frecuencia y filtró. ¡La voz suena nítida!",
                       self.fm_rec
                   )).pack(fill=tk.X)

        info_box = tk.Frame(izq, bg="#0f172a", padx=10, pady=10)
        info_box.pack(fill=tk.X, pady=(15, 0))
        self.lbl_badge_fm = tk.Label(info_box, text="👉 HAZ CLIC EN EL PASO 1", font=("Segoe UI", 9, "bold"),
                                     fg=COLOR_VOZ, bg="#0f172a")
        self.lbl_badge_fm.pack(anchor="w")
        self.lbl_desc_fm = tk.Label(info_box, text="En FM la altura de la onda es fija (1.0). Fíjate en la gráfica temporal de la derecha cómo nunca cambia el techo ni el suelo de la onda.",
                                    font=("Segoe UI", 8), fg=TEXT_MUTED, bg="#0f172a", wraplength=310, justify="left")
        self.lbl_desc_fm.pack(anchor="w", pady=(4, 0))

        der = tk.Frame(paned, bg=BG_CARD)
        paned.add(der, minsize=500)

        self.fig_fm = Figure(figsize=(7, 5), dpi=100, facecolor=BG_CARD)
        self.canvas_fm = FigureCanvasTkAgg(self.fig_fm, master=der)
        self.canvas_fm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._configurar_toolbar(self.canvas_fm, der)

        self.mostrar_grafica(self.canvas_fm, self.fig_fm, self.voz,
                             "Paso 1: Voz Original (0-4 kHz)", COLOR_VOZ)

    # ────────────────────────────────────────────────────────────
    # PESTAÑA 3: COMPARATIVA CON RUIDO (EVIDENCIA CLARA)
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
                 "Por este motivo, para la Fase 3 y 4 usaremos FM como modulación base para transmitir por la red UDP.",
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

        # 1. Gráfica Temporal (zoom a 10 ms para ver forma de onda clara)
        n_zoom = min(len(señal), int(0.010 * FS))
        t_ms = np.arange(n_zoom) / FS * 1000
        ax1.plot(t_ms, señal[:n_zoom], color=color, lw=1.2)
        ax1.set_title(f"{titulo} — Forma de Onda (10 ms)", fontsize=10, fontweight='bold', color=TEXT_WHITE)
        ax1.set_xlabel("Tiempo (ms)", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylabel("Amplitud [-1, 1]", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylim(-1.15, 1.15)
        ax1.grid(True, color="#1e293b", linestyle="--", alpha=0.7)

        # 2. Gráfica Espectro FFT
        N = len(señal)
        mags = 20 * np.log10(np.abs(np.fft.rfft(señal)) / N + 1e-12)
        freqs = np.fft.rfftfreq(N, 1 / FS) / 1000  # kHz
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

        # 1. Tiempo: Original vs Recuperada con ruido
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

        # 2. Espectro FFT
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
