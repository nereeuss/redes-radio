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

# Añadir vendor al path si existe (numpy, scipy, matplotlib)
_vendor = os.path.join(os.path.dirname(__file__), "vendor")
if os.path.isdir(_vendor) and _vendor not in sys.path:
    sys.path.insert(0, _vendor)

# Añadir carpeta modulos/ al path para protocolo_sdr y modulador_en_linea
_modulos = os.path.join(os.path.dirname(__file__), "modulos")
if _modulos not in sys.path:
    sys.path.insert(0, _modulos)

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import wave
import threading
import socket
import struct
import time
import queue

# Activar nitidez HD (HiDPI) en Windows para evitar texto y gráficas borrosas
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def obtener_ip_local():
    """Detecta la dirección IP de la interfaz de red local (Wi-Fi o Ethernet)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
matplotlib.rcParams["axes.edgecolor"] = "#E2E8F0"
matplotlib.rcParams["axes.linewidth"] = 0.8
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

# Importar moduladores en línea y funciones de la Fase 5
from modulador_en_linea import (
    modular_am_bloque, modular_fm_bloque,
    demodular_am_bloque, demodular_fm_bloque,
    serializar_bloque, deserializar_bloque,
    EstadoFM, calcular_snr,
    FC_AM_DEFAULT, FC_FM_DEFAULT, KA_DEFAULT, KF_DEFAULT
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

# Paleta Tema Claro Suave (Notion / Apple / Stripe)
BG_DARK = "#F8FAFC"       # Gris Perla suave (fondo general)
BG_CARD = "#FFFFFF"       # Blanco Puro (tarjetas y gráficas)
BG_INNER = "#F1F5F9"      # Gris muy tenue (cajas internas, consola, KPIs)
BG_CARD_HOVER = "#FFFFFF" # Pestaña activa
BORDER_CARD = "#E2E8F0"   # Borde gris fino
BG_TERMINAL = "#F1F5F9"   # Consola clara legible
TEXT_WHITE = "#0F172A"    # Gris Antracita (títulos y texto principal)
TEXT_SEC = "#475569"      # Gris Pizarra (subtítulos y descripciones)
TEXT_MUTED = "#475569"    # Gris Pizarra

# Acento principal único (Azul Índigo) y semánticos cerrados
ACCENT = "#4F46E5"        # Indigo 600
ACCENT_HOVER = "#4338CA"  # Indigo 700
ACCENT_SOFT = "#4F46E5"   # Indigo 600
COLOR_VOZ = "#4F46E5"     # Curva principal
COLOR_AM = "#4F46E5"      # Curva principal
COLOR_FM = "#4F46E5"      # Curva principal
COLOR_REC = "#059669"     # Verde Esmeralda (éxito / voz recuperada / grabar)
COLOR_ALERT = "#DC2626"   # Rojo Coral (detener / pérdidas / error)
COLOR_PURPLE = "#4F46E5"  # Indigo 600
COLOR_CYAN = "#64748B"    # Gris pizarra para serie secundaria en gráficas
COLOR_PINK = "#475569"    # Gris pizarra para botón neutro

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
    # Filtro paso-banda IF (regla de Carson alrededor de fc = 8 kHz) para eliminar ruido fuera de banda
    b_bp, a_bp = butter(4, [2000.0 / (FS / 2), 15000.0 / (FS / 2)], btype='band')
    s_fm_filt = filtfilt(b_bp, a_bp, s_fm)

    analitica = hilbert(s_fm_filt)
    fase_inst = np.unwrap(np.angle(analitica))
    freq_inst = np.diff(fase_inst) * FS / (2 * np.pi)
    demod = (freq_inst - FC) / KF
    demod = np.append(demod, 0.0)

    b, a = butter(4, CORTE_FM / (FS / 2), btype='low')
    demod = filtfilt(b, a, demod)
    demod -= np.mean(demod)
    return np.clip(demod, -0.95, 0.95).astype(np.float32)


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

        # Variables de estado para transmisión UDP (Fase 4)
        self.f4_modo_red = tk.StringVar(value="LOCAL")
        self.f4_ip_dest = tk.StringVar(value="192.168.1.")
        self.f4_puerto = tk.IntVar(value=5005)
        self.transmision_en_curso = False
        self.detener_transmision_flag = False
        self.audio_recibido_red = None
        self.audio_enmascarado_red = None
        self.metricas_red_ultimo = None

        self.hist_seqs = []
        self.hist_latencia = []
        self.hist_jitter = []

        # Variables de estado para Fase 5 (Integración AM/FM sobre UDP)
        self.ip_local = obtener_ip_local()
        self.f5_modo_red = tk.StringVar(value="LOCAL")
        self.f5_ip_dest = tk.StringVar(value="192.168.1.")
        self.f5_puerto = tk.IntVar(value=5018)
        self.f5_modo = tk.StringVar(value="FM")
        self.f5_fc = tk.DoubleVar(value=FC_FM_DEFAULT)
        self.f5_param = tk.DoubleVar(value=KF_DEFAULT)
        self.f5_sim_perdida = tk.BooleanVar(value=False)
        self.f5_tam_bloque = tk.IntVar(value=512)
        self.f5_audio_modulado = None
        self.f5_audio_recuperado = None
        self.f5_en_curso = False
        self.f5_detener_flag = False
        self.f5_hist_seqs = []
        self.f5_hist_latencia = []
        self.f5_hist_jitter = []

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
                elif evento == "F4_LOG":
                    self._log_f4(*datos)
                elif evento == "F4_TX_FIN":
                    self._finalizar_f4_tx(*datos)
                elif evento == "F4_ERROR":
                    self._error_f4(*datos)
                elif evento == "F5_PAQUETE":
                    self._notificar_f5_paquete(*datos)
                elif evento == "F5_LOG":
                    self._log_f5(*datos)
                elif evento == "F5_TX_FIN":
                    self._finalizar_f5_tx(*datos)
                elif evento == "F5_FIN":
                    self._finalizar_f5(*datos)
                elif evento == "F5_CANCEL":
                    self._cancelar_f5()
                elif evento == "F5_ERROR":
                    self._error_f5(*datos)
                elif evento == "VOZ_GRABADA":
                    self._on_voz_actualizada(*datos)
                elif evento == "VOZ_ERROR":
                    self._on_voz_error(*datos)
                elif evento == "RX_LOG":
                    self._rx_log(*datos)
                elif evento == "RX_PKT":
                    self._rx_notificar_pkt(*datos)
                elif evento == "RX_FIN":
                    self._rx_finalizar(*datos)
                elif evento == "RX_CANCEL":
                    self._rx_cancelar()
                elif evento == "RX_ERROR":
                    self._rx_error(*datos)
            except queue.Empty:
                break
        self.after(20, self._verificar_cola_eventos)

    def _configurar_estilos(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TNotebook", background=BG_DARK, borderwidth=0, tabmargins=[4, 6, 4, 0])
        style.configure("TNotebook.Tab", background=BG_DARK, foreground=TEXT_SEC,
                        padding=[16, 9], font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_CARD), ("active", "#EFF6FF")],
                  foreground=[("selected", ACCENT), ("active", TEXT_WHITE)])

        style.configure("TFrame", background=BG_DARK)
        style.configure("Card.TFrame", background=BG_CARD, relief="flat")

        style.configure("TEntry", fieldbackground=BG_CARD, foreground=TEXT_WHITE,
                        bordercolor=BORDER_CARD, lightcolor=ACCENT, padding=[6, 4])
        style.configure("TSpinbox", fieldbackground=BG_CARD, foreground=TEXT_WHITE,
                        bordercolor=BORDER_CARD, arrowsize=12)

        style.configure("Horizontal.TProgressbar", troughcolor=BG_INNER, background=ACCENT,
                        borderwidth=0, thickness=6)

        # Botones cerrados al esquema de color (Índigo acción, Esmeralda éxito/grabar, Coral detener, Neutro suave)
        for nombre, bg, fg, act in [
            ("Modern.TButton", ACCENT, "white", ACCENT_HOVER),
            ("AM.TButton", ACCENT, "white", ACCENT_HOVER),
            ("FM.TButton", ACCENT, "white", ACCENT_HOVER),
            ("Purple.TButton", ACCENT, "white", ACCENT_HOVER),
            ("Green.TButton", COLOR_REC, "white", "#047857"),
            ("Stop.TButton", COLOR_ALERT, "white", "#B91C1C"),
            ("Blue.TButton", BG_INNER, TEXT_WHITE, "#E2E8F0"),
            ("Pink.TButton", BG_INNER, TEXT_WHITE, "#E2E8F0"),
        ]:
            style.configure(nombre, font=("Segoe UI", 9, "bold"), padding=[14, 8],
                            background=bg, foreground=fg, borderwidth=0, focusthickness=0)
            style.map(nombre,
                      background=[("active", act), ("disabled", "#E2E8F0")],
                      foreground=[("disabled", "#94A3B8")])

    def _crear_tarjeta(self, parent, titulo: str, acento: str = ACCENT_SOFT, pady: int = 6, estado: str = None):
        """Crea una tarjeta con esquinas redondeadas (border-radius) estilo Tailwind/Linear sin bordes duros."""
        canvas = tk.Canvas(parent, bg=BG_DARK, highlightthickness=0, bd=0)
        canvas.pack(fill=tk.X, pady=pady)

        inner = tk.Frame(canvas, bg=BG_CARD, padx=14, pady=12)
        win_id = canvas.create_window((1, 1), window=inner, anchor="nw")

        def _dibujar_fondo(event=None):
            w = canvas.winfo_width()
            h = inner.winfo_reqheight() + 2
            if w < 10:
                return
            canvas.configure(height=h)
            canvas.itemconfig(win_id, width=w - 2)
            canvas.delete("bg_round")
            r = 12
            x1, y1, x2, y2 = 1, 1, w - 1, h - 1
            pts = [
                x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
                x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
                x1, y2, x1, y2-r, x1, y1+r, x1, y1
            ]
            canvas.create_polygon(pts, smooth=True, fill=BG_CARD, outline=BORDER_CARD, width=1, tags="bg_round")
            canvas.tag_lower("bg_round")

        canvas.bind("<Configure>", _dibujar_fondo)
        inner.bind("<Configure>", _dibujar_fondo)

        hdr = tk.Frame(inner, bg=BG_CARD)
        hdr.pack(fill=tk.X, pady=(0, 8))
        tk.Label(hdr, text=titulo, font=("Segoe UI", 9, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side=tk.LEFT)
        if estado:
            tk.Label(hdr, text=estado, font=("Segoe UI", 8), fg=acento, bg=BG_CARD).pack(side=tk.RIGHT)
        return inner

    def _crear_panel_izq_scroll(self, paned, width=445):
        """Crea un panel izquierdo con scroll suave para que las tarjetas respiren con aire."""
        cont = tk.Frame(paned, bg=BG_DARK, width=width)
        paned.add(cont, minsize=width - 25)

        sc_canvas = tk.Canvas(cont, bg=BG_DARK, highlightthickness=0, bd=0)
        sc_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        izq = tk.Frame(sc_canvas, bg=BG_DARK, padx=14, pady=12)
        win = sc_canvas.create_window((0, 0), window=izq, anchor="nw")

        def _on_cfg(e):
            sc_canvas.configure(scrollregion=sc_canvas.bbox("all"))
        def _on_c_cfg(e):
            sc_canvas.itemconfig(win, width=e.width)

        izq.bind("<Configure>", _on_cfg)
        sc_canvas.bind("<Configure>", _on_c_cfg)
        sc_canvas.bind_all("<MouseWheel>", lambda e: sc_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        return izq

    def _estilizar_eje(self, ax):
        """Aplica tema claro blanco puro (#FFFFFF) con rejilla muy tenue (#E2E8F0)."""
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=TEXT_SEC, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BORDER_CARD)
        ax.spines["bottom"].set_color(BORDER_CARD)
        ax.grid(True, color=BORDER_CARD, linestyle="-", alpha=0.8)

    def _construir_ui(self):
        # Variables del Launcher inicial
        self.launcher_puerto = tk.IntVar(value=5005)
        self.launcher_tab_rx = tk.StringVar(value="TAB4")

        # Dos contenedores principales: Launcher, Laboratorio (Emisor/Local) y Receptor dedicado
        self.frame_launcher = tk.Frame(self, bg=BG_DARK)
        self.frame_lab = tk.Frame(self, bg=BG_DARK)
        self.frame_receptor = tk.Frame(self, bg=BG_DARK)

        self._construir_launcher()
        self._construir_laboratorio()
        self._construir_receptor()

        # Mostrar primero la pantalla de bienvenida
        self.frame_launcher.pack(fill=tk.BOTH, expand=True)

    def _construir_launcher(self):
        centro = tk.Frame(self.frame_launcher, bg=BG_DARK)
        centro.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(
            centro, text="SDR Voice Radio — Laboratorio de Redes",
            font=("Segoe UI", 20, "bold"), fg=TEXT_WHITE, bg=BG_DARK
        ).pack(pady=(0, 6))

        tk.Label(
            centro,
            text="Graba o carga una señal de voz, modúlala en AM o FM y transmítela por red UDP entre dos ordenadores.",
            font=("Segoe UI", 10), fg=TEXT_SEC, bg=BG_DARK
        ).pack(pady=(0, 26))

        cols = tk.Frame(centro, bg=BG_DARK)
        cols.pack()
        cols.grid_columnconfigure(0, minsize=410)
        cols.grid_columnconfigure(1, minsize=410)

        # ── OPCIÓN A: Modo Emisor / Laboratorio Local ──
        col_a = tk.Frame(cols, bg=BG_DARK)
        col_a.grid(row=0, column=0, padx=14, sticky="nsew")

        card_a = self._crear_tarjeta(col_a, "Opción A · Modo Emisor / Laboratorio Local", estado="Audio propio", pady=0)
        tk.Label(
            card_a,
            text="Tengo o voy a grabar un audio para analizar las modulaciones o transmitir por UDP.",
            font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD, wraplength=340, justify="left"
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(card_a, text="● Grabar voz nueva (4s)", style="Green.TButton",
                   command=self._launcher_grabar_voz).pack(fill=tk.X, pady=(0, 8))

        ttk.Button(card_a, text="Cargar archivo .wav", style="Blue.TButton",
                   command=self._launcher_cargar_wav).pack(fill=tk.X, pady=(0, 8))

        ttk.Button(card_a, text="Usar audio de ejemplo", style="Blue.TButton",
                   command=self._launcher_usar_ejemplo).pack(fill=tk.X, pady=(0, 12))

        pill_a = tk.Frame(card_a, bg=BG_INNER, padx=10, pady=8)
        pill_a.pack(fill=tk.X, pady=(0, 14))
        self.lbl_launcher_audio = tk.Label(
            pill_a, text="Selecciona una fuente de audio para continuar",
            font=("Segoe UI", 8, "bold"), fg=TEXT_SEC, bg=BG_INNER, wraplength=320, justify="left"
        )
        self.lbl_launcher_audio.pack(anchor="w")

        self.btn_entrar_lab = ttk.Button(
            card_a, text="Entrar al Laboratorio  ›", style="Modern.TButton",
            command=self._entrar_laboratorio, state=tk.DISABLED
        )
        self.btn_entrar_lab.pack(fill=tk.X)

        # ── OPCIÓN B: Modo Receptor en Red ──
        col_b = tk.Frame(cols, bg=BG_DARK)
        col_b.grid(row=0, column=1, padx=14, sticky="nsew")

        card_b = self._crear_tarjeta(col_b, "Opción B · Modo Receptor en Red", estado="Esperar señal", pady=0)
        tk.Label(
            card_b,
            text="Recibe una transmisión UDP enviada desde el ordenador de tu compañera sin cargar audio previo.",
            font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD, wraplength=340, justify="left"
        ).pack(anchor="w", pady=(0, 12))

        ip_pill = tk.Frame(card_b, bg=BG_INNER, padx=10, pady=8)
        ip_pill.pack(fill=tk.X, pady=(0, 12))
        tk.Label(ip_pill, text=f"Tu IP local: {self.ip_local}", font=("Segoe UI", 9, "bold"),
                 fg=COLOR_REC, bg=BG_INNER).pack(anchor="w")
        tk.Label(ip_pill, text="Comparte esta IP con quien vaya a emitir.",
                 font=("Segoe UI", 8), fg=TEXT_SEC, bg=BG_INNER).pack(anchor="w")

        tk.Label(card_b, text="Destino de escucha:", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w", pady=(0, 4))

        tk.Radiobutton(
            card_b, text="Pestaña 4 · Red UDP & Espía (puerto 5005)",
            variable=self.launcher_tab_rx, value="TAB4",
            command=lambda: self.launcher_puerto.set(5005),
            font=("Segoe UI", 8), fg=TEXT_WHITE, bg=BG_CARD, selectcolor=BG_CARD,
            activebackground=BG_CARD, activeforeground=TEXT_WHITE
        ).pack(anchor="w")

        tk.Radiobutton(
            card_b, text="Pestaña 5 · Radio AM/FM en Red (puerto 5018)",
            variable=self.launcher_tab_rx, value="TAB5",
            command=lambda: self.launcher_puerto.set(5018),
            font=("Segoe UI", 8), fg=TEXT_WHITE, bg=BG_CARD, selectcolor=BG_CARD,
            activebackground=BG_CARD, activeforeground=TEXT_WHITE
        ).pack(anchor="w", pady=(0, 10))

        f_port = tk.Frame(card_b, bg=BG_CARD)
        f_port.pack(fill=tk.X, pady=(0, 16))
        tk.Label(f_port, text="Puerto de escucha:", font=("Segoe UI", 9),
                 fg=TEXT_SEC, bg=BG_CARD).pack(side=tk.LEFT)
        ttk.Entry(f_port, textvariable=self.launcher_puerto, width=8).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Button(
            card_b, text="Iniciar como Receptor  ›", style="Modern.TButton",
            command=self._iniciar_como_receptor
        ).pack(fill=tk.X)

    def _launcher_grabar_voz(self):
        self.lbl_launcher_audio.config(text="● Grabando desde el micrófono (4s)...", fg=ACCENT)
        self._iniciar_grabacion_voz()

    def _launcher_cargar_wav(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo de audio WAV",
            filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
        )
        if not ruta:
            return
        try:
            with wave.open(ruta, "rb") as wf:
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
            if tasa != FS and len(señal) > 1:
                dur = len(señal) / tasa
                x_old = np.linspace(0, dur, len(señal))
                x_new = np.linspace(0, dur, int(dur * FS))
                señal = np.interp(x_new, x_old, señal).astype(np.float32)
            señal -= np.mean(señal)
            pico = np.max(np.abs(señal))
            if pico > 0:
                señal = señal / pico * 0.95
            nombre = os.path.basename(ruta)
            self._on_voz_actualizada(señal, nombre)
        except Exception as e:
            messagebox.showerror("Error al cargar WAV", f"No se pudo leer el archivo:\n{e}")

    def _launcher_usar_ejemplo(self):
        self._restaurar_voz_original()

    def _entrar_laboratorio(self):
        self.frame_launcher.pack_forget()
        self.frame_receptor.pack_forget()
        self.frame_lab.pack(fill=tk.BOTH, expand=True)

    def _ir_a_inicio(self):
        self.parar_audio()
        self._detener_receptor_dedicado()
        self.frame_lab.pack_forget()
        self.frame_receptor.pack_forget()
        self.frame_launcher.pack(fill=tk.BOTH, expand=True)

    def _iniciar_como_receptor(self):
        puerto = int(self.launcher_puerto.get())
        self.rx_puerto.set(puerto)
        self.frame_launcher.pack_forget()
        self.frame_lab.pack_forget()
        self.frame_receptor.pack(fill=tk.BOTH, expand=True)
        self.after(150, self._iniciar_receptor_dedicado)

    # ────────────────────────────────────────────────────────────
    # VISTA DEDICADA DE RECEPTOR EN RED (OPCIÓN B)
    # ────────────────────────────────────────────────────────────
    def _construir_receptor(self):
        self.rx_puerto = tk.IntVar(value=5005)
        self.rx_en_curso = False
        self.rx_detener_flag = False
        self.rx_audio_recuperado = None
        self.rx_audio_espia = None

        # Header limpio de Receptor
        header = tk.Frame(self.frame_receptor, bg=BG_CARD, height=64, padx=20)
        header.pack(fill=tk.X, side=tk.TOP)
        tk.Frame(self.frame_receptor, bg=BORDER_CARD, height=1).pack(fill=tk.X, side=tk.TOP)

        titulo_box = tk.Frame(header, bg=BG_CARD)
        titulo_box.pack(side=tk.LEFT, pady=10)
        tk.Label(titulo_box, text="Modo Receptor — Escucha en Red UDP",
                 font=("Segoe UI", 12, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        tk.Label(titulo_box, text="Detecta automáticamente paquetes de Fase 4 (con máscara) o Fase 5 (AM/FM)",
                 font=("Segoe UI", 8), fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w", pady=(2, 0))

        f_hdr = tk.Frame(header, bg=BG_CARD)
        f_hdr.pack(side=tk.RIGHT, pady=12)
        ttk.Button(f_hdr, text="← Inicio / Cambiar modo", style="Blue.TButton",
                   command=self._ir_a_inicio).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(f_hdr, text="■ Detener audio", style="Stop.TButton",
                   command=self.parar_audio).pack(side=tk.LEFT)

        paned = tk.PanedWindow(self.frame_receptor, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        izq = self._crear_panel_izq_scroll(paned, width=445)

        # Paso 1: Conexión de escucha
        c1 = self._crear_tarjeta(izq, "Paso 1 · Conexión de escucha", estado="Red UDP", pady=5)
        ip_box = tk.Frame(c1, bg=BG_INNER, padx=10, pady=8)
        ip_box.pack(fill=tk.X, pady=(0, 8))
        tk.Label(ip_box, text=f"Tu IP local: {self.ip_local}", font=("Segoe UI", 10, "bold"),
                 fg=COLOR_REC, bg=BG_INNER).pack(anchor="w")
        tk.Label(ip_box, text="Dile a tu compañera que envíe a esta IP.",
                 font=("Segoe UI", 8), fg=TEXT_SEC, bg=BG_INNER).pack(anchor="w")

        f_p = tk.Frame(c1, bg=BG_CARD)
        f_p.pack(fill=tk.X, pady=(0, 10))
        tk.Label(f_p, text="Puerto de escucha:", font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD).pack(side=tk.LEFT)
        ttk.Entry(f_p, textvariable=self.rx_puerto, width=8).pack(side=tk.LEFT, padx=(8, 12))
        self.lbl_rx_estado = tk.Label(f_p, text="○ En espera", font=("Segoe UI", 8, "bold"),
                                      fg=TEXT_SEC, bg=BG_CARD)
        self.lbl_rx_estado.pack(side=tk.RIGHT)

        f_btns = tk.Frame(c1, bg=BG_CARD)
        f_btns.pack(fill=tk.X)
        self.btn_rx_iniciar = ttk.Button(f_btns, text="▶ Ponerse a la escucha", style="Modern.TButton",
                                         command=self._iniciar_receptor_dedicado)
        self.btn_rx_iniciar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.btn_rx_detener = ttk.Button(f_btns, text="■ Detener", style="Stop.TButton",
                                         command=self._detener_receptor_dedicado, state=tk.DISABLED)
        self.btn_rx_detener.pack(side=tk.RIGHT)

        # Paso 2: Paquetes recibidos
        c2 = self._crear_tarjeta(izq, "Paso 2 · Paquetes recibidos", estado="En vivo", pady=5)
        self.rx_prog = ttk.Progressbar(c2, orient=tk.HORIZONTAL, mode="determinate")
        self.rx_prog.pack(fill=tk.X, pady=(0, 8))

        self.rx_terminal = tk.Text(c2, height=6, bg=BG_TERMINAL, fg=TEXT_WHITE,
                                   font=("Consolas", 8), bd=0, padx=8, pady=6, wrap=tk.NONE)
        self.rx_terminal.pack(fill=tk.X, pady=(0, 8))
        self.rx_terminal.insert(tk.END, "[Receptor] Listo para escuchar paquetes UDP...\n")
        self.rx_terminal.config(state=tk.DISABLED)

        f_grid = tk.Frame(c2, bg=BG_CARD)
        f_grid.pack(fill=tk.X)
        for titulo, attr in [
            ("PROTOCOLO", "lbl_rx_kpi_modo"),
            ("BLOQUES", "lbl_rx_kpi_pkts"),
            ("PÉRDIDAS", "lbl_rx_kpi_perd"),
            ("JITTER", "lbl_rx_kpi_jit"),
        ]:
            box = tk.Frame(f_grid, bg=BG_INNER, padx=6, pady=6)
            box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            tk.Label(box, text=titulo, font=("Segoe UI", 6, "bold"), fg=TEXT_SEC, bg=BG_INNER).pack()
            lbl = tk.Label(box, text="--", font=("Segoe UI", 8, "bold"), fg=TEXT_WHITE, bg=BG_INNER)
            lbl.pack()
            setattr(self, attr, lbl)

        # Paso 3: Reproducir audio recibido
        c3 = self._crear_tarjeta(izq, "Paso 3 · Escuchar audio recibido", estado="Resultado", pady=5)
        f_play = tk.Frame(c3, bg=BG_CARD)
        f_play.pack(fill=tk.X)
        self.btn_rx_play = ttk.Button(f_play, text="▶ Escuchar voz recuperada", style="Green.TButton",
                                      command=self._rx_reproducir_voz, state=tk.DISABLED)
        self.btn_rx_play.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.btn_rx_espia = ttk.Button(f_play, text="Espía (sin clave)", style="Blue.TButton",
                                       command=self._rx_reproducir_espia, state=tk.DISABLED)
        self.btn_rx_espia.pack(side=tk.RIGHT)

        # Panel Derecho: Gráficas de audio recibido
        der = tk.Frame(paned, bg=BG_DARK, padx=10, pady=10)
        paned.add(der, minsize=500)
        self.fig_rx = Figure(figsize=(7, 5), dpi=120, facecolor=BG_CARD)
        self.canvas_rx = FigureCanvasTkAgg(self.fig_rx, master=der)
        self.canvas_rx.get_tk_widget().configure(highlightthickness=0, bd=0)
        self.canvas_rx.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._plot_rx_inicial()

    def _plot_rx_inicial(self):
        self.fig_rx.clf()
        ax = self.fig_rx.add_subplot(1, 1, 1)
        self._estilizar_eje(ax)
        ax.text(0.5, 0.5, "Esperando señal de voz desde otro ordenador...\n\nCuando lleguen los paquetes, aquí verás la onda y el espectro recuperado.",
                ha="center", va="center", fontsize=10, color=TEXT_SEC)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.fig_rx.tight_layout(pad=2.0)
        self.canvas_rx.draw()

    def _iniciar_receptor_dedicado(self):
        if self.rx_en_curso:
            return
        self.rx_en_curso = True
        self.rx_detener_flag = False
        puerto = int(self.rx_puerto.get())

        self.btn_rx_iniciar.config(state=tk.DISABLED)
        self.btn_rx_detener.config(state=tk.NORMAL)
        self.btn_rx_play.config(state=tk.DISABLED)
        self.btn_rx_espia.config(state=tk.DISABLED)
        self.lbl_rx_estado.config(text=f"● Escuchando :{puerto}", fg=ACCENT)
        self.rx_prog["value"] = 0

        self.rx_terminal.config(state=tk.NORMAL)
        self.rx_terminal.delete("1.0", tk.END)
        self.rx_terminal.insert(tk.END, f"[Receptor] Escuchando en 0.0.0.0:{puerto} (IP: {self.ip_local})\n")
        self.rx_terminal.config(state=tk.DISABLED)
        self._plot_rx_inicial()

        threading.Thread(target=self._hilo_receptor_dedicado, args=(puerto,), daemon=True).start()

    def _detener_receptor_dedicado(self):
        if self.rx_en_curso:
            self.rx_detener_flag = True

    def _rx_log(self, msg):
        self.rx_terminal.config(state=tk.NORMAL)
        self.rx_terminal.insert(tk.END, msg + "\n")
        self.rx_terminal.see(tk.END)
        self.rx_terminal.config(state=tk.DISABLED)

    def _rx_notificar_pkt(self, n_rx, total_est, proto_txt, jitter_ms, log_txt):
        self.rx_prog["value"] = min(100.0, (n_rx / max(1, total_est)) * 100)
        self.lbl_rx_kpi_modo.config(text=proto_txt)
        self.lbl_rx_kpi_pkts.config(text=f"{n_rx}/{total_est}")
        self.lbl_rx_kpi_perd.config(text="0")
        self.lbl_rx_kpi_jit.config(text=f"{jitter_ms:.2f} ms")
        if log_txt:
            self._rx_log(log_txt)

    def _rx_finalizar(self, audio_rec, audio_espia, proto_txt, recibidos, esperados, jitter_ms):
        self.rx_en_curso = False
        self.rx_audio_recuperado = audio_rec
        self.rx_audio_espia = audio_espia
        self.btn_rx_iniciar.config(state=tk.NORMAL)
        self.btn_rx_detener.config(state=tk.DISABLED)
        self.btn_rx_play.config(state=tk.NORMAL)
        if audio_espia is not None:
            self.btn_rx_espia.config(state=tk.NORMAL)
        self.lbl_rx_estado.config(text="● Audio recibido", fg=COLOR_REC)
        perd = max(0, esperados - recibidos)
        self.lbl_rx_kpi_pkts.config(text=f"{recibidos}/{esperados}")
        self.lbl_rx_kpi_perd.config(text=f"{perd}", fg=COLOR_ALERT if perd > 0 else COLOR_REC)
        self._rx_log(f"[OK] Recepción completada ({recibidos}/{esperados} bloques, {proto_txt}).")

        # Dibujar forma de onda y espectro del audio recibido
        self.fig_rx.clf()
        ax1 = self.fig_rx.add_subplot(2, 1, 1)
        ax2 = self.fig_rx.add_subplot(2, 1, 2)
        self._estilizar_eje(ax1)
        self._estilizar_eje(ax2)

        n_win = min(len(audio_rec), int(0.040 * FS))
        peak_idx = int(np.argmax(np.abs(audio_rec)))
        i0 = max(0, min(peak_idx - n_win // 3, len(audio_rec) - n_win))
        i1 = i0 + n_win
        t_ms = (np.arange(i0, i1) / FS) * 1000
        ax1.plot(t_ms, audio_rec[i0:i1], color=COLOR_REC, lw=1.2)
        ax1.set_title(f"Voz recibida por red ({proto_txt})", fontsize=9.5, fontweight="bold", color=TEXT_WHITE, loc="left", pad=8)
        ax1.set_xlabel("Tiempo (ms)", color=TEXT_SEC, fontsize=8)
        ax1.set_ylabel("Amplitud", color=TEXT_SEC, fontsize=8)
        ax1.set_ylim(-1.15, 1.15)

        N = len(audio_rec)
        mags = 20 * np.log10(np.abs(np.fft.rfft(audio_rec)) / max(1, N) + 1e-12)
        freqs = np.fft.rfftfreq(N, 1 / FS) / 1000
        ax2.plot(freqs, mags, color=ACCENT, lw=1.0)
        ax2.set_title("Espectro de frecuencias recibido (0 - 8 kHz)", fontsize=9.5, fontweight="bold", color=TEXT_WHITE, loc="left", pad=8)
        ax2.set_xlabel("Frecuencia (kHz)", color=TEXT_SEC, fontsize=8)
        ax2.set_ylabel("Magnitud (dB)", color=TEXT_SEC, fontsize=8)
        ax2.set_xlim(0, 8)

        self.fig_rx.tight_layout(pad=2.0)
        self.canvas_rx.draw()

    def _rx_cancelar(self):
        self.rx_en_curso = False
        self.btn_rx_iniciar.config(state=tk.NORMAL)
        self.btn_rx_detener.config(state=tk.DISABLED)
        self.lbl_rx_estado.config(text="○ Detenido", fg=TEXT_SEC)
        self._rx_log("[Receptor] Escucha detenida.")

    def _rx_error(self, err):
        self.rx_en_curso = False
        self.btn_rx_iniciar.config(state=tk.NORMAL)
        self.btn_rx_detener.config(state=tk.DISABLED)
        self.lbl_rx_estado.config(text="● Error", fg=COLOR_ALERT)
        self._rx_log(f"[Error] {err}")

    def _rx_reproducir_voz(self):
        if self.rx_audio_recuperado is not None:
            self.reproducir_audio(self.rx_audio_recuperado)

    def _rx_reproducir_espia(self):
        if self.rx_audio_espia is not None:
            self._rx_log("[Espía] Reproduciendo tráfico sin clave: solo se oye música 8-bits.")
            self.reproducir_audio(self.rx_audio_espia)

    def _hilo_receptor_dedicado(self, puerto: int):
        HDR5_FMT = "!4sIIdBffH"
        HDR5_SIZE = struct.calcsize(HDR5_FMT)
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            sock.bind(("0.0.0.0", puerto))
            sock.settimeout(0.25)

            proto = None
            bloques_rx = {}
            raw_espia = []
            ultimo_t = None
            total_est = 601
            modo_5 = "FM"
            fc_5 = FC_FM_DEFAULT
            param_5 = KF_DEFAULT
            jitter = 0.0
            prev_rx = 0.0
            prev_tx = 0.0

            while not self.rx_detener_flag:
                try:
                    data, addr = sock.recvfrom(4096)
                    t_rx = time.time()

                    if len(data) >= HDR5_SIZE and data[:4] == b"SDR5":
                        mag, s_num, t_tot, ts, mc, f_c, p_val, pay_len = struct.unpack(HDR5_FMT, data[:HDR5_SIZE])
                        if mc == 99:
                            break
                        proto = "F5"
                        modo_5 = "FM" if mc == 1 else "AM"
                        fc_5, param_5, total_est = f_c, p_val, t_tot
                        muestras = deserializar_bloque(data[HDR5_SIZE:HDR5_SIZE + pay_len])
                        if ultimo_t is None:
                            self.cola_eventos.put(("RX_LOG", (f"[RX] Flujo Fase 5 ({modo_5}) desde {addr[0]}:{addr[1]}",)))
                        ultimo_t = t_rx
                        if prev_rx > 0 and ts > 0:
                            jitter += (abs((t_rx - prev_rx) - (ts - prev_tx)) - jitter) / 16.0
                        prev_rx, prev_tx = t_rx, ts
                        bloques_rx[s_num] = muestras
                        n_rx = len(bloques_rx)
                        log_txt = f"[RX-{modo_5}] Bloque #{s_num:03d}/{total_est} ({len(data)}B)" if (n_rx == 1 or n_rx % 25 == 0) else None
                        self.cola_eventos.put(("RX_PKT", (n_rx, total_est, f"F5 {modo_5}", jitter * 1000.0, log_txt)))
                    else:
                        pkt = desempaquetar(data)
                        if pkt["seq"] == 0xFFFFFFFF:
                            break
                        proto = "F4"
                        if ultimo_t is None:
                            self.cola_eventos.put(("RX_LOG", (f"[RX] Flujo Fase 4 (UDP) desde {addr[0]}:{addr[1]}",)))
                        ultimo_t = t_rx
                        ts = pkt["timestamp"]
                        if prev_rx > 0 and ts > 0:
                            jitter += (abs((t_rx - prev_rx) - (ts - prev_tx)) - jitter) / 16.0
                        prev_rx, prev_tx = t_rx, ts
                        if pkt["mask"] == MASK_ON:
                            raw_espia.append(pkt["muestras"])
                            muestras = retirar_mascara(pkt["muestras"], self.mascara, n_seq=pkt["seq"], bloque=SAMPLES_PER_BLOCK)
                        else:
                            muestras = pkt["muestras"]
                        bloques_rx[pkt["seq"]] = muestras
                        n_rx = len(bloques_rx)
                        log_txt = f"[RX-UDP] Paquete #{pkt['seq']:03d} ({len(data)}B)" if (n_rx == 1 or n_rx % 25 == 0) else None
                        self.cola_eventos.put(("RX_PKT", (n_rx, max(641, n_rx), "F4 UDP", jitter * 1000.0, log_txt)))
                except socket.timeout:
                    if ultimo_t is not None and (time.time() - ultimo_t) > 1.8:
                        break
                    continue
                except ValueError:
                    continue

            if self.rx_detener_flag or not bloques_rx:
                self.cola_eventos.put(("RX_CANCEL", ()))
                return

            seq_min, seq_max = min(bloques_rx.keys()), max(bloques_rx.keys())
            esperados = max(total_est, seq_max - seq_min + 1) if proto == "F5" else (seq_max - seq_min + 1)
            partes = []
            for s in range(seq_min, seq_max + 1):
                if s in bloques_rx:
                    partes.append(bloques_rx[s])
                else:
                    partes.append(partes[-1] * 0.5 if partes else np.zeros(512, dtype=np.float32))
            señal_rx = np.concatenate(partes)

            if proto == "F5":
                if modo_5 == "AM":
                    audio_rec, _ = demodular_am_bloque(señal_rx, fc=fc_5, ka=param_5, fs=FS, sample_offset=0)
                else:
                    audio_rec = demodular_fm_bloque(señal_rx, fc=fc_5, kf=param_5, fs=FS)
                espia_sig = None
                proto_lbl = f"Fase 5 ({modo_5})"
            else:
                audio_rec = demodular_fm(señal_rx)
                espia_sig = np.concatenate(raw_espia) if raw_espia else None
                proto_lbl = "Fase 4 (UDP + Máscara)" if espia_sig is not None else "Fase 4 (UDP)"

            self.cola_eventos.put(("RX_FIN", (audio_rec, espia_sig, proto_lbl, len(bloques_rx), esperados, jitter * 1000.0)))
        except Exception as e:
            self.cola_eventos.put(("RX_ERROR", (f"{type(e).__name__}: {e}",)))
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    def _construir_laboratorio(self):
        # Header superior limpio
        header = tk.Frame(self.frame_lab, bg=BG_CARD, height=64, padx=20)
        header.pack(fill=tk.X, side=tk.TOP)
        tk.Frame(self.frame_lab, bg=BORDER_CARD, height=1).pack(fill=tk.X, side=tk.TOP)

        titulo_box = tk.Frame(header, bg=BG_CARD)
        titulo_box.pack(side=tk.LEFT, pady=10)

        tk.Label(titulo_box, text="SDR Voice Radio — Laboratorio de Redes",
                 font=("Segoe UI", 12, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        dur_s = len(self.voz) / FS
        self.lbl_header_sub = tk.Label(
            titulo_box,
            text=f"Audio activo: audio_original.wav ({dur_s:.1f} s a 48 kHz)  •  Portadora fc = 8 kHz  •  UDP",
            font=("Segoe UI", 8), fg=TEXT_SEC, bg=BG_CARD
        )
        self.lbl_header_sub.pack(anchor="w", pady=(2, 0))

        f_hdr_btns = tk.Frame(header, bg=BG_CARD)
        f_hdr_btns.pack(side=tk.RIGHT, pady=12)

        ttk.Button(f_hdr_btns, text="← Inicio / Cambiar modo", style="Blue.TButton",
                   command=self._ir_a_inicio).pack(side=tk.LEFT, padx=(0, 8))

        self.btn_grabar_voz = ttk.Button(f_hdr_btns, text="● Grabar voz (4s)", style="Green.TButton",
                                         command=self._iniciar_grabacion_voz)
        self.btn_grabar_voz.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(f_hdr_btns, text="↺ Audio original", style="Blue.TButton",
                   command=self._restaurar_voz_original).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(f_hdr_btns, text="■ Detener audio", style="Stop.TButton",
                   command=self.parar_audio).pack(side=tk.LEFT)

        # Notebook con las 5 pestañas claras
        self.notebook = ttk.Notebook(self.frame_lab)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        self.tab_am = ttk.Frame(self.notebook)
        self.tab_fm = ttk.Frame(self.notebook)
        self.tab_ruido = ttk.Frame(self.notebook)
        self.tab_red = ttk.Frame(self.notebook)
        self.tab_fase5 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_am, text="  1. Modulación AM  ")
        self.notebook.add(self.tab_fm, text="  2. Modulación FM  ")
        self.notebook.add(self.tab_ruido, text="  3. AM vs FM (Ruido)  ")
        self.notebook.add(self.tab_red, text="  4. Red UDP & Espía  ")
        self.notebook.add(self.tab_fase5, text="  5. Radio AM/FM en Red  ")

        self._crear_tab_am()
        self._crear_tab_fm()
        self._crear_tab_ruido()
        self._crear_tab_red()
        self._crear_tab_fase5()

    # ────────────────────────────────────────────────────────────
    # PESTAÑA 1: AM PASO A PASO
    # ────────────────────────────────────────────────────────────
    def _crear_tab_am(self):
        paned = tk.PanedWindow(self.tab_am, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)

        izq = self._crear_panel_izq_scroll(paned, width=445)

        tk.Label(izq, text="Cadena de Audio AM", font=("Segoe UI", 12, "bold"),
                 fg=TEXT_WHITE, bg=BG_DARK).pack(anchor="w", pady=(0, 10))

        # Paso 1
        card1 = self._crear_tarjeta(izq, "Paso 1 · Voz original", acento=COLOR_VOZ, pady=4)
        tk.Label(card1, text="Voz en frecuencias bajas (0 a 4 kHz). Escucha cómo suena la persona hablando.",
                 font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card1, text="▶ 1. Escuchar Voz Original", style="Modern.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_am, self.fig_am, self.voz,
                       "Paso 1: Voz Original (Bajas Frecuencias 0-4 kHz)",
                       COLOR_VOZ, self.lbl_badge_am, self.lbl_desc_am,
                       "Estado: Voz Humana Natural", COLOR_VOZ,
                       "Escuchas a la persona hablando en su frecuencia normal. En la gráfica inferior ves que toda la energía está concentrada por debajo de los 4 kHz.",
                       self.voz
                   )).pack(fill=tk.X)

        # Paso 2
        card2 = self._crear_tarjeta(izq, "Paso 2 · Modulación AM", acento=COLOR_AM, pady=4)
        tk.Label(card2, text="Montamos la voz sobre una portadora de 8.000 Hz. El volumen de la onda dibuja la voz.",
                 font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card2, text="▶ 2. Escuchar Señal AM Modulada", style="AM.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_am, self.fig_am, self.am_mod,
                       "Paso 2: Señal AM Modulada (Viajando a 8.000 Hz)",
                       COLOR_AM, self.lbl_badge_am, self.lbl_desc_am,
                       "Nota: Pitido Portadora a 8.000 Hz", COLOR_AM,
                       "¡Escuchas un pitido agudo! No suena a voz humana porque la información está escondida en la envolvente (amplitud) a 8 kHz. Así viaja la radio.",
                       self.am_mod
                   )).pack(fill=tk.X)

        # Paso 3
        card3 = self._crear_tarjeta(izq, "Paso 3 · Voz recuperada", acento=COLOR_REC, pady=4)
        tk.Label(card3, text="El receptor multiplica por 8 kHz y filtra el residuo agudo para rescatar la voz.",
                 font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card3, text="▶ 3. Escuchar Voz Recuperada", style="Green.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_am, self.fig_am, self.am_rec,
                       "Paso 3: Voz Recuperada por el Receptor AM",
                       COLOR_REC, self.lbl_badge_am, self.lbl_desc_am,
                       "Completado: Voz Rescatada y Limpia", COLOR_REC,
                       "El receptor eliminó la portadora de 8 kHz y el filtro paso-bajo dejó pasar únicamente la voz. ¡Vuelves a entender a la persona con total claridad!",
                       self.am_rec
                   )).pack(fill=tk.X)

        # Caja info
        info_box = tk.Frame(izq, bg=BG_DARK, padx=10, pady=10)
        info_box.pack(fill=tk.X, pady=(15, 0))
        self.lbl_badge_am = tk.Label(info_box, text="Paso 1 · Selecciona una acción", font=("Segoe UI", 9, "bold"),
                                     fg=COLOR_VOZ, bg=BG_DARK)
        self.lbl_badge_am.pack(anchor="w")
        self.lbl_desc_am = tk.Label(info_box, text="Haz clic en los tres botones para ver y escuchar cómo la voz se transforma para viajar por el aire y se reconstruye en el receptor.",
                                    font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_DARK, wraplength=310, justify="left")
        self.lbl_desc_am.pack(anchor="w", pady=(4, 0))

        # Panel derecho: Canvas
        der = tk.Frame(paned, bg=BG_DARK, padx=10, pady=10)
        paned.add(der, minsize=500)

        self.fig_am = Figure(figsize=(7, 5), dpi=120, facecolor=BG_CARD)
        self.canvas_am = FigureCanvasTkAgg(self.fig_am, master=der)
        self.canvas_am.get_tk_widget().configure(highlightthickness=0, bd=0)
        self.canvas_am.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._configurar_toolbar(self.canvas_am, der)

        self.mostrar_grafica(self.canvas_am, self.fig_am, self.voz, "Paso 1: Voz Original (Bajas Frecuencias)", COLOR_VOZ)

    # ────────────────────────────────────────────────────────────
    # PESTAÑA 2: FM PASO A PASO
    # ────────────────────────────────────────────────────────────
    def _crear_tab_fm(self):
        paned = tk.PanedWindow(self.tab_fm, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)

        izq = self._crear_panel_izq_scroll(paned, width=445)

        tk.Label(izq, text="Cadena de Audio FM", font=("Segoe UI", 12, "bold"),
                 fg=TEXT_WHITE, bg=BG_DARK).pack(anchor="w", pady=(0, 10))

        # Paso 1
        card1 = self._crear_tarjeta(izq, "Paso 1 · Voz original", acento=COLOR_VOZ, pady=4)
        tk.Label(card1, text="Voz en banda base (0 a 4 kHz). Esta es la señal moduladora que viaja en la frecuencia.",
                 font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card1, text="▶ 1. Escuchar Voz Original", style="Modern.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_fm, self.fig_fm, self.voz,
                       "Paso 1: Voz Original (Banda Base 0-4 kHz)",
                       COLOR_VOZ, self.lbl_badge_fm, self.lbl_desc_fm,
                       "Estado: Voz Humana Natural", COLOR_VOZ,
                       "La voz humana original. Observa en la gráfica inferior que el ancho de banda ocupa estrictamente hasta los 4 kHz.",
                       self.voz
                   )).pack(fill=tk.X)

        # Paso 2
        card2 = self._crear_tarjeta(izq, "Paso 2 · Modulación FM", acento=COLOR_FM, pady=4)
        tk.Label(card2, text="La amplitud es constante (inmune al ruido). La voz hace que la onda se comprima y se expanda.",
                 font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card2, text="▶ 2. Escuchar Señal FM Modulada", style="FM.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_fm, self.fig_fm, self.fm_mod,
                       "Paso 2: Señal FM Modulada (8.000 Hz ± 3.500 Hz)",
                       COLOR_FM, self.lbl_badge_fm, self.lbl_desc_fm,
                       "Nota: Sonido Sintético Modulado", COLOR_FM,
                       "Escuchas cómo el tono oscila rápidamente. En FM la información va en la velocidad de vibración, no en el volumen. ¡Por eso es inmune a las interferencias!",
                       self.fm_mod
                   )).pack(fill=tk.X)

        # Paso 3
        card3 = self._crear_tarjeta(izq, "Paso 3 · Voz recuperada", acento=COLOR_REC, pady=4)
        tk.Label(card3, text="El receptor mide los giros de fase instántaneos (Hilbert) y rescata la voz original limpia.",
                 font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card3, text="▶ 3. Escuchar Voz Recuperada FM", style="Green.TButton",
                   command=lambda: self.mostrar_y_reproducir(
                       self.canvas_fm, self.fig_fm, self.fm_rec,
                       "Paso 3: Voz Recuperada por el Receptor FM",
                       COLOR_REC, self.lbl_badge_fm, self.lbl_desc_fm,
                       "Completado: Voz FM Rescatada a la Perfección", COLOR_REC,
                       "¡El receptor FM extrajo la variación de frecuencia y aplicó el desénfasis! La voz suena con un timbre cálido, limpio y sin siseos molestos.",
                       self.fm_rec
                   )).pack(fill=tk.X)

        info_box = tk.Frame(izq, bg=BG_DARK, padx=10, pady=10)
        info_box.pack(fill=tk.X, pady=(15, 0))
        self.lbl_badge_fm = tk.Label(info_box, text="Paso 1 · Selecciona una acción", font=("Segoe UI", 9, "bold"),
                                     fg=COLOR_VOZ, bg=BG_DARK)
        self.lbl_badge_fm.pack(anchor="w")
        self.lbl_desc_fm = tk.Label(info_box, text="FM mantiene la amplitud fija en 1.0 y varía la frecuencia. Esto permite eliminar cualquier ruido electromagnético cortando los picos.",
                                    font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_DARK, wraplength=310, justify="left")
        self.lbl_desc_fm.pack(anchor="w", pady=(4, 0))

        der = tk.Frame(paned, bg=BG_DARK, padx=10, pady=10)
        paned.add(der, minsize=500)

        self.fig_fm = Figure(figsize=(7, 5), dpi=120, facecolor=BG_CARD)
        self.canvas_fm = FigureCanvasTkAgg(self.fig_fm, master=der)
        self.canvas_fm.get_tk_widget().configure(highlightthickness=0, bd=0)
        self.canvas_fm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._configurar_toolbar(self.canvas_fm, der)

        self.mostrar_grafica(self.canvas_fm, self.fig_fm, self.voz, "Paso 1: Voz Original (Banda Base)", COLOR_VOZ)

    # ────────────────────────────────────────────────────────────
    # PESTAÑA 3: COMPARATIVA CON RUIDO
    # ────────────────────────────────────────────────────────────
    def _crear_tab_ruido(self):
        paned = tk.PanedWindow(self.tab_ruido, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)

        izq = self._crear_panel_izq_scroll(paned, width=445)

        tk.Label(izq, text="Prueba de Canal Ruidoso", font=("Segoe UI", 12, "bold"),
                 fg=TEXT_WHITE, bg=BG_DARK).pack(anchor="w")
        tk.Label(izq, text="Inyectamos el mismo ruido electromagnético de radio a ambas señales. ¡Compara cómo suena cada una!",
                 font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD, wraplength=350, justify="left").pack(anchor="w", pady=(4, 12))

        # Tarjeta AM
        card_am = self._crear_tarjeta(izq, "Paso 1 · Recepción AM con ruido", acento=COLOR_AM, pady=4)
        tk.Label(card_am, text="AM guarda la voz en la amplitud. Como el ruido modifica la amplitud, se oye un siseo constante.",
                 font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD, wraplength=340, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card_am, text="▶ Escuchar AM con ruido", style="AM.TButton",
                   command=self._accion_am_ruido).pack(fill=tk.X)

        # Tarjeta FM
        card_fm = self._crear_tarjeta(izq, "Paso 2 · Recepción FM con ruido", acento=COLOR_FM, pady=4)
        tk.Label(card_fm, text="FM guarda la voz en la frecuencia. El limitador del receptor corta el ruido de amplitud y la voz suena limpia.",
                 font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD, wraplength=340, justify="left").pack(anchor="w", pady=(0, 6))
        ttk.Button(card_fm, text="▶ Escuchar FM con ruido", style="FM.TButton",
                   command=self._accion_fm_ruido).pack(fill=tk.X)

        # Resumen explicativo
        resumen_box = tk.Frame(izq, bg=BG_DARK, padx=12, pady=12)
        resumen_box.pack(fill=tk.X, pady=(15, 0))
        self.lbl_titulo_resumen = tk.Label(resumen_box, text="Por qué elegir FM",
                                           font=("Segoe UI", 9, "bold"), fg="#4F46E5", bg=BG_DARK)
        self.lbl_titulo_resumen.pack(anchor="w")
        self.lbl_texto_resumen = tk.Label(
            resumen_box,
            text="Pulsa en los botones para escuchar y ver la diferencia.\n\n"
                 "• En AM escucharás la voz tapada por ruido.\n"
                 "• En FM la voz sigue sonando clara y nítida.\n\n"
                 "Por este motivo, para la Fase 3 transmitimos la radio FM paquetizada por sockets UDP.",
            font=("Segoe UI", 8), fg=TEXT_WHITE, bg=BG_DARK, wraplength=330, justify="left"
        )
        self.lbl_texto_resumen.pack(anchor="w", pady=(5, 0))

        der = tk.Frame(paned, bg=BG_DARK, padx=10, pady=10)
        paned.add(der, minsize=500)

        self.fig_ruido = Figure(figsize=(7, 5), dpi=120, facecolor=BG_CARD)
        self.canvas_ruido = FigureCanvasTkAgg(self.fig_ruido, master=der)
        self.canvas_ruido.get_tk_widget().configure(highlightthickness=0, bd=0)
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

        izq = self._crear_panel_izq_scroll(paned, width=445)

        tk.Label(izq, text="Fase 4: Transporte UDP y Esteganografía", font=("Segoe UI", 12, "bold"),
                 fg=TEXT_WHITE, bg=BG_DARK).pack(anchor="w")
        tk.Label(izq, text="Envía la señal por paquetes UDP (en local o entre 2 PCs) camuflada bajo una melodía musical.",
                 font=("Segoe UI", 8), fg=TEXT_SEC, bg=BG_CARD, wraplength=400, justify="left").pack(anchor="w", pady=(2, 6))

        # Tarjeta 0: Conexión de Red (Local vs PC a PC)
        card_red = self._crear_tarjeta(izq, " Paso 1 · Conexión de red ", acento="#4F46E5", pady=4)

        f_roles = tk.Frame(card_red, bg=BG_CARD)
        f_roles.pack(fill=tk.X)

        for txt, val, col in [
            ("Local (1 PC)", "LOCAL", TEXT_WHITE),
            ("Emisor", "EMISOR", "#475569"),
            ("Receptor", "RECEPTOR", COLOR_REC),
        ]:
            tk.Radiobutton(
                f_roles, text=txt, variable=self.f4_modo_red, value=val,
                command=self._on_f4_red_change, font=("Segoe UI", 8, "bold"),
                fg=col, bg=BG_CARD, selectcolor=BG_CARD, activebackground=BG_CARD, activeforeground=TEXT_WHITE
            ).pack(side=tk.LEFT, padx=(0, 6))

        self.f4_net_info = tk.Frame(card_red, bg=BG_DARK, padx=8, pady=5)
        self.f4_net_info.pack(fill=tk.X, pady=(4, 0))

        # Tarjeta 1: Controles de Emisión
        card_ctrl = self._crear_tarjeta(izq, " Paso 2 · Control de emisión ", acento=COLOR_VOZ, pady=4)

        f_cfg = tk.Frame(card_ctrl, bg=BG_CARD)
        f_cfg.pack(fill=tk.X, pady=(0, 5))

        self.var_mascara = tk.BooleanVar(value=True)
        chk_mask = tk.Checkbutton(f_cfg, text="Cifrado musical (esteganografía)",
                                  variable=self.var_mascara, font=("Segoe UI", 8, "bold"),
                                  fg="#4F46E5", bg=BG_CARD, selectcolor=BG_CARD,
                                  activebackground=BG_CARD, activeforeground="#4F46E5")
        chk_mask.pack(side=tk.LEFT)

        self.var_modo_rt = tk.BooleanVar(value=True)
        chk_rt = tk.Checkbutton(f_cfg, text="Tiempo real",
                                variable=self.var_modo_rt, font=("Segoe UI", 8),
                                fg=TEXT_WHITE, bg=BG_CARD, selectcolor=BG_CARD,
                                activebackground=BG_CARD, activeforeground=TEXT_WHITE)
        chk_rt.pack(side=tk.RIGHT)

        f_btns = tk.Frame(card_ctrl, bg=BG_CARD)
        f_btns.pack(fill=tk.X)

        self.btn_iniciar_tx = ttk.Button(f_btns, text="▶ Iniciar transmisión", style="Purple.TButton",
                                         command=self._iniciar_transmision_live)
        self.btn_iniciar_tx.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_detener_tx = ttk.Button(f_btns, text="■ Detener", style="Stop.TButton",
                                         command=self._detener_transmision_live, state=tk.DISABLED)
        self.btn_detener_tx.pack(side=tk.RIGHT)

        # Tarjeta 2: Monitor de Terminal de Datagramas en Vivo
        card_feed = self._crear_tarjeta(izq, " Paso 3 · Tráfico en vivo ", acento="#4F46E5", pady=4)

        self.feed_terminal = tk.Text(card_feed, height=5, bg=BG_INNER, fg="#4F46E5",
                                     font=("Consolas", 8), bd=0, padx=6, pady=4, wrap=tk.NONE)
        self.feed_terminal.pack(fill=tk.BOTH, expand=True)
        self.feed_terminal.insert(tk.END, "[SDR-NET] Esperando inicio de transmisión...\n[CONFIG] Puerto UDP 5005 | MTU=1500B | Bloque=987B\n")
        self.feed_terminal.config(state=tk.DISABLED)

        # Tarjeta 3: Telemetría de Red en Vivo (KPIs)
        card_kpis = self._crear_tarjeta(izq, " Métricas de red (RFC 3550) ", acento=COLOR_REC, pady=4)

        self.prog_tx = ttk.Progressbar(card_kpis, orient=tk.HORIZONTAL, mode="determinate")
        self.prog_tx.pack(fill=tk.X, pady=(0, 5))

        f_grid = tk.Frame(card_kpis, bg=BG_CARD)
        f_grid.pack(fill=tk.X)

        for col, (titulo, attr) in enumerate([
            ("PAQUETES", "lbl_kpi_pkts"),
            ("PÉRDIDAS", "lbl_kpi_perd"),
            ("RETARDO", "lbl_kpi_ret"),
            ("JITTER", "lbl_kpi_jit"),
            ("VELOCIDAD", "lbl_kpi_kbps")
        ]):
            box = tk.Frame(f_grid, bg=BG_INNER, padx=4, pady=4)
            box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            tk.Label(box, text=titulo, font=("Segoe UI", 6, "bold"), fg=TEXT_MUTED, bg=BG_INNER).pack()
            lbl = tk.Label(box, text="--", font=("Segoe UI", 8, "bold"), fg=TEXT_WHITE, bg=BG_INNER)
            lbl.pack()
            setattr(self, attr, lbl)

        # Tarjeta 4: Verificación Auditiva + Demo Interceptor
        card_audio = self._crear_tarjeta(izq, " Paso 4 · Espía vs Receptor ", acento="#4F46E5", pady=4)

        tk.Label(card_audio, text="Enseña al profesor qué oye un atacante sin clave vs el receptor legítimo:",
                 font=("Segoe UI", 7), fg=TEXT_SEC, bg=BG_CARD, justify="left", wraplength=390).pack(anchor="w", pady=(0, 4))

        f_play_btns = tk.Frame(card_audio, bg=BG_CARD)
        f_play_btns.pack(fill=tk.X)

        self.btn_play_mask = ttk.Button(f_play_btns, text="Espía (sin clave)", style="Pink.TButton",
                                        command=self._reproducir_audio_enmascarado)
        self.btn_play_mask.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        self.btn_play_rec = ttk.Button(f_play_btns, text="▶ Voz recuperada", style="Green.TButton",
                                       command=self._reproducir_audio_red, state=tk.DISABLED)
        self.btn_play_rec.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))

        self._on_f4_red_change()

        # Panel Derecho: Gráfica Canvas
        der = tk.Frame(paned, bg=BG_DARK, padx=10, pady=10)
        paned.add(der, minsize=500)

        self.fig_red = Figure(figsize=(7, 5), dpi=120, facecolor=BG_CARD)
        self.canvas_red = FigureCanvasTkAgg(self.fig_red, master=der)
        self.canvas_red.get_tk_widget().configure(highlightthickness=0, bd=0)
        self.canvas_red.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._configurar_toolbar(self.canvas_red, der)

        self._plot_red_inicial()

    def _plot_red_inicial(self):
        self.fig_red.clf()
        self.ax_scope = self.fig_red.add_subplot(2, 1, 1)
        self.ax_net = self.fig_red.add_subplot(2, 1, 2)

        for ax in (self.ax_scope, self.ax_net):
            self._estilizar_eje(ax)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#E2E8F0")

        # Subplot 1: Osciloscopio del datagrama actual
        t_ms_pkt = np.arange(SAMPLES_PER_BLOCK) / FS * 1000
        self.line_scope, = self.ax_scope.plot(t_ms_pkt, np.zeros(SAMPLES_PER_BLOCK),
                                              color=COLOR_REC, lw=1.2, label="Datagrama 10 ms (480 muestras int16)")
        self.ax_scope.set_title("Osciloscopio del Datagrama UDP Actual (Pulsa 'Iniciar Transmisión en Vivo')",
                                fontsize=9, fontweight="bold", color=TEXT_WHITE)
        self.ax_scope.set_xlabel("Tiempo dentro del datagrama (ms)", color=TEXT_MUTED, fontsize=8)
        self.ax_scope.set_ylabel("Amplitud [-1, 1]", color=TEXT_MUTED, fontsize=8)
        self.ax_scope.set_ylim(-1.15, 1.15)
        self.ax_scope.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)
        self.ax_scope.legend(loc="upper right", facecolor=BG_CARD, edgecolor=BORDER_CARD, labelcolor=TEXT_WHITE, fontsize=7)

        # Subplot 2: Dinámica de Red
        self.ax_net.set_title("Telemetría de Red — Retardo y Jitter por Datagrama en Tiempo Real (RFC 3550)",
                              fontsize=9, fontweight="bold", color=TEXT_WHITE)
        self.ax_net.set_xlabel("Número de Secuencia del Paquete", color=TEXT_MUTED, fontsize=8)
        self.ax_net.set_ylabel("Milisegundos (ms)", color=TEXT_MUTED, fontsize=8)
        self.ax_net.set_xlim(0, 641)
        self.ax_net.set_ylim(0, 2.5)
        self.line_latencia, = self.ax_net.plot([], [], color=COLOR_CYAN, marker='.', markersize=3, linestyle='', label="Retardo (ms)")
        self.line_jitter, = self.ax_net.plot([], [], color="#475569", lw=1.2, label="Jitter RFC 3550 (ms)")
        self.ax_net.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)
        self.ax_net.legend(loc="upper right", facecolor=BG_CARD, edgecolor=BORDER_CARD, labelcolor=TEXT_WHITE, fontsize=7)

        self.fig_red.tight_layout(pad=2.0)
        self.canvas_red.draw()

    def _on_f4_red_change(self):
        for w in self.f4_net_info.winfo_children():
            w.destroy()
        rol = self.f4_modo_red.get()
        if rol == "LOCAL":
            tk.Label(self.f4_net_info, text="Bucle local (127.0.0.1:5005) para pruebas en un equipo.",
                     font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_DARK).pack(anchor="w")
            self.btn_iniciar_tx.config(text="▶ Iniciar transmisión")
        elif rol == "EMISOR":
            f_row = tk.Frame(self.f4_net_info, bg=BG_DARK)
            f_row.pack(fill=tk.X)
            tk.Label(f_row, text="IP Destino:", font=("Segoe UI", 8, "bold"), fg="#475569", bg=BG_DARK).pack(side=tk.LEFT)
            ttk.Entry(f_row, textvariable=self.f4_ip_dest, width=16).pack(side=tk.LEFT, padx=(4, 10))
            tk.Label(f_row, text="Puerto:", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_DARK).pack(side=tk.LEFT)
            ttk.Entry(f_row, textvariable=self.f4_puerto, width=6).pack(side=tk.LEFT, padx=(4, 0))
            self.btn_iniciar_tx.config(text="▶ Enviar por UDP")
        else:
            tk.Label(self.f4_net_info,
                     text=f"Tu IP local: {self.ip_local}  |  PUERTO: {self.f4_puerto.get()}",
                     font=("Segoe UI", 8, "bold"), fg=COLOR_REC, bg=BG_DARK).pack(anchor="w")
            tk.Label(self.f4_net_info,
                     text="Dile a tu compañera que escriba esta IP en su modo Emisor.",
                     font=("Segoe UI", 7), fg=TEXT_MUTED, bg=BG_DARK).pack(anchor="w")
            self.btn_iniciar_tx.config(text="● Escuchar en red")

    def _log_f4(self, texto):
        self.feed_terminal.config(state=tk.NORMAL)
        self.feed_terminal.insert(tk.END, texto + "\n")
        self.feed_terminal.see(tk.END)
        self.feed_terminal.config(state=tk.DISABLED)

    def _error_f4(self, err_msg):
        self.transmision_en_curso = False
        self.btn_iniciar_tx.config(state=tk.NORMAL)
        self.btn_detener_tx.config(state=tk.DISABLED)
        self._log_f4(f"[ERROR] {err_msg}")

    def _finalizar_f4_tx(self, total_pkts, ip_dest, puerto):
        self.transmision_en_curso = False
        self.btn_iniciar_tx.config(state=tk.NORMAL)
        self.btn_detener_tx.config(state=tk.DISABLED)
        if self.audio_enmascarado_red is not None:
            self.btn_play_mask.config(state=tk.NORMAL)
        self._log_f4(f"\n[OK] Emisor ¡{total_pkts} datagramas enviados a {ip_dest}:{puerto}!")

    # ────────────────────────────────────────────────────────────
    # CONTROL DE TRANSMISIÓN LIVE (HILOS Y SOCKETS)
    # ────────────────────────────────────────────────────────────
    def _iniciar_transmision_live(self):
        if self.transmision_en_curso:
            return

        self.transmision_en_curso = True
        self.detener_transmision_flag = False

        rol = self.f4_modo_red.get()
        ip_dest = self.f4_ip_dest.get().strip()
        puerto = int(self.f4_puerto.get())
        usar_mascara = bool(self.var_mascara.get())
        modo_rt = bool(self.var_modo_rt.get())

        # Preparar botones y terminal
        self.btn_iniciar_tx.config(state=tk.DISABLED)
        self.btn_detener_tx.config(state=tk.NORMAL)
        self.btn_play_rec.config(state=tk.DISABLED)

        self.feed_terminal.config(state=tk.NORMAL)
        self.feed_terminal.delete("1.0", tk.END)
        if rol == "LOCAL":
            self.feed_terminal.insert(tk.END, f"[TX-LOCAL] Iniciando transmisión UDP en 127.0.0.1:{puerto}...\n")
        elif rol == "EMISOR":
            self.feed_terminal.insert(tk.END, f"[EMISOR] Enviando datagramas UDP hacia {ip_dest}:{puerto}...\n")
        else:
            self.feed_terminal.insert(tk.END, f"[RECEPTOR] Escuchando en 0.0.0.0:{puerto} (Tu IP: {self.ip_local})\n")
            self.feed_terminal.insert(tk.END, f"           Esperando paquetes de tu compañera...\n")
        self.feed_terminal.config(state=tk.DISABLED)

        # Resetear métricas e historial
        self.hist_seqs.clear()
        self.hist_latencia.clear()
        self.hist_jitter.clear()
        self.prog_tx["value"] = 0

        bloques_tx = []
        mask_flag = MASK_ON if usar_mascara else MASK_OFF

        if rol in ("LOCAL", "EMISOR"):
            bloques_raw = trocear_señal(self.fm_mod, SAMPLES_PER_BLOCK)
            if usar_mascara:
                bloques_tx = [aplicar_mascara(b, self.mascara, n_seq=i, bloque=SAMPLES_PER_BLOCK)
                              for i, b in enumerate(bloques_raw)]
                self.audio_enmascarado_red = np.concatenate(bloques_tx)
            else:
                bloques_tx = bloques_raw
                self.audio_enmascarado_red = None

        # Resetear gráficas
        self._plot_red_inicial()

        cfg = {
            "rol": rol,
            "ip_dest": ip_dest,
            "puerto": puerto,
            "bloques": bloques_tx,
            "mask_flag": mask_flag,
            "modo_rt": modo_rt,
        }
        threading.Thread(target=self._hilo_transmision_live, args=(cfg,), daemon=True).start()

    def _detener_transmision_live(self):
        if self.transmision_en_curso:
            self.detener_transmision_flag = True
            self._log_f4("[SDR-NET] Deteniendo...")

    def _hilo_transmision_live(self, cfg):
        sock_tx = None
        sock_rx = None
        try:
            rol = cfg["rol"]
            puerto = cfg["puerto"]
            bloques = cfg["bloques"]
            mask_flag = cfg["mask_flag"]
            modo_rt = cfg["modo_rt"]

            paquetes_rx = {}
            metricas = MetricasRed()
            delay = (BLOCK_MS / 1000.0) if modo_rt else 0.0015

            if rol in ("LOCAL", "RECEPTOR"):
                sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
                bind_ip = "127.0.0.1" if rol == "LOCAL" else "0.0.0.0"
                try:
                    sock_rx.bind((bind_ip, puerto))
                except Exception:
                    puerto += 1
                    sock_rx.bind((bind_ip, puerto))
                sock_rx.settimeout(0.25)

            if rol in ("LOCAL", "EMISOR"):
                sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                ip_dest = "127.0.0.1" if rol == "LOCAL" else cfg["ip_dest"]
                total_bloques = len(bloques)

                for i, bloque in enumerate(bloques):
                    if self.detener_transmision_flag:
                        break

                    t_tx = time.time()
                    datagrama = empaquetar(bloque, n_seq=i, mod=MOD_FM, fc=int(FC), mask=mask_flag)
                    sock_tx.sendto(datagrama, (ip_dest, puerto))

                    if rol == "LOCAL":
                        try:
                            data, addr = sock_rx.recvfrom(4096)
                            t_rx = time.time()
                            pkt = desempaquetar(data)
                            metricas.registrar_paquete(pkt, len(data))

                            muestras = pkt['muestras']
                            if pkt['mask'] == MASK_ON and self.mascara is not None:
                                muestras = retirar_mascara(muestras, self.mascara, n_seq=pkt['seq'], bloque=SAMPLES_PER_BLOCK)

                            paquetes_rx[pkt['seq']] = muestras
                            retardo_ms = max(0.05, (t_rx - pkt['timestamp']) * 1000)

                            actualizar_grafica = (i % 12 == 0) or (i == total_bloques - 1)
                            log_line = (i == 0) or (i % 25 == 24) or (i == total_bloques - 1)
                            self.cola_eventos.put(("PAQUETE", (
                                i + 1, total_bloques, pkt['seq'], retardo_ms,
                                metricas.jitter, len(data), pkt['mask'] == MASK_ON,
                                muestras, actualizar_grafica, log_line
                            )))
                        except socket.timeout:
                            pass
                    else:
                        # Modo EMISOR hacia otro PC
                        actualizar_grafica = (i % 12 == 0) or (i == total_bloques - 1)
                        log_line = (i == 0) or (i % 25 == 24) or (i == total_bloques - 1)
                        self.cola_eventos.put(("PAQUETE", (
                            i + 1, total_bloques, i, 0.0,
                            0.0, len(datagrama), mask_flag == MASK_ON,
                            bloque, actualizar_grafica, log_line
                        )))

                    t_transcurrido = time.time() - t_tx
                    t_espera = max(0, delay - t_transcurrido)
                    if t_espera > 0:
                        time.sleep(t_espera)

                if rol == "EMISOR" and not self.detener_transmision_flag:
                    eof = empaquetar(np.zeros(4, dtype=np.float32), n_seq=0xFFFFFFFF, mod=MOD_FM, fc=int(FC), mask=mask_flag)
                    for _ in range(3):
                        sock_tx.sendto(eof, (ip_dest, puerto))
                        time.sleep(0.02)
                    self.cola_eventos.put(("F4_TX_FIN", (total_bloques, ip_dest, puerto)))
                    return

            elif rol == "RECEPTOR":
                ultimo_t = None
                total_est = 641
                rx_raw_list = []

                while not self.detener_transmision_flag:
                    try:
                        data, addr = sock_rx.recvfrom(4096)
                        pkt = desempaquetar(data)
                        if pkt['seq'] == 0xFFFFFFFF:
                            break
                        t_rx = time.time()
                        if ultimo_t is None:
                            self.cola_eventos.put(("F4_LOG", (f"[RX] ¡Recibiendo flujo desde {addr[0]}:{addr[1]}!",)))
                        ultimo_t = t_rx

                        metricas.registrar_paquete(pkt, len(data))
                        rx_raw_list.append(pkt['muestras'])

                        muestras = pkt['muestras']
                        if pkt['mask'] == MASK_ON and self.mascara is not None:
                            muestras = retirar_mascara(muestras, self.mascara, n_seq=pkt['seq'], bloque=SAMPLES_PER_BLOCK)

                        paquetes_rx[pkt['seq']] = muestras
                        retardo_ms = max(0.05, abs(t_rx - pkt['timestamp']) * 1000)
                        if retardo_ms > 5000:
                            retardo_ms = 1.5

                        n_rx = len(paquetes_rx)
                        actualizar_grafica = (n_rx % 12 == 0)
                        log_line = (n_rx == 1) or (n_rx % 25 == 0)
                        self.cola_eventos.put(("PAQUETE", (
                            n_rx, max(total_est, n_rx), pkt['seq'], retardo_ms,
                            metricas.jitter, len(data), pkt['mask'] == MASK_ON,
                            muestras, actualizar_grafica, log_line
                        )))
                    except socket.timeout:
                        if ultimo_t is not None and (time.time() - ultimo_t) > 1.8:
                            break
                        continue
                    except ValueError:
                        continue

                if rx_raw_list:
                    self.audio_enmascarado_red = np.concatenate(rx_raw_list)

            if self.detener_transmision_flag:
                self.cola_eventos.put(("CANCEL", ()))
                return

            if paquetes_rx:
                seqs_ordenadas = sorted(paquetes_rx.keys())
                señal_fm_reensamblada = np.concatenate([paquetes_rx[s] for s in seqs_ordenadas])
                audio_recuperado = demodular_fm(señal_fm_reensamblada)
                resumen = metricas.resumen()
                self.cola_eventos.put(("FIN", (audio_recuperado, resumen)))
            else:
                self.cola_eventos.put(("CANCEL", ()))

        except Exception as e:
            self.cola_eventos.put(("F4_ERROR", (f"{type(e).__name__}: {e}",)))
        finally:
            if sock_tx is not None:
                try:
                    sock_tx.close()
                except Exception:
                    pass
            if sock_rx is not None:
                try:
                    sock_rx.close()
                except Exception:
                    pass

    def _notificar_paquete_live(self, num_pkt, total_pkts, pkt_seq, retardo_ms,
                                jitter_ms, tamaño_bytes, con_mascara, muestras_bloque,
                                actualizar_grafica, log_line=True):
        pct = min(100.0, (num_pkt / max(1, total_pkts)) * 100)
        self.prog_tx["value"] = pct

        if log_line:
            estado_mask = "MASK" if con_mascara else "RAW "
            msg = f"[UDP] #{pkt_seq:03d} ({tamaño_bytes}B) | {retardo_ms:.2f}ms | Jit:{jitter_ms:.2f}ms | {estado_mask}\n"
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
            if len(muestras_bloque) == SAMPLES_PER_BLOCK:
                color_scope = "#4F46E5" if con_mascara else COLOR_REC
                self.line_scope.set_ydata(muestras_bloque)
                self.line_scope.set_color(color_scope)
                tipo_txt = "Con máscara musical" if con_mascara else "FM directa"
                self.ax_scope.set_title(f"Datagrama #{pkt_seq:03d} en Vivo ({len(muestras_bloque)} muestras) — {tipo_txt}",
                                        color=TEXT_WHITE, fontsize=9, fontweight="bold")

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
        self.btn_play_mask.config(state=tk.NORMAL)

        # Mensaje final en terminal
        self.feed_terminal.config(state=tk.NORMAL)
        self.feed_terminal.insert(tk.END, f"\n[OK] Transmisión completada ({resumen['paquetes_recibidos']} paquetes recibidos)\n")
        self.feed_terminal.insert(tk.END, f"   Pérdidas: {resumen['perdida_pct']:.1f}% | Retardo Medio: {resumen['retardo_medio_ms']:.2f} ms | Jitter: {resumen['jitter_ms']:.2f} ms\n")
        self.feed_terminal.see(tk.END)
        self.feed_terminal.config(state=tk.DISABLED)

        # Mostrar gráfica final en ventana con voz activa
        self.ax_scope.cla()
        self.ax_scope.set_facecolor(BG_INNER)
        for spine in self.ax_scope.spines.values():
            spine.set_color("#E2E8F0")

        i0, i1 = self._get_f5_voiced_slice()
        i1 = min(i1, len(audio_recuperado))
        t_ms = (np.arange(i0, i1) / FS) * 1000

        self.ax_scope.plot(t_ms, self.voz[i0:i1], color=COLOR_VOZ, alpha=0.45, lw=1.5, label="Voz Original")
        self.ax_scope.plot(t_ms, audio_recuperado[i0:i1], color=COLOR_REC, lw=1.1, label="Voz Reconstruida tras Red UDP")
        self.ax_scope.set_title("Transmision Completada: Voz Reconstruida tras Sockets UDP (0.0% Perdida)",
                                color="#059669", fontsize=10, fontweight="bold")
        self.ax_scope.set_xlabel("Tiempo (ms)", color=TEXT_MUTED, fontsize=8)
        self.ax_scope.set_ylabel("Amplitud", color=TEXT_MUTED, fontsize=8)
        self.ax_scope.set_ylim(-1.2, 1.2)
        self.ax_scope.legend(loc="upper right", facecolor=BG_CARD, edgecolor=BORDER_CARD, labelcolor=TEXT_WHITE, fontsize=7)
        self.ax_scope.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)

        self.canvas_red.draw()

    def _cancelar_transmision_live(self):
        self.transmision_en_curso = False
        self.btn_iniciar_tx.config(state=tk.NORMAL)
        self.btn_detener_tx.config(state=tk.DISABLED)

        self.feed_terminal.config(state=tk.NORMAL)
        self.feed_terminal.insert(tk.END, "\n[Detenido] Transmisión detenida por el usuario.\n")
        self.feed_terminal.see(tk.END)
        self.feed_terminal.config(state=tk.DISABLED)

    def _reproducir_audio_enmascarado(self):
        if self.audio_enmascarado_red is not None:
            sig = self.audio_enmascarado_red
        elif self.mascara is not None:
            sig = self.mascara[:len(self.voz)]
        else:
            return
        self._log_f4("[ESPÍA] Interceptando paquetes SIN clave: ¡Solo se escucha música 8-bits! La voz está oculta.")
        self.reproducir_audio(sig)

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
            self._estilizar_eje(ax)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#E2E8F0")

        # 1. Gráfica Temporal
        n_muestras_plot = int(0.010 * FS)
        t_ms = np.arange(n_muestras_plot) / FS * 1000
        ax1.plot(t_ms, señal[:n_muestras_plot], color=color, lw=1.2)
        self._estilizar_eje(ax1)
        self._estilizar_eje(ax2)
        ax1.set_title(f"{titulo} (10 ms)", fontsize=9.5, fontweight="bold", color=TEXT_WHITE, loc="left", pad=8)
        ax1.set_xlabel("Tiempo (ms)", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylabel("Amplitud [-1, 1]", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylim(-1.15, 1.15)
        ax1.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)

        # 2. Gráfica Espectro FFT
        N = len(señal)
        mags = 20 * np.log10(np.abs(np.fft.rfft(señal)) / N + 1e-12)
        freqs = np.fft.rfftfreq(N, 1 / FS) / 1000
        ax2.plot(freqs, mags, color=color, lw=0.9)
        ax2.axvline(4.0, color="#64748b", linestyle=":", label="4 kHz (límite voz)")
        ax2.axvline(FC / 1000, color="#DC2626", linestyle="--", label=f"Portadora fc={FC/1000:.0f} kHz")
        ax2.set_title("Espectro de frecuencias (FFT)", fontsize=9.5, fontweight="bold", color=TEXT_WHITE, loc="left", pad=8)
        ax2.set_xlabel("Frecuencia (kHz)", color=TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("Magnitud (dB)", color=TEXT_MUTED, fontsize=8)
        ax2.set_xlim(0, FS / 2 / 1000)
        ax2.legend(loc="lower right", facecolor=BG_CARD, edgecolor=BORDER_CARD, labelcolor=TEXT_WHITE, fontsize=7)
        ax2.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)

        fig.tight_layout(pad=2.0)
        canvas.draw()

    def _plot_ruido_inicial(self):
        self.fig_ruido.clf()
        ax = self.fig_ruido.add_subplot(1, 1, 1)
        self._estilizar_eje(ax)
        for spine in ax.spines.values():
            spine.set_color("#E2E8F0")
        ax.text(0.5, 0.5, "Pulsa en 'Escuchar AM con Ruido' o 'Escuchar FM con Ruido'\npara ver la comparación gráfica y auditiva en tiempo real.",
                ha="center", va="center", fontsize=11, color=TEXT_MUTED)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        self.canvas_ruido.draw()

    def _plot_comparativa_ruido_grafica(self, sig_recup, titulo, color):
        self.fig_ruido.clf()
        ax1 = self.fig_ruido.add_subplot(2, 1, 1)
        ax2 = self.fig_ruido.add_subplot(2, 1, 2)

        for ax in (ax1, ax2):
            self._estilizar_eje(ax)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#E2E8F0")

        n_p = min(len(self.voz), int(0.040 * FS))
        t_ms = np.arange(n_p) / FS * 1000

        ax1.plot(t_ms, self.voz[:n_p], color=COLOR_VOZ, alpha=0.45, lw=1.5, label="Voz Original Limpia")
        ax1.plot(t_ms, sig_recup[:n_p], color=color, lw=1.1, label=titulo)
        ax1.set_title(f"{titulo} — Señal en el Tiempo", fontsize=10, fontweight="bold", color=TEXT_WHITE)
        ax1.set_xlabel("Tiempo (ms)", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylabel("Amplitud", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylim(-1.2, 1.2)
        ax1.legend(loc="upper right", facecolor=BG_CARD, edgecolor=BORDER_CARD, labelcolor=TEXT_WHITE, fontsize=7)
        ax1.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)

        N = len(sig_recup)
        mags = 20 * np.log10(np.abs(np.fft.rfft(sig_recup)) / N + 1e-12)
        freqs = np.fft.rfftfreq(N, 1 / FS) / 1000
        ax2.plot(freqs, mags, color=color, lw=0.9)
        ax2.set_title("Espectro de Frecuencias (FFT)", fontsize=10, fontweight="bold", color=TEXT_WHITE)
        ax2.set_xlabel("Frecuencia (kHz)", color=TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("Magnitud (dB)", color=TEXT_MUTED, fontsize=8)
        ax2.set_xlim(0, 8)
        ax2.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)

        self.fig_ruido.tight_layout(pad=2.0)
        self.canvas_ruido.draw()

    # ────────────────────────────────────────────────────────────
    # PESTAÑA 5: INTEGRACIÓN AM/FM SOBRE UDP (FASE 5)
    # ────────────────────────────────────────────────────────────
    def _get_f5_voiced_slice(self):
        """Devuelve (inicio, fin) de una ventana de 40 ms donde realmente haya voz activa."""
        n_win = min(len(self.voz), int(0.040 * FS))
        if len(self.voz) <= n_win:
            return 0, len(self.voz)
        peak_idx = int(np.argmax(np.abs(self.voz)))
        start = max(0, min(peak_idx - n_win // 3, len(self.voz) - n_win))
        return start, start + n_win

    def _crear_tab_fase5(self):
        paned = tk.PanedWindow(self.tab_fase5, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)

        izq = self._crear_panel_izq_scroll(paned, width=445)

        tk.Label(izq, text="Fase 5: Radio AM/FM en Vivo por Red UDP", font=("Segoe UI", 12, "bold"),
                 fg=TEXT_WHITE, bg=BG_DARK).pack(anchor="w")
        tk.Label(
            izq,
            text="Qué enseña esta pestaña: tu voz se modula (AM o FM) bloque a bloque, viaja por la red en paquetes UDP y el receptor la reconstruye al instante.",
            font=("Segoe UI", 8), fg=TEXT_SEC, bg=BG_CARD, wraplength=400, justify="left"
        ).pack(anchor="w", pady=(2, 6))

        # Tarjeta 0: Modo de Red (Local vs PC a PC)
        card_red = self._crear_tarjeta(izq, " Paso 1 · Conexión de red ", acento="#4F46E5", pady=4)

        f_roles = tk.Frame(card_red, bg=BG_CARD)
        f_roles.pack(fill=tk.X)

        for txt, val, col in [
            ("Local (1 PC)", "LOCAL", TEXT_WHITE),
            ("Emisor", "EMISOR", "#475569"),
            ("Receptor", "RECEPTOR", COLOR_REC),
        ]:
            tk.Radiobutton(
                f_roles, text=txt, variable=self.f5_modo_red, value=val,
                command=self._on_f5_red_change, font=("Segoe UI", 8, "bold"),
                fg=col, bg=BG_CARD, selectcolor=BG_CARD, activebackground=BG_CARD, activeforeground=TEXT_WHITE
            ).pack(side=tk.LEFT, padx=(0, 6))

        self.f5_net_info = tk.Frame(card_red, bg=BG_DARK, padx=8, pady=5)
        self.f5_net_info.pack(fill=tk.X, pady=(4, 0))

        # Tarjeta 1: Controles del Modulador SDR
        card_ctrl = self._crear_tarjeta(izq, " Paso 2 · Modulación y envío ", acento=COLOR_AM, pady=4)

        f_modos = tk.Frame(card_ctrl, bg=BG_CARD)
        f_modos.pack(fill=tk.X, pady=(0, 4))

        tk.Radiobutton(f_modos, text="FM (Frecuencia)", variable=self.f5_modo, value="FM",
                       command=self._on_f5_modo_change, font=("Segoe UI", 8, "bold"),
                       fg=COLOR_FM, bg=BG_CARD, selectcolor=BG_CARD, activebackground=BG_CARD, activeforeground=TEXT_WHITE).pack(side=tk.LEFT, padx=(0, 12))
        tk.Radiobutton(f_modos, text="AM (Amplitud)", variable=self.f5_modo, value="AM",
                       command=self._on_f5_modo_change, font=("Segoe UI", 8, "bold"),
                       fg=COLOR_AM, bg=BG_CARD, selectcolor=BG_CARD, activebackground=BG_CARD, activeforeground=TEXT_WHITE).pack(side=tk.LEFT)

        f_params = tk.Frame(card_ctrl, bg=BG_CARD)
        f_params.pack(fill=tk.X, pady=2)

        tk.Label(f_params, text="Portadora fc (Hz):", font=("Segoe UI", 8), fg=TEXT_SEC, bg=BG_CARD).pack(side=tk.LEFT)
        self.f5_fc_spin = ttk.Spinbox(f_params, from_=2000, to=18000, increment=500, textvariable=self.f5_fc, width=7)
        self.f5_fc_spin.pack(side=tk.LEFT, padx=(4, 10))

        self.f5_param_lbl = tk.Label(f_params, text="kf (Hz):", font=("Segoe UI", 8), fg=TEXT_SEC, bg=BG_CARD)
        self.f5_param_lbl.pack(side=tk.LEFT)
        self.f5_param_spin = ttk.Spinbox(f_params, from_=0.1, to=8000, increment=100, textvariable=self.f5_param, width=7)
        self.f5_param_spin.pack(side=tk.LEFT, padx=(4, 0))

        f_opt = tk.Frame(card_ctrl, bg=BG_CARD)
        f_opt.pack(fill=tk.X, pady=3)

        chk_perd = tk.Checkbutton(f_opt, text="Simular pérdida de paquete #25 (PLC)",
                                  variable=self.f5_sim_perdida, font=("Segoe UI", 8),
                                  fg="#475569", bg=BG_CARD, selectcolor=BG_CARD, activebackground=BG_CARD, activeforeground=TEXT_WHITE)
        chk_perd.pack(side=tk.LEFT)

        f_btns = tk.Frame(card_ctrl, bg=BG_CARD)
        f_btns.pack(fill=tk.X, pady=(4, 0))

        self.btn_f5_iniciar = ttk.Button(f_btns, text="▶ Transmitir y demodular", style="Purple.TButton",
                                         command=self._iniciar_f5)
        self.btn_f5_iniciar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_f5_detener = ttk.Button(f_btns, text="■ Detener", style="Stop.TButton",
                                         command=self._detener_f5, state=tk.DISABLED)
        self.btn_f5_detener.pack(side=tk.RIGHT)

        # Tarjeta 2: Monitor de Terminal
        card_feed = self._crear_tarjeta(izq, " Paso 3 · Registro y métricas ", acento="#4F46E5", pady=4)

        self.f5_terminal = tk.Text(card_feed, height=5, bg=BG_TERMINAL, fg="#4F46E5",
                                   font=("Consolas", 8), bd=0, padx=6, pady=4, wrap=tk.NONE)
        self.f5_terminal.pack(fill=tk.BOTH, expand=True)
        self.f5_terminal.insert(tk.END, "[SDR-F5] Elige modo de red y pulsa el botón para iniciar...\n")
        self.f5_terminal.config(state=tk.DISABLED)

        # Tarjeta 3: Telemetría y Fidelidad
        card_kpis = self._crear_tarjeta(izq, " Resumen de transmisión ", acento=COLOR_REC, pady=4)

        self.f5_prog = ttk.Progressbar(card_kpis, orient=tk.HORIZONTAL, mode="determinate")
        self.f5_prog.pack(fill=tk.X, pady=(0, 5))

        f_grid = tk.Frame(card_kpis, bg=BG_CARD)
        f_grid.pack(fill=tk.X)

        for col, (titulo, attr) in enumerate([
            ("MODO", "lbl_f5_kpi_modo"),
            ("BLOQUES", "lbl_f5_kpi_pkts"),
            ("PÉRDIDAS", "lbl_f5_kpi_perd"),
            ("JITTER", "lbl_f5_kpi_jit"),
            ("CALIDAD SNR", "lbl_f5_kpi_snr")
        ]):
            box = tk.Frame(f_grid, bg=BG_INNER, padx=4, pady=4)
            box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            tk.Label(box, text=titulo, font=("Segoe UI", 6, "bold"), fg=TEXT_MUTED, bg=BG_INNER).pack()
            lbl = tk.Label(box, text="--", font=("Segoe UI", 8, "bold"), fg=TEXT_WHITE, bg=BG_INNER)
            lbl.pack()
            setattr(self, attr, lbl)

        # Tarjeta 4: Escucha y Verificación
        card_audio = self._crear_tarjeta(izq, " Paso 4 · Escuchar resultado ", acento="#4F46E5", pady=4)

        f_play_btns = tk.Frame(card_audio, bg=BG_CARD)
        f_play_btns.pack(fill=tk.X)

        self.btn_f5_play_orig = ttk.Button(f_play_btns, text="▶ 1. Original", style="Blue.TButton",
                                           command=self._reproducir_f5_orig)
        self.btn_f5_play_orig.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_f5_play_mod = ttk.Button(f_play_btns, text="▶ 2. En red", style="Pink.TButton",
                                          command=self._reproducir_f5_mod, state=tk.DISABLED)
        self.btn_f5_play_mod.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_f5_play_rec = ttk.Button(f_play_btns, text="▶ 3. Recuperada", style="Green.TButton",
                                          command=self._reproducir_f5_rec, state=tk.DISABLED)
        self.btn_f5_play_rec.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self._on_f5_red_change()

        # Panel Derecho: Canvas Matplotlib (3 subplots)
        der = tk.Frame(paned, bg=BG_DARK, padx=10, pady=10)
        paned.add(der, minsize=500)

        self.fig_f5 = Figure(figsize=(7, 5.5), dpi=120, facecolor=BG_CARD)
        self.canvas_f5 = FigureCanvasTkAgg(self.fig_f5, master=der)
        self.canvas_f5.get_tk_widget().configure(highlightthickness=0, bd=0)
        self.canvas_f5.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._configurar_toolbar(self.canvas_f5, der)

        self._plot_f5_inicial()

    def _on_f5_red_change(self):
        for w in self.f5_net_info.winfo_children():
            w.destroy()
        rol = self.f5_modo_red.get()
        if rol == "LOCAL":
            tk.Label(self.f5_net_info, text="Bucle local (127.0.0.1:5018) para pruebas en un equipo.",
                     font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_DARK).pack(anchor="w")
            self.btn_f5_iniciar.config(text="▶ Transmitir y demodular")
        elif rol == "EMISOR":
            f_row = tk.Frame(self.f5_net_info, bg=BG_DARK)
            f_row.pack(fill=tk.X)
            tk.Label(f_row, text="IP Destino:", font=("Segoe UI", 8, "bold"), fg="#475569", bg=BG_DARK).pack(side=tk.LEFT)
            ttk.Entry(f_row, textvariable=self.f5_ip_dest, width=16).pack(side=tk.LEFT, padx=(4, 10))
            tk.Label(f_row, text="Puerto:", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=BG_DARK).pack(side=tk.LEFT)
            ttk.Entry(f_row, textvariable=self.f5_puerto, width=6).pack(side=tk.LEFT, padx=(4, 0))
            self.btn_f5_iniciar.config(text="▶ Enviar audio a IP")
        else:
            tk.Label(self.f5_net_info,
                     text=f"Tu IP local: {self.ip_local}  |  PUERTO: {self.f5_puerto.get()}",
                     font=("Segoe UI", 8, "bold"), fg=COLOR_REC, bg=BG_DARK).pack(anchor="w")
            tk.Label(self.f5_net_info,
                     text="Dile a tu compañera que escriba esta IP en su modo Emisor.",
                     font=("Segoe UI", 7), fg=TEXT_MUTED, bg=BG_DARK).pack(anchor="w")
            self.btn_f5_iniciar.config(text="● Escuchar en red")

    def _plot_f5_inicial(self):
        self.fig_f5.clf()
        self.ax_f5_tiempo = self.fig_f5.add_subplot(3, 1, 1)
        self.ax_f5_espectro_mod = self.fig_f5.add_subplot(3, 1, 2)
        self.ax_f5_espectro_rec = self.fig_f5.add_subplot(3, 1, 3)

        for ax in (self.ax_f5_tiempo, self.ax_f5_espectro_mod, self.ax_f5_espectro_rec):
            self._estilizar_eje(ax)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#E2E8F0")

        modo = self.f5_modo.get().upper()
        fc = self.f5_fc.get()

        # Subplot 1: Ventana con voz activa (no silencio inicial)
        i0, i1 = self._get_f5_voiced_slice()
        t_ms = (np.arange(i0, i1) / FS) * 1000
        self.ax_f5_tiempo.plot(t_ms, self.voz[i0:i1], color=COLOR_VOZ, alpha=0.85, lw=1.2, label="Voz Original")
        self.ax_f5_tiempo.set_title("1. Forma de onda: Original vs Recuperada",
                                    fontsize=9, fontweight="bold", color=TEXT_WHITE)
        self.ax_f5_tiempo.set_xlabel("Tiempo (ms)", color=TEXT_MUTED, fontsize=8)
        self.ax_f5_tiempo.set_ylabel("Amplitud", color=TEXT_MUTED, fontsize=8)
        self.ax_f5_tiempo.set_ylim(-1.15, 1.15)
        self.ax_f5_tiempo.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)
        self.ax_f5_tiempo.legend(loc="upper right", facecolor=BG_CARD, edgecolor=BORDER_CARD, labelcolor=TEXT_WHITE, fontsize=7)

        # Subplot 2: Espectro modulado
        self.ax_f5_espectro_mod.set_title(f"2. Espectro {modo} en red (fc = {fc/1000:.1f} kHz)",
                                          fontsize=9, fontweight="bold", color=TEXT_WHITE)
        self.ax_f5_espectro_mod.set_xlabel("Frecuencia (kHz)", color=TEXT_MUTED, fontsize=8)
        self.ax_f5_espectro_mod.set_ylabel("Magnitud (dB)", color=TEXT_MUTED, fontsize=8)
        self.ax_f5_espectro_mod.set_xlim(0, 24)
        self.ax_f5_espectro_mod.set_ylim(-100, 0)
        self.ax_f5_espectro_mod.text(12, -50, "Pulsa el botón para transmitir y ver el espectro modulado en la red",
                                     color=TEXT_MUTED, fontsize=8, ha="center", va="center")
        self.ax_f5_espectro_mod.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)

        # Subplot 3: Espectro recuperado
        self.ax_f5_espectro_rec.set_title("3. Espectro de voz recuperada (0 - 8 kHz)",
                                          fontsize=9, fontweight="bold", color=TEXT_WHITE)
        self.ax_f5_espectro_rec.set_xlabel("Frecuencia (kHz)", color=TEXT_MUTED, fontsize=8)
        self.ax_f5_espectro_rec.set_ylabel("Magnitud (dB)", color=TEXT_MUTED, fontsize=8)
        self.ax_f5_espectro_rec.set_xlim(0, 8)
        self.ax_f5_espectro_rec.set_ylim(-100, 0)
        self.ax_f5_espectro_rec.text(4, -50, "Aquí aparecerá la voz limpia extraída tras demodular",
                                     color=TEXT_MUTED, fontsize=8, ha="center", va="center")
        self.ax_f5_espectro_rec.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)

        self.fig_f5.tight_layout(pad=2.0)
        self.canvas_f5.draw()

    def _on_f5_modo_change(self):
        modo = self.f5_modo.get().upper()
        if modo == "AM":
            self.f5_fc.set(FC_AM_DEFAULT)
            self.f5_param.set(KA_DEFAULT)
            self.f5_param_lbl.config(text="ka (0-1):")
            self.f5_param_spin.config(from_=0.1, to=1.0, increment=0.05)
        else:
            self.f5_fc.set(FC_FM_DEFAULT)
            self.f5_param.set(KF_DEFAULT)
            self.f5_param_lbl.config(text="kf (Hz):")
            self.f5_param_spin.config(from_=100, to=8000, increment=200)
        self._plot_f5_inicial()

    def _iniciar_f5(self):
        if self.f5_en_curso:
            return
        self.f5_en_curso = True
        self.f5_detener_flag = False

        self.btn_f5_iniciar.config(state=tk.DISABLED)
        self.btn_f5_detener.config(state=tk.NORMAL)
        self.btn_f5_play_mod.config(state=tk.DISABLED)
        self.btn_f5_play_rec.config(state=tk.DISABLED)

        self.f5_terminal.config(state=tk.NORMAL)
        self.f5_terminal.delete("1.0", tk.END)
        rol = self.f5_modo_red.get()
        modo = self.f5_modo.get().upper()
        fc = self.f5_fc.get()
        param = self.f5_param.get()
        p_name = "ka" if modo == "AM" else "kf"

        if rol == "LOCAL":
            self.f5_terminal.insert(tk.END, f"[F5-LOCAL] Iniciando bucle {modo} | fc={fc:.0f} Hz | {p_name}={param}\n")
        elif rol == "EMISOR":
            ip_d = self.f5_ip_dest.get().strip()
            puerto = int(self.f5_puerto.get())
            self.f5_terminal.insert(tk.END, f"[F5-EMISOR] Enviando {modo} hacia {ip_d}:{puerto} | fc={fc:.0f} Hz\n")
        else:
            puerto = int(self.f5_puerto.get())
            self.f5_terminal.insert(tk.END, f"[F5-RECEPTOR] Escuchando en 0.0.0.0:{puerto} (Tu IP: {self.ip_local})\n")
            self.f5_terminal.insert(tk.END, f"              Esperando paquetes de tu compañera... (Pulsa Detener para cancelar)\n")
        self.f5_terminal.config(state=tk.DISABLED)

        self.f5_hist_seqs.clear()
        self.f5_hist_latencia.clear()
        self.f5_hist_jitter.clear()
        self.f5_prog["value"] = 0

        self._plot_f5_inicial()

        cfg = {
            "rol": rol,
            "modo": modo,
            "fc": float(fc),
            "param": float(param),
            "sim_perdida": bool(self.f5_sim_perdida.get()),
            "tam_bloque": int(self.f5_tam_bloque.get()),
            "puerto": int(self.f5_puerto.get()),
            "ip_dest": self.f5_ip_dest.get().strip(),
        }
        threading.Thread(target=self._hilo_f5, args=(cfg,), daemon=True).start()

    def _detener_f5(self):
        if self.f5_en_curso:
            self.f5_detener_flag = True
            self._log_f5("[F5] Deteniendo...")

    def _log_f5(self, texto):
        self.f5_terminal.config(state=tk.NORMAL)
        self.f5_terminal.insert(tk.END, texto + "\n")
        self.f5_terminal.see(tk.END)
        self.f5_terminal.config(state=tk.DISABLED)

    def _error_f5(self, err_msg):
        self.f5_en_curso = False
        self.btn_f5_iniciar.config(state=tk.NORMAL)
        self.btn_f5_detener.config(state=tk.DISABLED)
        self._log_f5(f"[ERROR] {err_msg}")

    def _finalizar_f5_tx(self, total_pkts, ip_dest, puerto):
        self.f5_en_curso = False
        self.btn_f5_iniciar.config(state=tk.NORMAL)
        self.btn_f5_detener.config(state=tk.DISABLED)
        self.btn_f5_play_mod.config(state=tk.NORMAL)
        self._log_f5(f"[F5-EMISOR] ¡Envío completado! {total_pkts} bloques enviados a {ip_dest}:{puerto}.")

    def _hilo_f5(self, cfg):
        HDR_FMT = "!4sIIdBffH"
        HDR_SIZE = struct.calcsize(HDR_FMT)  # 31 bytes

        sock_tx = None
        sock_rx = None

        try:
            rol = cfg["rol"]
            modo = cfg["modo"]
            fc = cfg["fc"]
            param = cfg["param"]
            sim_perdida = cfg["sim_perdida"]
            tam_bloque = cfg["tam_bloque"]
            puerto = cfg["puerto"]

            bloques = [self.voz[i:i + tam_bloque] for i in range(0, len(self.voz), tam_bloque)]
            total_bloques = len(bloques)

            estado_fm = EstadoFM()
            offset_am_tx = 0
            offset_am_rx = 0

            paquetes_modulados = []
            bloques_recibidos = {}
            jitter = 0.0
            prev_rx = 0.0
            prev_tx = 0.0

            if rol in ("LOCAL", "RECEPTOR"):
                sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
                bind_ip = "127.0.0.1" if rol == "LOCAL" else "0.0.0.0"
                try:
                    sock_rx.bind((bind_ip, puerto))
                except Exception:
                    puerto = puerto + 1
                    sock_rx.bind((bind_ip, puerto))
                sock_rx.settimeout(0.25)

            if rol in ("LOCAL", "EMISOR"):
                sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                ip_dest = "127.0.0.1" if rol == "LOCAL" else cfg["ip_dest"]

                for idx, bloque in enumerate(bloques):
                    if self.f5_detener_flag:
                        break

                    seq = idx + 1
                    t_tx = time.time()

                    if modo == "AM":
                        bloque_mod, offset_am_tx = modular_am_bloque(
                            bloque, fc=fc, ka=param, fs=FS, sample_offset=offset_am_tx
                        )
                    else:
                        bloque_mod, estado_fm = modular_fm_bloque(
                            bloque, fc=fc, kf=param, fs=FS, estado=estado_fm
                        )

                    paquetes_modulados.append(bloque_mod)
                    payload = serializar_bloque(bloque_mod)

                    # Prueba de descarte voluntario
                    if sim_perdida and seq == 25:
                        time.sleep(0.002)
                        continue

                    m_code = 1 if modo == "FM" else 0
                    datagrama = struct.pack(
                        HDR_FMT, b"SDR5", seq, total_bloques, t_tx, m_code, fc, param, len(payload)
                    ) + payload
                    sock_tx.sendto(datagrama, (ip_dest, puerto))

                    if rol == "LOCAL":
                        try:
                            data, addr = sock_rx.recvfrom(4096)
                            t_rx = time.time()
                            mag, s_num, t_tot, ts, mc, f_c, p_val, pay_len = struct.unpack(HDR_FMT, data[:HDR_SIZE])
                            m_rx = deserializar_bloque(data[HDR_SIZE:HDR_SIZE + pay_len])

                            retardo_ms = max(0.02, (t_rx - ts) * 1000.0)
                            if prev_rx > 0 and ts > 0:
                                d = abs((t_rx - prev_rx) - (ts - prev_tx))
                                jitter += (d - jitter) / 16.0
                            prev_rx = t_rx
                            prev_tx = ts

                            bloques_recibidos[s_num] = m_rx
                            log_line = (seq == 1) or (seq % 25 == 0) or (seq == total_bloques)
                            self.cola_eventos.put(("F5_PAQUETE", (
                                seq, total_bloques, modo, retardo_ms, jitter * 1000.0, len(data), log_line
                            )))
                        except socket.timeout:
                            pass
                        time.sleep(0.002)
                    else:
                        # Modo EMISOR hacia otro PC
                        log_line = (seq == 1) or (seq % 25 == 0) or (seq == total_bloques)
                        self.cola_eventos.put(("F5_PAQUETE", (
                            seq, total_bloques, modo, 0.0, 0.0, len(datagrama), log_line
                        )))
                        time.sleep(0.010)

                # Enviar marcador de fin en modo EMISOR
                if rol == "EMISOR" and not self.f5_detener_flag:
                    eof_pkt = struct.pack(HDR_FMT, b"SDR5", 0, total_bloques, time.time(), 99, fc, param, 0)
                    for _ in range(3):
                        sock_tx.sendto(eof_pkt, (ip_dest, puerto))
                        time.sleep(0.02)
                    self.f5_audio_modulado = np.concatenate(paquetes_modulados) if paquetes_modulados else None
                    self.cola_eventos.put(("F5_TX_FIN", (total_bloques, ip_dest, puerto)))
                    return

            elif rol == "RECEPTOR":
                ultimo_paquete_t = None
                total_esp_rx = total_bloques
                modo_rx = modo
                fc_rx = fc
                param_rx = param

                while not self.f5_detener_flag:
                    try:
                        data, addr = sock_rx.recvfrom(4096)
                        if len(data) < HDR_SIZE:
                            continue
                        mag, s_num, t_tot, ts, mc, f_c, p_val, pay_len = struct.unpack(HDR_FMT, data[:HDR_SIZE])
                        if mag != b"SDR5":
                            continue
                        if mc == 99:
                            # Fin de transmisión
                            break

                        t_rx = time.time()
                        m_rx = deserializar_bloque(data[HDR_SIZE:HDR_SIZE + pay_len])
                        modo_rx = "FM" if mc == 1 else "AM"
                        fc_rx = f_c
                        param_rx = p_val
                        total_esp_rx = t_tot

                        if ultimo_paquete_t is None:
                            self.cola_eventos.put(("F5_LOG", (f"[RX] ¡Recibiendo flujo {modo_rx} desde {addr[0]}:{addr[1]}!",)))
                        ultimo_paquete_t = t_rx

                        retardo_ms = max(0.02, abs(t_rx - ts) * 1000.0)
                        if retardo_ms > 5000:
                            retardo_ms = 1.5  # relojes no sincronizados entre PCs
                        if prev_rx > 0 and ts > 0:
                            d = abs((t_rx - prev_rx) - (ts - prev_tx))
                            jitter += (d - jitter) / 16.0
                        prev_rx = t_rx
                        prev_tx = ts

                        bloques_recibidos[s_num] = m_rx
                        paquetes_modulados.append(m_rx)
                        n_rx = len(bloques_recibidos)
                        log_line = (n_rx == 1) or (n_rx % 25 == 0) or (s_num == total_esp_rx)
                        self.cola_eventos.put(("F5_PAQUETE", (
                            s_num, total_esp_rx, modo_rx, retardo_ms, jitter * 1000.0, len(data), log_line
                        )))
                    except socket.timeout:
                        if ultimo_paquete_t is not None and (time.time() - ultimo_paquete_t) > 1.8:
                            break
                        continue

                modo = modo_rx
                fc = fc_rx
                param = param_rx

            if self.f5_detener_flag:
                self.cola_eventos.put(("F5_CANCEL", ()))
                return

            if bloques_recibidos:
                seq_min = min(bloques_recibidos.keys())
                seq_max = max(bloques_recibidos.keys())
                total_esp = seq_max - seq_min + 1
                señal_mod_partes = []

                for s in range(seq_min, seq_max + 1):
                    if s in bloques_recibidos:
                        b_rx = bloques_recibidos[s]
                    else:
                        # PLC sobre la señal modulada recibida (repetición atenuada del último bloque)
                        b_rx = señal_mod_partes[-1] * 0.5 if señal_mod_partes else np.zeros(tam_bloque, dtype=np.float32)
                    señal_mod_partes.append(b_rx)

                señal_modulada_rx = np.concatenate(señal_mod_partes)
                señal_modulada = np.concatenate(paquetes_modulados)

                # Demodulación continua del flujo reensamblado (evita clics y siseos en bordes de bloque)
                if modo == "AM":
                    audio_recuperado, _ = demodular_am_bloque(
                        señal_modulada_rx, fc=fc, ka=param, fs=FS, sample_offset=0
                    )
                else:
                    audio_recuperado = demodular_fm_bloque(
                        señal_modulada_rx, fc=fc, kf=param, fs=FS
                    )

                snr_info = calcular_snr(self.voz, audio_recuperado)

                metricas = {
                    "modo": modo,
                    "esperados": total_esp,
                    "recibidos": len(bloques_recibidos),
                    "perdidos": total_esp - len(bloques_recibidos),
                    "jitter_ms": jitter * 1000.0,
                    "snr_db": snr_info["SNR_dB"]
                }
                self.cola_eventos.put(("F5_FIN", (señal_modulada, audio_recuperado, metricas)))
            else:
                self.cola_eventos.put(("F5_CANCEL", ()))

        except Exception as e:
            self.cola_eventos.put(("F5_ERROR", (f"{type(e).__name__}: {e}",)))
        finally:
            if sock_tx is not None:
                try:
                    sock_tx.close()
                except Exception:
                    pass
            if sock_rx is not None:
                try:
                    sock_rx.close()
                except Exception:
                    pass

    def _notificar_f5_paquete(self, num_pkt, total_pkts, modo, retardo_ms, jitter_ms, tam_bytes, log_line):
        pct = min(100.0, (num_pkt / max(1, total_pkts)) * 100)
        self.f5_prog["value"] = pct

        if log_line:
            msg = f"[{modo}] #{num_pkt:03d}/{total_pkts} | {tam_bytes}B | Ret:{retardo_ms:.2f}ms | Jit:{jitter_ms:.2f}ms\n"
            self.f5_terminal.config(state=tk.NORMAL)
            self.f5_terminal.insert(tk.END, msg)
            self.f5_terminal.see(tk.END)
            self.f5_terminal.config(state=tk.DISABLED)

        self.lbl_f5_kpi_modo.config(text=modo, fg=COLOR_FM if modo == "FM" else COLOR_AM)
        self.lbl_f5_kpi_pkts.config(text=f"{num_pkt}/{total_pkts} ({pct:.0f}%)")
        self.lbl_f5_kpi_perd.config(text="0.0%", fg=COLOR_REC)
        self.lbl_f5_kpi_jit.config(text=f"{jitter_ms:.2f} ms")

    def _finalizar_f5(self, señal_modulada, audio_recuperado, metricas):
        self.f5_en_curso = False
        self.btn_f5_iniciar.config(state=tk.NORMAL)
        self.btn_f5_detener.config(state=tk.DISABLED)
        self.btn_f5_play_mod.config(state=tk.NORMAL)
        self.btn_f5_play_rec.config(state=tk.NORMAL)

        self.f5_audio_modulado = señal_modulada
        self.f5_audio_recuperado = audio_recuperado

        modo = metricas.get("modo", self.f5_modo.get().upper())
        col = COLOR_FM if modo == "FM" else COLOR_AM
        perd = metricas["perdidos"]
        snr_db = metricas["snr_db"]

        self.lbl_f5_kpi_perd.config(
            text=f"{perd} ({perd/metricas['esperados']*100:.1f}%)" if perd > 0 else "0.0%",
            fg="#475569" if perd > 0 else COLOR_REC
        )
        self.lbl_f5_kpi_snr.config(text=f"{snr_db:+.1f} dB", fg=COLOR_REC if snr_db > 5 else "#475569")

        self.f5_terminal.config(state=tk.NORMAL)
        self.f5_terminal.insert(tk.END, f"[F5-ÉXITO] Completado ({metricas['recibidos']}/{metricas['esperados']} bloques). SNR: {snr_db:+.1f} dB\n")
        self.f5_terminal.see(tk.END)
        self.f5_terminal.config(state=tk.DISABLED)

        # Actualizar gráficas completas usando ventana de voz activa
        self.fig_f5.clf()
        ax1 = self.fig_f5.add_subplot(3, 1, 1)
        ax2 = self.fig_f5.add_subplot(3, 1, 2)
        ax3 = self.fig_f5.add_subplot(3, 1, 3)

        for ax in (ax1, ax2, ax3):
            self._estilizar_eje(ax)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#E2E8F0")

        # 1. Comparativa temporal en ventana con voz activa
        i0, i1 = self._get_f5_voiced_slice()
        i1 = min(i1, len(audio_recuperado))
        t_ms = (np.arange(i0, i1) / FS) * 1000
        ax1.plot(t_ms, self.voz[i0:i1], color=COLOR_VOZ, alpha=0.55, lw=1.6, label="Voz Original")
        ax1.plot(t_ms, audio_recuperado[i0:i1], color=COLOR_REC, lw=1.1, label=f"Voz {modo} Recuperada (SNR {snr_db:+.1f} dB)")
        ax1.set_title(f"1. Forma de Onda: Voz Original vs {modo} Recuperada por Red UDP",
                      fontsize=9, fontweight="bold", color=TEXT_WHITE)
        ax1.set_xlabel("Tiempo (ms)", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylabel("Amplitud", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylim(-1.15, 1.15)
        ax1.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)
        ax1.legend(loc="upper right", facecolor=BG_CARD, edgecolor=BORDER_CARD, labelcolor=TEXT_WHITE, fontsize=7)

        # 2. Espectro señal modulada en tránsito
        N_m = len(señal_modulada)
        mags_m = 20 * np.log10(np.abs(np.fft.rfft(señal_modulada)) / N_m + 1e-12)
        freqs_m = np.fft.rfftfreq(N_m, 1 / FS) / 1000
        ax2.plot(freqs_m, mags_m, color=col, lw=0.9, label=f"Señal {modo} en Red")
        ax2.set_title(f"2. Espectro {modo} en tránsito por UDP",
                      fontsize=9, fontweight="bold", color=TEXT_WHITE)
        ax2.set_xlabel("Frecuencia (kHz)", color=TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("Magnitud (dB)", color=TEXT_MUTED, fontsize=8)
        ax2.set_xlim(0, 24)
        ax2.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)
        ax2.legend(loc="upper right", facecolor=BG_CARD, edgecolor=BORDER_CARD, labelcolor=TEXT_WHITE, fontsize=7)

        # 3. Espectro voz recuperada
        N_r = len(audio_recuperado)
        mags_r = 20 * np.log10(np.abs(np.fft.rfft(audio_recuperado)) / N_r + 1e-12)
        freqs_r = np.fft.rfftfreq(N_r, 1 / FS) / 1000
        ax3.plot(freqs_r, mags_r, color=COLOR_REC, lw=0.9, label="Voz Demodulada")
        ax3.set_title("3. Espectro de voz recuperada (filtro paso-bajo)",
                      fontsize=9, fontweight="bold", color=TEXT_WHITE)
        ax3.set_xlabel("Frecuencia (kHz)", color=TEXT_MUTED, fontsize=8)
        ax3.set_ylabel("Magnitud (dB)", color=TEXT_MUTED, fontsize=8)
        ax3.set_xlim(0, 8)
        ax3.grid(True, color="#E2E8F0", linestyle="--", alpha=0.7)
        ax3.legend(loc="upper right", facecolor=BG_CARD, edgecolor=BORDER_CARD, labelcolor=TEXT_WHITE, fontsize=7)

        self.fig_f5.tight_layout(pad=2.0)
        self.canvas_f5.draw()

    def _cancelar_f5(self):
        self.f5_en_curso = False
        self.btn_f5_iniciar.config(state=tk.NORMAL)
        self.btn_f5_detener.config(state=tk.DISABLED)
        self.f5_terminal.config(state=tk.NORMAL)
        self.f5_terminal.insert(tk.END, "[F5-DETENIDO] Proceso detenido.\n")
        self.f5_terminal.see(tk.END)
        self.f5_terminal.config(state=tk.DISABLED)

    def _reproducir_f5_orig(self):
        self.reproducir_audio(self.voz)

    def _reproducir_f5_mod(self):
        if self.f5_audio_modulado is not None:
            self.reproducir_audio(self.f5_audio_modulado)

    def _reproducir_f5_rec(self):
        if self.f5_audio_recuperado is not None:
            self.reproducir_audio(self.f5_audio_recuperado)

    def _iniciar_grabacion_voz(self):
        if not AUDIO_OK:
            messagebox.showerror("Error de audio", "No se detectó sounddevice para grabar desde el micrófono.")
            return
        self.btn_grabar_voz.config(text="● Grabando (4s)...", state=tk.DISABLED)
        self.lbl_header_sub.config(text="● Micrófono activo — habla durante 4 segundos...", fg="#475569")
        threading.Thread(target=self._hilo_grabar_voz, daemon=True).start()

    def _hilo_grabar_voz(self):
        try:
            dur_s = 4.0
            grab = sd.rec(int(dur_s * FS), samplerate=FS, channels=1, dtype='float32')
            sd.wait()
            voz = grab.flatten().astype(np.float32)
            voz -= np.mean(voz)
            pico = np.max(np.abs(voz))
            if pico > 1e-4:
                voz = voz / pico * 0.95
            self.cola_eventos.put(("VOZ_GRABADA", (voz, "Grabación en vivo")))
        except Exception as e:
            self.cola_eventos.put(("VOZ_ERROR", (f"{type(e).__name__}: {e}",)))

    def _on_voz_actualizada(self, nueva_voz, origen):
        self.voz = nueva_voz
        self.am_mod = modular_am(self.voz)
        self.am_rec = demodular_am(self.am_mod)
        self.fm_mod = modular_fm(self.voz)
        self.fm_rec = demodular_fm(self.fm_mod)
        self.am_canal_ruido = simular_ruido_canal(self.am_mod, self.ruido_snr)
        self.am_rec_ruido = demodular_am(self.am_canal_ruido)
        self.fm_canal_ruido = simular_ruido_canal(self.fm_mod, self.ruido_snr)
        self.fm_rec_ruido = demodular_fm(self.fm_canal_ruido)

        dur_s = len(self.voz) / FS
        self.mascara = cargar_mascara_musical(duracion_s=dur_s + 2.0, fs=FS, ruta_wav="mascara_musical.wav")

        self.btn_grabar_voz.config(text="● Grabar voz (4s)", state=tk.NORMAL)
        self.lbl_header_sub.config(
            text=f"Audio activo: {origen} ({dur_s:.1f} s a 48 kHz) — Listo para modular y enviar por UDP",
            fg=COLOR_REC if "Grabación" in origen else TEXT_SEC
        )
        if hasattr(self, "lbl_launcher_audio"):
            self.lbl_launcher_audio.config(
                text=f"● Audio listo: {origen} ({dur_s:.1f} s)",
                fg=COLOR_REC
            )
            self.btn_entrar_lab.config(state=tk.NORMAL)
        self._plot_f5_inicial()

    def _on_voz_error(self, err):
        self.btn_grabar_voz.config(text="● Grabar voz (4s)", state=tk.NORMAL)
        self.lbl_header_sub.config(text=f"Error de micrófono: {err}", fg="#DC2626")

    def _restaurar_voz_original(self):
        voz = cargar_audio_voz()
        self._on_voz_actualizada(voz, "audio_original.wav")

    def _configurar_toolbar(self, canvas, parent):
        # Barra de herramientas oculta para mantener estética limpia y moderna
        pass

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
