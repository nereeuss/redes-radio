"""
=============================================================
 FASE 1 — Lectura, análisis y reproducción de audio de voz
=============================================================

¿Qué hace este archivo?
  1. Lee un archivo WAV (o genera un tono de prueba si no hay ninguno).
  2. Elimina el valor medio (componente DC) y normaliza la señal.
  3. Muestra gráficas del dominio temporal y del espectro (FFT).
  4. Guarda el audio original procesado.
  5. Permite reproducirlo por los altavoces.

¿Por qué cada decisión técnica?
  - Frecuencia de muestreo 48 000 Hz: el audio de voz ocupa < 4 kHz;
    con 48 kHz tenemos margen de sobra para representar también la
    portadora simulada que usaremos en la Fase 2 (< 24 kHz = Nyquist).
  - float32: formato estándar de audio en Python. Rango [-1, 1],
    4 bytes por muestra. Compatible con NumPy, SciPy y sounddevice.
  - Eliminar DC: un valor medio no nulo desplaza la señal y puede
    causar saturación o distorsión al modular.
  - Normalizar a 0.95: dejamos un 5 % de margen para no saturar (clip).

Autor: Equipo SDR — Fase 1
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTACIONES
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np          # Álgebra de arrays y FFT
import wave                 # Leer y escribir archivos WAV (librería estándar)
import os                   # Manejo de rutas y archivos
import sys                  # Para salir con mensaje de error si falta algo

# sounddevice: reproduce y graba audio usando el driver del sistema operativo.
# Si no está instalado, el programa funciona igualmente pero sin reproducción.
try:
    import sounddevice as sd
    AUDIO_PLAYBACK = True
except ImportError:
    AUDIO_PLAYBACK = False
    print("[AVISO] sounddevice no instalado → sin reproducción de audio.")
    print("        Para instalarlo: pip install sounddevice")

# matplotlib: biblioteca de gráficas científicas en Python.
try:
    import matplotlib.pyplot as plt
    PLOTS = True
except ImportError:
    PLOTS = False
    print("[AVISO] matplotlib no instalado → sin gráficas.")
    print("        Para instalarlo: pip install matplotlib")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────

FS_DEFAULT = 48_000   # Frecuencia de muestreo en Hz. Explicación arriba.
CARPETA_PLOTS = "plots"  # Subcarpeta donde se guardarán las imágenes PNG


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 1: Leer un archivo WAV
# ─────────────────────────────────────────────────────────────────────────────

def leer_wav(ruta: str) -> tuple[np.ndarray, int]:
    """
    Lee un archivo WAV y devuelve la señal como array float32.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo .wav

    Devuelve
    --------
    señal : np.ndarray float32, valores en [-1, 1]
    fs    : int, frecuencia de muestreo del archivo en Hz

    ¿Cómo funciona por dentro?
    - wave.open() lee la cabecera WAV (canales, bits, Hz, nº de frames).
    - readframes() devuelve los bytes crudos de las muestras.
    - np.frombuffer() convierte esos bytes en un array de enteros.
    - Dividimos por el valor máximo del tipo (32767 para int16) para
      llevar los valores al rango float32 [-1, 1].
    - Si el audio es estéreo (2 canales), promediamos ambos → mono.
    """

    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No se encuentra el archivo: '{ruta}'")

    with wave.open(ruta, 'rb') as wf:
        n_canales  = wf.getnchannels()   # 1 = mono, 2 = estéreo
        ancho      = wf.getsampwidth()   # bytes por muestra (1=8bit, 2=16bit)
        fs         = wf.getframerate()   # muestras por segundo (Hz)
        n_frames   = wf.getnframes()     # total de muestras (por canal)
        raw        = wf.readframes(n_frames)  # bytes crudos

    # Mapear anchura en bytes → tipo numpy
    tipos = {1: np.int8, 2: np.int16, 4: np.int32}
    if ancho not in tipos:
        raise ValueError(f"Formato de audio no soportado: {ancho} bytes/muestra")

    muestras = np.frombuffer(raw, dtype=tipos[ancho])

    # Estéreo → mono (promedio de canales izquierdo y derecho)
    if n_canales == 2:
        muestras = muestras.reshape(-1, 2).mean(axis=1)
        print("[INFO] Audio estéreo convertido a mono.")

    # Convertir enteros → float32 en [-1, 1]
    maximo = np.iinfo(tipos[ancho]).max   # 32767 para int16
    señal  = muestras.astype(np.float32) / maximo

    print(f"[OK] WAV leído: '{ruta}'")
    print(f"     Frecuencia de muestreo : {fs} Hz")
    print(f"     Nº de muestras         : {len(señal)}")
    print(f"     Duración               : {len(señal)/fs:.3f} s")
    print(f"     Rango antes de procesar: [{señal.min():.4f}, {señal.max():.4f}]")

    return señal, fs


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 2: Preprocesar la señal (eliminar DC y normalizar)
# ─────────────────────────────────────────────────────────────────────────────

def preprocesar(señal: np.ndarray) -> np.ndarray:
    """
    Elimina el valor medio y normaliza la señal al rango [-0.95, 0.95].

    ¿Por qué eliminar el valor medio (DC)?
        La componente DC es un nivel constante que no aporta información
        de audio. Si está presente, desplaza la onda y puede provocar
        saturación cuando sumamos la portadora en la modulación AM.
        La fórmula es: señal_sin_DC = señal - mean(señal)

    ¿Por qué normalizar a 0.95 y no a 1.0?
        Para dejar un 5 % de margen de seguridad. Si normalizamos
        exactamente a 1.0 y luego multiplicamos por la portadora,
        podríamos superar el rango [-1, 1] y saturar (clipear) la señal.
    """

    # Paso 1: Eliminar DC
    señal = señal - np.mean(señal)

    # Paso 2: Normalizar — dividir por el pico máximo y escalar a 0.95
    pico = np.max(np.abs(señal))
    if pico > 0:
        señal = señal * (0.95 / pico)

    print(f"[OK] Señal preprocesada.")
    print(f"     Media (DC eliminado)   : {np.mean(señal):.8f}  (≈ 0)")
    print(f"     Pico máximo            : {np.max(np.abs(señal)):.4f}  (≤ 0.95)")

    return señal.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 3: Guardar señal como WAV
# ─────────────────────────────────────────────────────────────────────────────

def guardar_wav(señal: np.ndarray, fs: int, ruta: str) -> None:
    """
    Guarda un array float32 como archivo WAV PCM de 16 bits.

    ¿Por qué PCM 16 bits?
        Es el formato más universal. Compatible con cualquier reproductor
        de audio. 16 bits = 65536 niveles → más que suficiente para voz.

    ¿Por qué multiplicar por 32767?
        Para convertir de float32 [-1, 1] a int16 [-32768, 32767].
        Usamos 32767 (no 32768) para evitar desbordamiento en el límite.
    """

    # Convertir float32 → int16
    muestras_int16 = (señal * 32767).astype(np.int16)

    with wave.open(ruta, 'wb') as wf:
        wf.setnchannels(1)          # Mono
        wf.setsampwidth(2)          # 2 bytes = 16 bits por muestra
        wf.setframerate(fs)         # Frecuencia de muestreo
        wf.writeframes(muestras_int16.tobytes())

    tamaño_kb = os.path.getsize(ruta) / 1024
    print(f"[OK] WAV guardado: '{ruta}'  ({tamaño_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 4: Reproducir por los altavoces
# ─────────────────────────────────────────────────────────────────────────────

def reproducir(señal: np.ndarray, fs: int) -> None:
    """
    Reproduce la señal float32 por los altavoces del sistema.

    Usa sounddevice, que se comunica directamente con el driver de audio.
    sd.wait() bloquea hasta que termina la reproducción.
    """

    if not AUDIO_PLAYBACK:
        print("[AVISO] sounddevice no disponible. Instala: pip install sounddevice")
        return

    duracion = len(señal) / fs
    print(f"[OK] Reproduciendo {duracion:.2f} s... (espera a que termine)")
    sd.play(señal, samplerate=fs)
    sd.wait()
    print("[OK] Reproducción terminada.")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 5: Calcular FFT
# ─────────────────────────────────────────────────────────────────────────────

def calcular_fft(señal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la Transformada de Fourier (FFT) y devuelve frecuencias y magnitudes.

    ¿Qué es la FFT?
        La Transformada Rápida de Fourier descompone una señal temporal
        en sus componentes de frecuencia. Nos permite ver QUÉ frecuencias
        contiene la señal y con QUÉ intensidad.

    ¿Por qué rfft() y no fft()?
        rfft() calcula solo la mitad positiva del espectro (0 Hz hasta
        fs/2). Como la señal es real (no compleja), la mitad negativa
        es simétrica → no aporta información extra y se descarta.

    ¿Por qué dividir por N?
        Para normalizar la magnitud. Sin dividir, la magnitud crece con
        el número de muestras N, lo que dificulta comparar señales de
        distinta longitud.

    ¿Por qué dB (20·log10)?
        La escala logarítmica en dB permite ver tanto componentes muy
        fuertes como muy débiles en la misma gráfica. En escala lineal,
        las componentes débiles quedarían aplastadas contra el eje.
    """

    N      = len(señal)
    fft_v  = np.fft.rfft(señal)              # FFT real → mitad positiva
    mags   = np.abs(fft_v) / N               # Magnitud normalizada
    mags_dB = 20 * np.log10(mags + 1e-12)   # Convertir a dB (1e-12 evita log(0))
    freqs  = np.fft.rfftfreq(N, d=1.0 / fs) # Eje de frecuencias en Hz

    return freqs, mags_dB


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 6: Graficar (dominio temporal + espectro)
# ─────────────────────────────────────────────────────────────────────────────

def graficar(señal: np.ndarray, fs: int,
             titulo: str = "Señal de voz — Fase 1",
             guardar_en: str = None,
             ms_fragmento: float = 100.0) -> None:
    """
    Genera dos gráficas lado a lado:
      - Izquierda : fragmento temporal de la señal (en milisegundos)
      - Derecha   : espectro de amplitud FFT (en kHz y dB)

    Parámetros
    ----------
    señal        : array float32 de la señal
    fs           : frecuencia de muestreo en Hz
    titulo       : título de la figura
    guardar_en   : ruta PNG donde guardar (None = no guardar)
    ms_fragmento : cuántos milisegundos mostrar en la gráfica temporal
    """

    if not PLOTS:
        print("[AVISO] matplotlib no disponible. No se pueden generar gráficas.")
        return

    # Calcular número de muestras a mostrar en el fragmento temporal
    n_fragmento = min(int(ms_fragmento / 1000 * fs), len(señal))
    t_ms = np.arange(n_fragmento) / fs * 1000  # Eje tiempo en milisegundos

    freqs, mags_dB = calcular_fft(señal, fs)

    # ── Crear figura con 2 subplots ──────────────────────────────────────────
    fig, ejes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')

    # ── Subplot izquierdo: Dominio temporal ──────────────────────────────────
    # Mostramos solo los primeros ms_fragmento ms para que se vea la forma de onda
    ejes[0].plot(t_ms, señal[:n_fragmento], linewidth=0.8, color='steelblue')
    ejes[0].set_title(f"Dominio temporal — primeros {ms_fragmento:.0f} ms")
    ejes[0].set_xlabel("Tiempo [ms]")          # Unidad: milisegundos
    ejes[0].set_ylabel("Amplitud (normalizada)")
    ejes[0].set_ylim(-1.05, 1.05)              # Rango fijo para comparar fácilmente
    ejes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')  # Línea DC=0
    ejes[0].grid(True, alpha=0.4)

    # ── Subplot derecho: Espectro FFT ────────────────────────────────────────
    # Mostramos de 0 Hz a fs/2 (límite de Nyquist) en kHz
    ejes[1].plot(freqs / 1000, mags_dB, linewidth=0.7, color='darkorange')
    ejes[1].set_title("Espectro de amplitud (FFT)")
    ejes[1].set_xlabel("Frecuencia [kHz]")     # En kHz para mejor lectura
    ejes[1].set_ylabel("Magnitud [dB]")
    ejes[1].set_xlim(0, fs / 2 / 1000)        # Desde 0 hasta Nyquist
    ejes[1].axvline(x=4, color='red', linestyle='--',
                    linewidth=1, label='4 kHz (máx voz)')
    ejes[1].legend(fontsize=9)
    ejes[1].grid(True, alpha=0.4)

    plt.tight_layout()

    # Guardar imagen PNG si se pide
    if guardar_en:
        os.makedirs(os.path.dirname(guardar_en) if os.path.dirname(guardar_en) else ".", exist_ok=True)
        plt.savefig(guardar_en, dpi=150, bbox_inches='tight')
        print(f"[OK] Gráfica guardada en: '{guardar_en}'")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN AUXILIAR: Generar tono sintético (para pruebas sin micrófono)
# ─────────────────────────────────────────────────────────────────────────────

def generar_tono_prueba(duracion: float = 5.0,
                         freq_hz: float = 300.0,
                         fs: int = FS_DEFAULT,
                         ruta: str = "audio_prueba.wav") -> np.ndarray:
    """
    Genera un tono sintético que imita la frecuencia fundamental de una voz.

    ¿Por qué 300 Hz como frecuencia fundamental?
        La voz humana tiene una fundamental típica entre 80 Hz (voz grave)
        y 300 Hz (voz aguda). 300 Hz con armónicos produce un sonido
        parecido a una vocal sostenida, útil para verificar el sistema.

    ¿Qué son los armónicos?
        Un instrumento real (o la voz) produce la frecuencia fundamental
        más sus múltiplos (2×, 3×, 4×...). Al sumarlos con amplitudes
        decrecientes obtenemos un timbre más natural y no un simple "pita".

    ¿Qué es el fade in/out?
        Suaviza el inicio y el final para evitar el "clic" que se produce
        cuando una señal empieza abruptamente desde cero.
    """

    t = np.linspace(0, duracion, int(duracion * fs), endpoint=False)

    # Fundamental + 6 armónicos con amplitudes decrecientes
    amplitudes = [1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]
    señal = sum(a * np.sin(2 * np.pi * freq_hz * (i + 1) * t)
                for i, a in enumerate(amplitudes))

    # Fade in y fade out de 50 ms cada uno
    n_fade = int(0.05 * fs)
    señal[:n_fade]  *= np.linspace(0, 1, n_fade)
    señal[-n_fade:] *= np.linspace(1, 0, n_fade)

    señal = preprocesar(señal.astype(np.float32))
    guardar_wav(señal, fs, ruta)

    print(f"[OK] Tono sintético generado: {freq_hz} Hz, {duracion} s → '{ruta}'")
    return señal


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE PRINCIPAL — Se ejecuta cuando haces: python fase1_audio.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # argparse: módulo estándar de Python para leer argumentos de la línea
    # de comandos. Permite configurar el script sin editar el código.
    parser = argparse.ArgumentParser(
        description="Fase 1 SDR — Leer, analizar y reproducir audio de voz"
    )
    parser.add_argument(
        "--modo", choices=["cargar", "tono"], default="tono",
        help="'cargar': lee un WAV existente | 'tono': genera un tono sintético (por defecto)"
    )
    parser.add_argument(
        "--entrada", default="audio_original.wav",
        help="Ruta del WAV a leer (modo 'cargar')"
    )
    parser.add_argument(
        "--salida", default="audio_original.wav",
        help="Ruta donde guardar el WAV procesado"
    )
    parser.add_argument(
        "--duracion", type=float, default=5.0,
        help="Duración en segundos para el tono sintético (por defecto: 5 s)"
    )
    parser.add_argument(
        "--freq", type=float, default=300.0,
        help="Frecuencia fundamental del tono en Hz (por defecto: 300 Hz)"
    )
    parser.add_argument(
        "--fs", type=int, default=FS_DEFAULT,
        help=f"Frecuencia de muestreo en Hz (por defecto: {FS_DEFAULT})"
    )
    parser.add_argument(
        "--reproducir", action="store_true",
        help="Reproduce el audio por los altavoces después de procesarlo"
    )
    parser.add_argument(
        "--sin-graficas", action="store_true",
        help="No mostrar gráficas (útil en entornos sin pantalla)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FASE 1 — Lectura y análisis de señal de voz")
    print("=" * 60)

    # ── OBTENER LA SEÑAL ─────────────────────────────────────────────────────
    if args.modo == "cargar":
        print(f"\n[MODO] Cargando archivo WAV: '{args.entrada}'")
        señal, fs = leer_wav(args.entrada)
        señal = preprocesar(señal)        # Siempre preprocesar tras leer

    else:  # modo "tono"
        print(f"\n[MODO] Generando tono sintético de prueba...")
        fs     = args.fs
        señal  = generar_tono_prueba(duracion=args.duracion,
                                      freq_hz=args.freq,
                                      fs=fs,
                                      ruta=args.salida)

    # ── GUARDAR WAV ORIGINAL PROCESADO ───────────────────────────────────────
    if args.modo == "cargar":
        # Solo guardamos de nuevo si lo cargamos (el tono ya se guardó)
        guardar_wav(señal, fs, args.salida)

    # ── RESUMEN EN PANTALLA ──────────────────────────────────────────────────
    print("\n── Resumen de la señal ──────────────────────────────")
    print(f"  Frecuencia de muestreo : {fs} Hz")
    print(f"  Nº de muestras         : {len(señal)}")
    print(f"  Duración               : {len(señal)/fs:.3f} s")
    print(f"  Mínimo / Máximo        : {señal.min():.4f} / {señal.max():.4f}")
    print(f"  Media (DC)             : {señal.mean():.8f}")
    print(f"  Límite Nyquist         : {fs//2} Hz (máx frecuencia representable)")
    print("─────────────────────────────────────────────────────")

    # ── REPRODUCIR ───────────────────────────────────────────────────────────
    if args.reproducir:
        reproducir(señal, fs)

    # ── GRAFICAR ─────────────────────────────────────────────────────────────
    if not args.sin_graficas:
        ruta_plot = os.path.join(CARPETA_PLOTS, "fase1_audio.png")
        graficar(señal, fs,
                 titulo="Fase 1 — Señal de voz original",
                 guardar_en=ruta_plot,
                 ms_fragmento=100.0)
