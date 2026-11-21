"""
=============================================================
 FASE 2 — Modulación AM y FM sobre señal de voz
=============================================================

¿Qué hace este archivo?
  1. Lee la señal de voz procesada en Fase 1.
  2. Modula la señal en AM o FM sobre una portadora simulada.
  3. Muestra gráficas comparativas (original vs modulada).
  4. Guarda la señal modulada como WAV.
  5. Demodula localmente para verificar que todo funciona.

Fórmulas implementadas
─────────────────────
AM:  s_AM(t) = [1 + ka · m(t)] · cos(2π·fc·t)

FM:  s_FM(t) = cos(2π·fc·t + 2π·kf · ∫m(τ)dτ)

RESTRICCIÓN NYQUIST:
  fc debe estar claramente por debajo de fs/2.
  Con fs=48000 Hz → Nyquist = 24000 Hz → fc máxima segura ≈ 12000 Hz

Autor: Equipo SDR — Fase 2
"""

import numpy as np
import wave
import os
import sys

try:
    import matplotlib.pyplot as plt
    PLOTS = True
except ImportError:
    PLOTS = False
    print("[AVISO] matplotlib no instalado. pip install matplotlib")

try:
    from scipy.signal import butter, filtfilt, hilbert
    SCIPY = True
except ImportError:
    SCIPY = False
    print("[AVISO] scipy no instalado. pip install scipy")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

FS_DEFAULT       = 48_000   # Hz — frecuencia de muestreo
FC_AM_DEFAULT    = 8_000    # Hz — portadora AM por defecto
FC_FM_DEFAULT    = 10_000   # Hz — portadora FM por defecto
KA_DEFAULT       = 0.9      # Índice de modulación AM (0 < ka ≤ 1)
KF_DEFAULT       = 2_000    # Desviación de frecuencia FM en Hz


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDAD: Verificar límite de Nyquist
# ─────────────────────────────────────────────────────────────────────────────

def verificar_nyquist(freq: float, fs: int, etiqueta: str) -> None:
    """
    Comprueba que la frecuencia no supera el límite de Nyquist.

    ¿Qué es el límite de Nyquist?
        Con una frecuencia de muestreo fs, el máximo que podemos
        representar es fs/2. Por encima de ese límite ocurre aliasing:
        la frecuencia se "dobla" y aparece como una frecuencia falsa.

    Ejemplo con fs=48000 Hz:
        Nyquist = 24000 Hz
        Una frecuencia de 25000 Hz aparecería como 23000 Hz (alias)
        → distorsión irreversible
    """
    nyquist = fs / 2
    if freq >= nyquist:
        raise ValueError(
            f"[{etiqueta}] ERROR: {freq} Hz supera Nyquist ({nyquist} Hz).\n"
            f"Reduce fc o aumenta fs."
        )
    if freq > nyquist * 0.75:
        print(f"[{etiqueta}] ADVERTENCIA: {freq} Hz supera el 75% de Nyquist "
              f"({nyquist} Hz). Considera reducir fc.")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 1: Generar la onda portadora
# ─────────────────────────────────────────────────────────────────────────────

def generar_portadora(n_muestras: int, fc: float, fs: int) -> np.ndarray:
    """
    Genera una onda coseno pura: la portadora.

    Fórmula: c(t) = cos(2π · fc · t)

    ¿Por qué coseno y no seno?
        Matemáticamente es equivalente. El coseno es convención
        en comunicaciones porque facilita el análisis de fase.

    ¿Por qué una portadora?
        La voz (0-4 kHz) no se puede "ver" directamente en el espectro
        cuando hay varias señales. La portadora actúa como "transportador":
        traslada la información a otra zona del espectro (alrededor de fc).
        En radio real, cada emisora tiene su propia fc (88 MHz, 92 MHz...)
        para no interferirse. Aquí lo simulamos en software.
    """
    t = np.arange(n_muestras) / fs          # Vector de tiempo: [0, 1/fs, 2/fs, ...]
    portadora = np.cos(2 * np.pi * fc * t)  # Onda coseno a frecuencia fc
    return portadora.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 2: Modulación AM
# ─────────────────────────────────────────────────────────────────────────────

def modular_am(señal: np.ndarray, fs: int,
               fc: float = FC_AM_DEFAULT,
               ka: float = KA_DEFAULT) -> np.ndarray:
    """
    Modulación de Amplitud (AM).

    Fórmula completa:
        s_AM(t) = [1 + ka · m(t)] · cos(2π · fc · t)

    Desglose:
        m(t)           → señal de voz (mensaje)
        ka             → índice de modulación [0, 1]
        cos(2π·fc·t)   → portadora

        [1 + ka·m(t)]  → envolvente (la amplitud de la portadora
                          varía según la voz)

    ¿Qué modifica AM de la portadora?
        Solo la AMPLITUD. La frecuencia y la fase no cambian.
        Cuando la voz sube → la portadora se hace más grande.
        Cuando la voz baja → la portadora se hace más pequeña.

    ¿Qué es ka (índice de modulación)?
        ka = 0.0 → no hay modulación (portadora pura)
        ka = 0.5 → modulación al 50%
        ka = 1.0 → modulación al 100% (máximo sin distorsión)
        ka > 1.0 → sobremodulación (distorsión, no recuperable)

    Parámetros
    ----------
    señal : np.ndarray float32 normalizada en [-1, 1]
    fs    : frecuencia de muestreo en Hz
    fc    : frecuencia portadora en Hz
    ka    : índice de modulación (0 < ka ≤ 1)
    """
    verificar_nyquist(fc, fs, "AM")

    portadora = generar_portadora(len(señal), fc, fs)

    # Fórmula AM: s(t) = (1 + ka·m(t)) · c(t)
    s_am = (1.0 + ka * señal) * portadora

    # Normalizar para no saturar la cadena
    pico = np.max(np.abs(s_am))
    if pico > 0:
        s_am = s_am / pico * 0.95

    print(f"\n[AM] Modulación completada:")
    print(f"     Portadora fc : {fc} Hz")
    print(f"     Índice ka    : {ka}")
    print(f"     Muestras     : {len(s_am)}")
    print(f"     Pico máximo  : {np.max(np.abs(s_am)):.4f}")
    print(f"     BW teórico   : {2 * 4000} Hz (2 × máx. voz)")

    return s_am.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 3: Demodulación AM
# ─────────────────────────────────────────────────────────────────────────────

def demodular_am(s_am: np.ndarray, fs: int,
                 fc: float = FC_AM_DEFAULT,
                 corte_lp: float = 4000.0) -> np.ndarray:
    """
    Demodulación AM — Detector de producto.

    Proceso en 3 pasos:
        1. Multiplicar la señal AM por la misma portadora
        2. Filtrar con paso-bajo para quedarnos solo con la voz
        3. Eliminar DC y normalizar

    ¿Por qué funciona esto?
        s_AM(t) = [1 + ka·m(t)] · cos(2π·fc·t)

        Al multiplicar por cos(2π·fc·t):
        s_AM × cos = [1 + ka·m(t)] · cos²(2π·fc·t)

        Usando la identidad: cos²(x) = (1 + cos(2x)) / 2

        = [1 + ka·m(t)] · (1 + cos(4π·fc·t)) / 2
        = [1 + ka·m(t)]/2  +  [1 + ka·m(t)]·cos(4π·fc·t)/2
              ↑                        ↑
         ESTO queremos            esto está a 2·fc (muy alto)
         (la voz, en bajas        → el filtro paso-bajo lo elimina
          frecuencias)

    Parámetros
    ----------
    s_am      : señal AM recibida
    fs        : Hz
    fc        : frecuencia portadora conocida por el receptor
    corte_lp  : frecuencia de corte del filtro paso-bajo en Hz
    """
    if not SCIPY:
        print("[ERROR] scipy necesario para demodular. pip install scipy")
        return s_am

    portadora = generar_portadora(len(s_am), fc, fs)

    # Paso 1: mezcla (multiplicar por portadora local)
    mezclado = s_am * portadora

    # Paso 2: filtro paso-bajo Butterworth orden 5
    # ¿Por qué Butterworth? Respuesta plana en la banda de paso,
    # sin ondulaciones. Orden 5 = buen equilibrio entre pendiente y estabilidad.
    nyq  = fs / 2.0
    b, a = butter(5, corte_lp / nyq, btype='low')
    demod = filtfilt(b, a, mezclado)   # filtfilt = sin retardo de fase

    # Paso 3: eliminar DC y normalizar
    demod = demod - np.mean(demod)
    pico  = np.max(np.abs(demod))
    if pico > 0:
        demod = demod / pico * 0.95

    print(f"\n[AM] Demodulación completada. Filtro paso-bajo: {corte_lp} Hz")
    return demod.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 4: Modulación FM
# ─────────────────────────────────────────────────────────────────────────────

def modular_fm(señal: np.ndarray, fs: int,
               fc: float = FC_FM_DEFAULT,
               kf: float = KF_DEFAULT) -> np.ndarray:
    """
    Modulación de Frecuencia (FM).

    Fórmula:
        φ(t) = 2π·fc·t  +  2π·kf · ∫m(τ)dτ
        s_FM(t) = cos(φ(t))

    ¿Qué modifica FM de la portadora?
        Solo la FRECUENCIA instantánea. La amplitud no cambia.
        Cuando la voz sube → la portadora oscila más rápido.
        Cuando la voz baja → la portadora oscila más lento.
        Cuando la voz vale 0 → la portadora oscila exactamente a fc.

    ¿Qué es kf (desviación de frecuencia)?
        Es cuántos Hz se aleja la frecuencia instantánea de fc
        cuando la señal vale su máximo (1.0).

        Ejemplo con fc=10000 Hz, kf=2000 Hz:
          m(t)=0   → frecuencia instantánea = 10000 Hz
          m(t)=1   → frecuencia instantánea = 12000 Hz
          m(t)=-1  → frecuencia instantánea =  8000 Hz

    ¿Por qué la integral de m(t)?
        La frecuencia es la derivada de la fase.
        Para controlar la frecuencia, integramos la señal
        y usamos esa integral como modulador de fase.
        Así conseguimos que la frecuencia instantánea
        sea proporcional a m(t).

    Ancho de banda FM (Regla de Carson):
        BW ≈ 2 · (kf + f_max_voz) = 2 · (2000 + 4000) = 12000 Hz

    Parámetros
    ----------
    señal : señal de voz normalizada en [-1, 1]
    fs    : Hz
    fc    : frecuencia portadora en Hz
    kf    : desviación máxima de frecuencia en Hz
    """
    verificar_nyquist(fc + kf, fs, "FM")

    t = np.arange(len(señal)) / fs

    # Integral numérica de m(t) → suma acumulada / fs
    # (aproximación de la integral por suma de Riemann)
    integral = np.cumsum(señal) / fs

    # Fase instantánea total
    fase = 2 * np.pi * fc * t + 2 * np.pi * kf * integral

    # Señal FM
    s_fm = np.cos(fase).astype(np.float32)

    bw_carson = 2 * (kf + 4000)
    print(f"\n[FM] Modulación completada:")
    print(f"     Portadora fc     : {fc} Hz")
    print(f"     Desviación kf    : {kf} Hz")
    print(f"     BW (Carson)      : {bw_carson} Hz")
    print(f"     Muestras         : {len(s_fm)}")

    return s_fm


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 5: Demodulación FM
# ─────────────────────────────────────────────────────────────────────────────

def demodular_fm(s_fm: np.ndarray, fs: int,
                 fc: float = FC_FM_DEFAULT,
                 kf: float = KF_DEFAULT,
                 corte_lp: float = 4000.0) -> np.ndarray:
    """
    Demodulación FM — Discriminador de frecuencia instantánea.

    Proceso en 4 pasos:
        1. Señal analítica (transformada de Hilbert)
        2. Fase instantánea (arctan)
        3. Frecuencia instantánea (derivada de la fase)
        4. Restar portadora, normalizar y filtrar

    ¿Qué es la señal analítica?
        Convierte la señal real en compleja añadiendo la parte
        imaginaria (transformada de Hilbert). Esto permite
        extraer la fase y la amplitud instantáneas sin ambigüedad.

    ¿Qué es np.unwrap?
        La fase va de -π a +π y da "saltos" en los bordes.
        unwrap elimina esos saltos para tener una fase continua
        que crece suavemente con el tiempo.
    """
    if not SCIPY:
        print("[ERROR] scipy necesario para demodular FM. pip install scipy")
        return s_fm

    # Paso 1: señal analítica vía transformada de Hilbert
    analitica = hilbert(s_fm)

    # Paso 2: fase instantánea continua
    fase_inst = np.unwrap(np.angle(analitica))

    # Paso 3: frecuencia instantánea = derivada de la fase / 2π
    freq_inst = np.diff(fase_inst) * fs / (2 * np.pi)

    # Paso 4: eliminar portadora y normalizar por desviación
    demod = (freq_inst - fc) / kf

    # Añadir muestra perdida por diff() y filtrar
    demod = np.append(demod, 0.0)

    nyq  = fs / 2.0
    b, a = butter(5, corte_lp / nyq, btype='low')
    demod = filtfilt(b, a, demod)

    # Eliminar DC y normalizar
    demod = demod - np.mean(demod)
    pico  = np.max(np.abs(demod))
    if pico > 0:
        demod = demod / pico * 0.95

    print(f"\n[FM] Demodulación completada. Filtro paso-bajo: {corte_lp} Hz")
    return demod.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 6: Calcular error entre original y recuperada
# ─────────────────────────────────────────────────────────────────────────────

def calcular_error(original: np.ndarray, recuperada: np.ndarray) -> dict:
    """
    Calcula métricas de error entre la señal original y la recuperada.

    Métricas:
        MSE  : Error Cuadrático Medio — promedio de (diferencia)²
        NMSE : MSE normalizado — MSE / potencia_original
        SNR  : Relación Señal/Ruido en dB — cuánto mayor es la señal
               respecto al error
    """
    n = min(len(original), len(recuperada))
    orig = original[:n]
    rec  = recuperada[:n]

    mse  = np.mean((orig - rec) ** 2)
    pot  = np.mean(orig ** 2)
    nmse = mse / (pot + 1e-12)
    snr  = 10 * np.log10(pot / (mse + 1e-12))

    return {"MSE": mse, "NMSE": nmse, "SNR_dB": snr}


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 7: Gráficas comparativas
# ─────────────────────────────────────────────────────────────────────────────

def graficar_modulacion(original: np.ndarray,
                        modulada: np.ndarray,
                        recuperada: np.ndarray,
                        fs: int,
                        tipo: str = "AM",
                        fc: float = 8000.0,
                        guardar_en: str = None) -> None:
    """
    Genera una figura con 3 filas × 2 columnas:

    Fila 1: Señal original    → tiempo | espectro
    Fila 2: Señal modulada    → tiempo | espectro
    Fila 3: Señal recuperada  → tiempo | espectro

    Permite comparar visualmente los tres estados de la señal
    y verificar que la modulación/demodulación funciona.
    """
    if not PLOTS:
        return

    ms_fragmento = 5.0
    n_frag = int(ms_fragmento / 1000 * fs)
    n_frag = min(n_frag, len(original), len(modulada), len(recuperada))
    t_ms   = np.arange(n_frag) / fs * 1000

    fig, ejes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"Fase 2 — Modulación {tipo}  (fc = {fc/1000:.1f} kHz, fs = {fs} Hz)",
                 fontsize=14, fontweight='bold')

    filas = [
        (original,   "Señal ORIGINAL",   'steelblue'),
        (modulada,   f"Señal {tipo} MODULADA", 'darkorange'),
        (recuperada, "Señal RECUPERADA", 'seagreen'),
    ]

    for i, (sig, etiqueta, color) in enumerate(filas):

        # ── Columna izquierda: dominio temporal ───────────────────────────
        ejes[i, 0].plot(t_ms, sig[:n_frag], color=color, linewidth=0.7)
        ejes[i, 0].set_title(f"{etiqueta} — tiempo")
        ejes[i, 0].set_xlabel("Tiempo [ms]")
        ejes[i, 0].set_ylabel("Amplitud")
        ejes[i, 0].set_ylim(-1.1, 1.1)
        ejes[i, 0].axhline(0, color='gray', linewidth=0.4, linestyle='--')
        ejes[i, 0].grid(True, alpha=0.3)

        # ── Columna derecha: espectro FFT ─────────────────────────────────
        N     = len(sig)
        fft_v = np.fft.rfft(sig)
        mags  = 20 * np.log10(np.abs(fft_v) / N + 1e-12)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs) / 1000   # kHz

        ejes[i, 1].plot(freqs, mags, color=color, linewidth=0.6)
        ejes[i, 1].set_title(f"{etiqueta} — espectro")
        ejes[i, 1].set_xlabel("Frecuencia [kHz]")
        ejes[i, 1].set_ylabel("Magnitud [dB]")
        ejes[i, 1].set_xlim(0, fs / 2 / 1000)
        ejes[i, 1].axvline(x=4,       color='gray', linestyle=':', linewidth=1,
                            label='4 kHz (máx voz)')
        ejes[i, 1].axvline(x=fc/1000, color='red',  linestyle='--', linewidth=1,
                            label=f'fc={fc/1000:.1f} kHz')
        ejes[i, 1].legend(fontsize=7)
        ejes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if guardar_en:
        os.makedirs(os.path.dirname(guardar_en) if os.path.dirname(guardar_en) else ".", exist_ok=True)
        plt.savefig(guardar_en, dpi=150, bbox_inches='tight')
        print(f"[OK] Gráfica guardada: '{guardar_en}'")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES WAV (reutilizadas de Fase 1)
# ─────────────────────────────────────────────────────────────────────────────

def cargar_wav(ruta: str) -> tuple[np.ndarray, int]:
    with wave.open(ruta, 'rb') as wf:
        n_ch = wf.getnchannels()
        ancho = wf.getsampwidth()
        fs = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    tipos = {1: np.int8, 2: np.int16, 4: np.int32}
    muestras = np.frombuffer(raw, dtype=tipos[ancho])
    if n_ch == 2:
        muestras = muestras.reshape(-1, 2).mean(axis=1)
    señal = muestras.astype(np.float32) / np.iinfo(tipos[ancho]).max
    señal = señal - np.mean(señal)
    pico = np.max(np.abs(señal))
    if pico > 0:
        señal = señal * (0.95 / pico)
    return señal, fs


def guardar_wav(señal: np.ndarray, fs: int, ruta: str) -> None:
    int16 = (señal * 32767).astype(np.int16)
    with wave.open(ruta, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(int16.tobytes())
    print(f"[OK] WAV guardado: '{ruta}'")


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fase 2 SDR — Modulación AM y FM"
    )
    parser.add_argument("--entrada", default="audio_original.wav",
                        help="WAV de entrada (Fase 1)")
    parser.add_argument("--mod", choices=["AM", "FM"], default="AM",
                        help="Tipo de modulación")
    parser.add_argument("--fc", type=float, default=8000.0,
                        help="Frecuencia portadora en Hz (por defecto 8000)")
    parser.add_argument("--ka", type=float, default=0.9,
                        help="[AM] Índice de modulación (por defecto 0.9)")
    parser.add_argument("--kf", type=float, default=2000.0,
                        help="[FM] Desviación de frecuencia en Hz (por defecto 2000)")
    parser.add_argument("--sin-graficas", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  FASE 2 — Modulación {args.mod}")
    print("=" * 60)

    # ── Cargar señal de Fase 1 ────────────────────────────────────────────────
    if not os.path.isfile(args.entrada):
        print(f"[INFO] No se encuentra '{args.entrada}'. Generando tono...")
        from fase1_audio import generar_tono_prueba
        señal = generar_tono_prueba(duracion=5.0, freq_hz=300.0,
                                     fs=FS_DEFAULT, ruta=args.entrada)
        fs = FS_DEFAULT
    else:
        señal, fs = cargar_wav(args.entrada)
        print(f"[OK] Señal cargada: {len(señal)} muestras a {fs} Hz")

    # ── Modular y demodular ───────────────────────────────────────────────────
    if args.mod == "AM":
        modulada   = modular_am(señal, fs, fc=args.fc, ka=args.ka)
        recuperada = demodular_am(modulada, fs, fc=args.fc)
        guardar_wav(modulada,   fs, "audio_am_modulada.wav")
        guardar_wav(recuperada, fs, "audio_am_recuperada_local.wav")
    else:
        modulada   = modular_fm(señal, fs, fc=args.fc, kf=args.kf)
        recuperada = demodular_fm(modulada, fs, fc=args.fc, kf=args.kf)
        guardar_wav(modulada,   fs, "audio_fm_modulada.wav")
        guardar_wav(recuperada, fs, "audio_fm_recuperada_local.wav")

    # ── Métricas de error ─────────────────────────────────────────────────────
    err = calcular_error(señal, recuperada)
    print(f"\n── Métricas de calidad ──────────────────────────────")
    print(f"  MSE              : {err['MSE']:.6f}")
    print(f"  NMSE             : {err['NMSE']:.6f}")
    print(f"  SNR              : {err['SNR_dB']:.2f} dB")
    print(f"─────────────────────────────────────────────────────")

    # ── Gráficas ──────────────────────────────────────────────────────────────
    if not args.sin_graficas:
        ruta_plot = f"plots/fase2_{args.mod.lower()}.png"
        graficar_modulacion(señal, modulada, recuperada,
                            fs=fs, tipo=args.mod, fc=args.fc,
                            guardar_en=ruta_plot)
