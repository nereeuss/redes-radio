"""
=============================================================
 MODULADOR EN LÍNEA — SDR Fase 5
=============================================================
Asignatura: Redes de Ordenadores / Tratamiento de Señales
Fase 5: Integración Modulación AM/FM + Transporte UDP

Este módulo implementa la modulación y demodulación BLOQUE A BLOQUE,
lo que permite procesar el audio en tiempo real conforme llegan los
datagramas UDP, en lugar de hacerlo sobre la señal completa como en Fase 2.

DIFERENCIA CLAVE respecto a fase2_modulacion.py:
─────────────────────────────────────────────────
  Fase 2 → modula la señal ENTERA de golpe:
              integral = np.cumsum(señal_entera) / fs

  Fase 5 → modula BLOQUE A BLOQUE propagando la fase:
              al final del bloque i guarda fase_acumulada
              el bloque i+1 arranca desde esa fase
              → sin "clics" en las costuras entre paquetes

¿Por qué es crítica la propagación de fase en FM?
─────────────────────────────────────────────────
  La señal FM es: s(t) = cos(2π·fc·t + 2π·kf·∫m(τ)dτ)
  La integral es CONTINUA. Si cada bloque resetea el integral a 0:
    - Bloque 0 termina con fase  φ₀
    - Bloque 1 empieza con fase  0  ← SALTO BRUSCO → clic audible
  Solución: pasar φ₀ como argumento a la función del bloque 1.

Para AM no hay fase que propagar, pero sí necesitamos
  mantener coherencia de portadora entre bloques (mismo vector t).
  La solución: pasar el índice de inicio del bloque (sample_offset).
"""

import os
import sys

# Añadir vendor al path si existe (para numpy y scipy)
_vendor = os.path.join(os.path.dirname(__file__), "vendor")
if os.path.isdir(_vendor) and _vendor not in sys.path:
    sys.path.insert(0, _vendor)

import math
import struct
import numpy as np
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES POR DEFECTO (coherentes con fase2_modulacion.py)
# ─────────────────────────────────────────────────────────────────────────────

FS_DEFAULT    = 48_000      # Hz
FC_AM_DEFAULT = 8_000.0     # Hz — portadora AM
FC_FM_DEFAULT = 10_000.0    # Hz — portadora FM
KA_DEFAULT    = 0.90        # Índice de modulación AM
KF_DEFAULT    = 2_000.0     # Desviación máxima FM en Hz
CORTE_LP_AM   = 4_000.0     # Filtro paso-bajo tras demodulación AM (Hz)
CORTE_LP_FM   = 4_000.0     # Filtro paso-bajo tras demodulación FM (Hz)


# ─────────────────────────────────────────────────────────────────────────────
# CLASE: Estado de la FM entre bloques
# ─────────────────────────────────────────────────────────────────────────────

class EstadoFM:
    """
    Encapsula la fase acumulada de la integral ∫m(τ)dτ entre bloques.

    Al final de cada bloque FM, el transmisor llama a:
        estado.actualizar(bloque, fs, kf)

    El siguiente bloque arranca desde ese estado, garantizando
    continuidad de fase sin clics ni distorsiones en las costuras.
    """
    def __init__(self):
        self.integral_acumulada: float = 0.0   # ∫m(τ)dτ hasta el último bloque
        self.sample_offset: int = 0            # Muestra global (para t = offset/fs)

    def actualizar(self, bloque: np.ndarray, fs: int) -> None:
        """Integra el bloque actual y avanza el contador de muestras."""
        self.integral_acumulada += float(np.sum(bloque)) / fs
        self.sample_offset += len(bloque)

    def reset(self) -> None:
        """Reinicia el estado (inicio de nueva sesión de transmisión)."""
        self.integral_acumulada = 0.0
        self.sample_offset = 0


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES INTERNAS
# ─────────────────────────────────────────────────────────────────────────────

def _float32_a_int16(arr: np.ndarray) -> bytes:
    """Convierte float32 [-1,1] a bytes PCM int16 little-endian."""
    clipped = np.clip(arr, -1.0, 1.0)
    return (clipped * 32767.0).astype('<i2').tobytes()


def _int16_a_float32(data: bytes) -> np.ndarray:
    """Convierte bytes PCM int16 little-endian a float32 [-1,1]."""
    arr = np.frombuffer(data, dtype='<i2').astype(np.float32)
    return arr / 32767.0


def _butter_lp(corte_hz: float, fs: int, orden: int = 5):
    """
    Calcula coeficientes de un filtro Butterworth paso-bajo.
    Devuelve (b, a) listos para scipy.signal.filtfilt,
    o None si scipy no está disponible (degradación elegante).
    """
    try:
        from scipy.signal import butter
        nyq = fs / 2.0
        b, a = butter(orden, corte_hz / nyq, btype='low')
        return b, a
    except ImportError:
        return None, None


def _aplicar_filtro(señal: np.ndarray, b, a) -> np.ndarray:
    """Aplica filtfilt si scipy está disponible, si no devuelve la señal sin filtrar."""
    if b is None:
        return señal
    try:
        from scipy.signal import filtfilt
        return filtfilt(b, a, señal).astype(np.float32)
    except Exception:
        return señal


# ─────────────────────────────────────────────────────────────────────────────
# MODULACIÓN AM — BLOQUE A BLOQUE
# ─────────────────────────────────────────────────────────────────────────────

def modular_am_bloque(
    bloque: np.ndarray,
    fc: float = FC_AM_DEFAULT,
    ka: float = KA_DEFAULT,
    fs: int = FS_DEFAULT,
    sample_offset: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Modula un bloque de audio en AM manteniendo coherencia de portadora.

    Fórmula:
        s_AM[n] = (1 + ka · m[n]) · cos(2π · fc · (n + offset) / fs)

    El parámetro `sample_offset` es el índice global de la primera muestra
    del bloque, garantizando que la portadora es continua entre bloques.

    Parámetros
    ----------
    bloque        : np.ndarray float32 normalizado en [-1, 1]
    fc            : frecuencia portadora en Hz
    ka            : índice de modulación (0 < ka ≤ 1)
    fs            : frecuencia de muestreo en Hz
    sample_offset : índice de muestra global del inicio del bloque

    Retorna
    -------
    (señal_am, nuevo_offset)
    """
    n = np.arange(len(bloque)) + sample_offset
    portadora = np.cos(2.0 * np.pi * fc * n / fs).astype(np.float32)
    s_am = (1.0 + ka * bloque) * portadora

    # Escala fija por cota analítica (1 + ka) para preservar la envolvente entre bloques
    s_am = np.clip(s_am / (1.0 + ka) * 0.95, -0.95, 0.95)

    nuevo_offset = sample_offset + len(bloque)
    return s_am.astype(np.float32), nuevo_offset


def demodular_am_bloque(
    bloque_am: np.ndarray,
    fc: float = FC_AM_DEFAULT,
    ka: float = KA_DEFAULT,
    fs: int = FS_DEFAULT,
    sample_offset: int = 0,
    corte_lp: float = CORTE_LP_AM,
) -> Tuple[np.ndarray, int]:
    """
    Demodula un bloque AM (detector de producto coherente).

    Proceso:
      1. Multiplicar por portadora local (mismo fc y mismo offset)
      2. Filtro paso-bajo Butterworth para eliminar la componente en 2·fc
      3. Eliminar DC y normalizar

    Parámetros
    ----------
    bloque_am     : bloque de señal AM recibida (float32)
    fc, ka, fs    : mismos parámetros usados en el transmisor
    sample_offset : índice global de la primera muestra de este bloque
    corte_lp      : frecuencia de corte del filtro paso-bajo en Hz

    Retorna
    -------
    (audio_recuperado, nuevo_offset)
    """
    n = np.arange(len(bloque_am)) + sample_offset
    portadora = np.cos(2.0 * np.pi * fc * n / fs).astype(np.float32)

    mezclado = bloque_am * portadora

    b, a = _butter_lp(corte_lp, fs)
    demod = _aplicar_filtro(mezclado, b, a)

    # Ganancia analítica del detector coherente sin amplificar bloques silenciosos
    demod = demod * (2.0 * (1.0 + ka) / (0.95 * max(ka, 1e-6)))
    demod = np.clip(demod - np.mean(demod), -0.95, 0.95)

    nuevo_offset = sample_offset + len(bloque_am)
    return demod.astype(np.float32), nuevo_offset


# ─────────────────────────────────────────────────────────────────────────────
# MODULACIÓN FM — BLOQUE A BLOQUE CON PROPAGACIÓN DE FASE
# ─────────────────────────────────────────────────────────────────────────────

def modular_fm_bloque(
    bloque: np.ndarray,
    fc: float = FC_FM_DEFAULT,
    kf: float = KF_DEFAULT,
    fs: int = FS_DEFAULT,
    estado: EstadoFM = None,
) -> Tuple[np.ndarray, EstadoFM]:
    """
    Modula un bloque en FM propagando la fase acumulada ∫m(τ)dτ.

    Fórmula completa para el bloque i:
        φ_local[n]  = 2π·fc·(n + offset)/fs  +  2π·kf·(∫_prev + ∑_{k=0}^{n} m[k]/fs)
        s_FM[n]     = cos(φ_local[n])

    donde ∫_prev es la integral acumulada de los bloques anteriores.

    Parámetros
    ----------
    bloque  : señal de mensaje float32 normalizada en [-1, 1]
    fc, kf, fs : parámetros del modulador FM
    estado  : objeto EstadoFM con la fase acumulada del bloque anterior
              (None = inicio de sesión, se crea uno vacío)

    Retorna
    -------
    (señal_fm, estado_actualizado)
    """
    if estado is None:
        estado = EstadoFM()

    n = np.arange(len(bloque)) + estado.sample_offset

    # Integral local dentro del bloque (Riemann) + arrastre del bloque anterior
    integral_local = np.cumsum(bloque) / fs
    integral_total = estado.integral_acumulada + integral_local

    # Fase instantánea total
    fase = 2.0 * np.pi * fc * n / fs + 2.0 * np.pi * kf * integral_total

    s_fm = np.cos(fase).astype(np.float32)

    # Actualizar estado para el próximo bloque
    estado.actualizar(bloque, fs)

    return s_fm, estado


def demodular_fm_bloque(
    bloque_fm: np.ndarray,
    fc: float = FC_FM_DEFAULT,
    kf: float = KF_DEFAULT,
    fs: int = FS_DEFAULT,
    corte_lp: float = CORTE_LP_FM,
) -> np.ndarray:
    """
    Demodula un bloque FM usando discriminador de frecuencia instantánea.

    Proceso (igual que fase2_modulacion.py pero sobre bloques):
      1. Señal analítica vía transformada de Hilbert
      2. Fase instantánea continua (np.unwrap)
      3. Frecuencia instantánea = derivada de la fase / 2π
      4. Restar portadora fc, dividir por kf → recuperar m(t)
      5. Filtro paso-bajo para eliminar ruido de alta frecuencia

    Nota: El demodulador FM no necesita propagar estado entre bloques
    porque trabaja LOCALMENTE en la derivada de fase. La continuidad
    entre bloques la garantiza que el transmisor propagó correctamente.

    Parámetros
    ----------
    bloque_fm : bloque de señal FM recibida (float32)
    fc, kf    : mismos parámetros del transmisor
    fs        : frecuencia de muestreo
    corte_lp  : filtro paso-bajo final en Hz

    Retorna
    -------
    np.ndarray float32 con el audio recuperado
    """
    try:
        from scipy.signal import hilbert
    except ImportError:
        # Fallback sin scipy: diferenciador de fase simple
        return _demodular_fm_simple(bloque_fm, fc, kf, fs, corte_lp)

    # 1. Señal analítica
    analitica = hilbert(bloque_fm)

    # 2. Fase instantánea continua
    fase_inst = np.unwrap(np.angle(analitica))

    # 3. Frecuencia instantánea
    freq_inst = np.diff(fase_inst) * fs / (2.0 * np.pi)

    # 4. Eliminar portadora y normalizar por desviación
    demod = (freq_inst - fc) / kf
    demod = np.append(demod, 0.0)   # Restaurar muestra perdida por diff()

    # 5. Filtro paso-bajo sin inflar silencio
    b, a = _butter_lp(corte_lp, fs)
    demod = _aplicar_filtro(demod, b, a)
    demod = np.clip(demod - np.mean(demod), -0.95, 0.95)

    return demod.astype(np.float32)


def _demodular_fm_simple(bloque_fm, fc, kf, fs, corte_lp):
    """Demodulador FM de emergencia sin scipy (diferencias finitas)."""
    fase = np.arctan2(
        bloque_fm[1:] * bloque_fm[:-1],
        bloque_fm[:-1] ** 2 + bloque_fm[1:] ** 2 + 1e-12
    )
    freq_inst = fase * fs / (2.0 * np.pi)
    demod = (freq_inst - fc) / (kf + 1e-12)
    demod = np.append(demod, 0.0)
    demod = np.clip(demod - np.mean(demod), -0.95, 0.95)
    return demod.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SERIALIZACIÓN: float32 → int16 y viceversa (para payload UDP)
# ─────────────────────────────────────────────────────────────────────────────

def serializar_bloque(bloque_float32: np.ndarray) -> bytes:
    """
    Convierte un bloque float32 en bytes PCM int16 para el payload UDP.
    Escala [-1, 1] → [-32767, 32767] y serializa en little-endian.
    """
    return _float32_a_int16(bloque_float32)


def deserializar_bloque(payload: bytes) -> np.ndarray:
    """
    Reconstruye un bloque float32 desde bytes PCM int16 del payload UDP.
    """
    return _int16_a_float32(payload)


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS: SNR entre original y recuperado
# ─────────────────────────────────────────────────────────────────────────────

def calcular_snr(original: np.ndarray, recuperado: np.ndarray) -> dict:
    """
    Calcula métricas de fidelidad entre la señal original y la recuperada.
    Devuelve MSE, NMSE y SNR en dB (coherente con fase2_modulacion.calcular_error).
    """
    n = min(len(original), len(recuperado))
    orig = original[:n].astype(np.float64)
    rec  = recuperado[:n].astype(np.float64)

    mse  = float(np.mean((orig - rec) ** 2))
    pot  = float(np.mean(orig ** 2))
    nmse = mse / (pot + 1e-12)
    snr  = 10.0 * math.log10(pot / (mse + 1e-12))

    return {"MSE": mse, "NMSE": nmse, "SNR_dB": snr}


# ─────────────────────────────────────────────────────────────────────────────
# AUTOTEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import wave, os

    print("=" * 60)
    print("  AUTOTEST — Modulador en Línea (Fase 5)")
    print("=" * 60)

    fs = 48_000
    duracion = 0.5         # segundos de tono de prueba
    n_total  = int(fs * duracion)
    t        = np.arange(n_total) / fs
    señal    = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tam_bloque = 512

    # ── TEST AM ──────────────────────────────────────────────────
    print("\n[1] TEST AM — modulación y demodulación por bloques")
    bloques = [señal[i:i+tam_bloque] for i in range(0, n_total, tam_bloque)]
    am_bloques, dm_bloques = [], []
    offset_tx, offset_rx = 0, 0

    for b in bloques:
        bam, offset_tx = modular_am_bloque(b, sample_offset=offset_tx)
        am_bloques.append(bam)
    for bam in am_bloques:
        bdm, offset_rx = demodular_am_bloque(bam, sample_offset=offset_rx)
        dm_bloques.append(bdm)

    am_total = np.concatenate(am_bloques)
    dm_total = np.concatenate(dm_bloques)
    snr_am = calcular_snr(señal, dm_total)
    print(f"    SNR AM   : {snr_am['SNR_dB']:+.1f} dB | MSE: {snr_am['MSE']:.6f}")
    assert snr_am['SNR_dB'] > 7, "AM SNR demasiado bajo"
    print("    ✅ AM OK")

    # ── TEST FM ──────────────────────────────────────────────────
    print("\n[2] TEST FM — modulación con propagación de fase + demodulación")
    fm_bloques, dfm_bloques = [], []
    estado = EstadoFM()

    for b in bloques:
        bfm, estado = modular_fm_bloque(b, estado=estado)
        fm_bloques.append(bfm)
    for bfm in fm_bloques:
        bdfm = demodular_fm_bloque(bfm)
        dfm_bloques.append(bdfm)

    fm_total  = np.concatenate(fm_bloques)
    dfm_total = np.concatenate(dfm_bloques)
    snr_fm = calcular_snr(señal, dfm_total)
    print(f"    SNR FM   : {snr_fm['SNR_dB']:+.1f} dB | MSE: {snr_fm['MSE']:.6f}")
    assert snr_fm['SNR_dB'] > 6, "FM SNR demasiado bajo"
    print("    ✅ FM OK")

    # ── TEST SERIALIZACIÓN ───────────────────────────────────────
    print("\n[3] TEST SERIALIZACIÓN — float32 ↔ int16")
    payload = serializar_bloque(bloques[0])
    recuperado = deserializar_bloque(payload)
    err_max = float(np.max(np.abs(bloques[0] - recuperado)))
    print(f"    Error cuantización: {err_max:.6f} (< 1/32767 ≈ 0.0000305)")
    assert err_max < 0.0001, "Error de cuantización inesperadamente alto"
    print("    ✅ Serialización OK")

    print("\n" + "=" * 60)
    print("  ✅ TODOS LOS TESTS PASADOS")
    print("=" * 60)
