"""
=============================================================
 🕵️ INTERCEPTOR / SNIFFER UDP — Espía de Red SDR
=============================================================

¿Qué hace este script?
  Simula a un atacante no autorizado que está escuchando en la
  red local y capturando los paquetes UDP de la radio.

Demostración del Cifrado por Esteganografía:
  - Si el emisor transmitió con MÁSCARA ACTIVADA:
    El espía captura los paquetes pero NO conoce la clave
    musical. Al demodular lo que captura en la red, SOLO
    SE ESCUCHA MÚSICA (la canción de 8 bits). La voz humana
    queda totalmente indetectable e ininteligible.

  - Demuestra al profesor de forma práctica la utilidad del
    camuflaje acústico implementado en el protocolo SDR.

Uso:
  python interceptor_udp.py           # Escuchar en 0.0.0.0:5005
  python interceptor_udp.py --play    # Reproducir lo que captura el espía

Autor: Equipo SDR — Demostración de Intercepción
"""

import socket
import sys
import time
import argparse
import wave
import numpy as np


# Ruta al directorio de modulos (funciona tanto si se ejecuta desde la raiz como desde modulos/)
import os as _os, sys as _sys
_dir = _os.path.dirname(_os.path.abspath(__file__))
if _dir not in _sys.path:
    _sys.path.insert(0, _dir)

from protocolo_sdr import (
    desempaquetar, MetricasRed,
    SAMPLES_PER_BLOCK, FS, FC_DEFAULT,
    MOD_AM, MOD_FM, MASK_ON, MASK_OFF,
    DEFAULT_PORT
)

from scipy.signal import butter, filtfilt, hilbert

try:
    import sounddevice as sd
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False


def demodular_fm_espia(señal, fs=FS, fc=8000.0, kf=3500.0):
    """Demodula la señal capturada tal cual, sin retirar la máscara."""
    analitica = hilbert(señal)
    fase_inst = np.unwrap(np.angle(analitica))
    freq_inst = np.diff(fase_inst) * fs / (2 * np.pi)
    demod = (freq_inst - fc) / kf
    demod = np.append(demod, 0.0)
    b, a = butter(4, 3200.0 / (fs / 2), btype='low')
    demod = filtfilt(b, a, demod)
    demod -= np.mean(demod)
    umbral = np.percentile(np.abs(demod), 99.8)
    if umbral > 0:
        demod = np.clip(demod / umbral * 0.95, -0.95, 0.95)
    return demod.astype(np.float32)


def demodular_am_espia(señal, fs=FS, fc=8000.0):
    t = np.arange(len(señal)) / fs
    mezclado = señal * np.cos(2 * np.pi * fc * t)
    b, a = butter(5, 3800.0 / (fs / 2), btype='low')
    demod = filtfilt(b, a, mezclado)
    demod -= np.mean(demod)
    umbral = np.percentile(np.abs(demod), 99.8)
    if umbral > 0:
        demod = np.clip(demod / umbral * 0.95, -0.95, 0.95)
    return demod.astype(np.float32)


def guardar_wav(señal: np.ndarray, fs: int, ruta: str):
    int16 = (np.clip(señal, -1, 1) * 32767).astype(np.int16)
    with wave.open(ruta, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(int16.tobytes())
    print(f"[OK] Audio capturado guardado en: '{ruta}'")


def interceptar(puerto: int, timeout: float = 6.0, reproducir: bool = False):
    print("\n" + "=" * 62)
    print(" 🕵️  INTERCEPTOR / SNIFFER UDP — Escuchando Tráfico de Red")
    print("=" * 62)
    print(f"[*] Abriendo socket espía en 0.0.0.0:{puerto}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.bind(("0.0.0.0", puerto))
    except Exception as e:
        print(f"[ERROR] No se pudo vincular el puerto {puerto}: {e}")
        return

    sock.settimeout(timeout)

    print(f"[*] Escuchando datagramas UDP en tránsito...")
    print(f"[*] (El espía NO posee la clave de la máscara musical)\n")

    paquetes_capturados = {}
    mod_detectado = None
    fc_detectado = FC_DEFAULT
    mascara_detectada = False
    metricas = MetricasRed()

    recibiendo = True
    try:
        while recibiendo:
            try:
                datos, addr = sock.recvfrom(65535)
            except socket.timeout:
                if metricas.paquetes_recibidos > 0:
                    print(f"\n\n[INFO] Fin de la ráfaga de paquetes capturados.")
                    recibiendo = False
                    continue
                else:
                    sys.stdout.write(f"\r  ⏳ Esperando tráfico en la red UDP... ")
                    sys.stdout.flush()
                    continue

            try:
                paquete = desempaquetar(datos)
            except ValueError:
                continue

            metricas.registrar_paquete(paquete, len(datos))

            if mod_detectado is None:
                mod_detectado = paquete['mod']
                fc_detectado = paquete['fc']
                mascara_detectada = (paquete['mask'] == MASK_ON)

            # ¡EL ESPÍA GUARDA LA SEÑAL BRUTA SIN RETIRAR LA MÁSCARA!
            paquetes_capturados[paquete['seq']] = paquete['muestras']

            n = metricas.paquetes_recibidos
            tipo_mask = "🔒 CIFRADO (MÁSCARA)" if paquete['mask'] == MASK_ON else "📡 SIN CIFRAR"
            sys.stdout.write(
                f"\r  🕵️ Interceptados: {n:>4} paquetes | "
                f"Seq: #{paquete['seq']:>4} | "
                f"Desde: {addr[0]}:{addr[1]} | {tipo_mask}   "
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[!] Intercepción detenida por el usuario.")
    finally:
        sock.close()

    if not paquetes_capturados:
        print("\n[!] No se capturó ningún datagrama.")
        return

    print("\n\n" + "-" * 62)
    print(" 📊 RESULTADO DEL ANÁLISIS DE TRÁFICO CAPTURADO")
    print("-" * 62)

    seqs_ordenadas = sorted(paquetes_capturados.keys())
    señal_bruta = np.concatenate([paquetes_capturados[s] for s in seqs_ordenadas])

    print(f"• Paquetes interceptados : {len(paquetes_capturados)}")
    print(f"• Modulación detectada   : {'FM' if mod_detectado == MOD_FM else 'AM'} (fc = {fc_detectado} Hz)")
    print(f"• Cifrado esteganográfico: {'🔒 ACTIVADO' if mascara_detectada else '🔓 DESACTIVADO'}")

    # Demodular lo que capturó el espía
    print(f"\n[*] Demodulando señal capturada (sin clave)...")
    if mod_detectado == MOD_FM:
        audio_espia = demodular_fm_espia(señal_bruta, fs=FS, fc=float(fc_detectado))
    else:
        audio_espia = demodular_am_espia(señal_bruta, fs=FS, fc=float(fc_detectado))

    # Guardar
    guardar_wav(audio_espia, FS, "audio_interceptado_espia.wav")

    print("\n" + "=" * 62)
    if mascara_detectada:
        print(" 🔒 ¡DEMOSTRACIÓN DE ESTEGANOGRAFÍA EXITOSA!")
        print("   El emisor utilizó una máscara musical.")
        print("   Como el espía no tiene la clave para restarla,")
        print("   ¡AL REPRODUCIR SOLO SE ESCUCHA MÚSICA!")
        print("   La conversación de voz humana ha quedado 100% protegida.")
    else:
        print(" ⚠️ TRÁFICO EN CLARO:")
        print("   La transmisión no tenía máscara activa.")
        print("   El espía ha podido escuchar la voz sin protección.")
    print("=" * 62)

    if reproducir and AUDIO_OK:
        print(f"\n[▶] Reproduciendo audio tal como lo escucha el espía...")
        sd.play(audio_espia, FS)
        sd.wait()
        print(f"[⏹] Fin de la reproducción.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interceptor / Sniffer UDP SDR")
    parser.add_argument("--puerto", type=int, default=DEFAULT_PORT, help="Puerto a escuchar")
    parser.add_argument("--play", action="store_true", help="Reproducir audio interceptado al finalizar")
    args = parser.parse_args()

    interceptar(puerto=args.puerto, reproducir=args.play)
