"""
=============================================================
 RECEPTOR UDP — Receptor de Radio SDR por Red
=============================================================

Proceso completo:
  1. Abre un socket UDP escuchando en el puerto configurado
  2. Recibe datagramas y desempaqueta la cabecera binaria
  3. Si la máscara está activada, retira la música para descifrar
  4. Reensambla los bloques de audio en orden de secuencia
  5. Demodula la señal recibida (FM o AM según la cabecera)
  6. Guarda el audio y muestra las métricas de red

Uso:
  python receptor_udp.py                    # Escuchar en localhost:5005
  python receptor_udp.py --puerto 6000      # Puerto diferente
  python receptor_udp.py --mask             # Descifrar máscara musical
  python receptor_udp.py --play             # Reproducir audio al terminar

Autor: Equipo SDR — Fase 3
"""

import socket
import time
import sys
import os
import wave
import argparse
import numpy as np

from protocolo_sdr import (
    desempaquetar, cargar_mascara_musical, retirar_mascara,
    MetricasRed,
    SAMPLES_PER_BLOCK, BLOCK_MS, FS, FC_DEFAULT,
    MOD_AM, MOD_FM, MASK_ON,
    DEFAULT_IP, DEFAULT_PORT, HEADER_SIZE
)

from scipy.signal import butter, filtfilt, hilbert

try:
    import sounddevice as sd
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False


# ────────────────────────────────────────────────────────────
# DEMODULADORES (Reutilizados de Fase 2, adaptados)
# ────────────────────────────────────────────────────────────

def demodular_am_bloque(señal, fs=FS, fc=8000.0):
    """Demodula un bloque completo de señal AM."""
    t = np.arange(len(señal)) / fs
    mezclado = señal * np.cos(2 * np.pi * fc * t)
    b, a = butter(5, 3800.0 / (fs / 2), btype='low')
    demod = filtfilt(b, a, mezclado)
    demod -= np.mean(demod)
    umbral = np.percentile(np.abs(demod), 99.8)
    if umbral > 0:
        demod = np.clip(demod / umbral * 0.95, -0.95, 0.95)
    return demod.astype(np.float32)


def demodular_fm_bloque(señal, fs=FS, fc=8000.0, kf=3500.0):
    """Demodula un bloque completo de señal FM."""
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


# ────────────────────────────────────────────────────────────
# GUARDAR AUDIO
# ────────────────────────────────────────────────────────────

def guardar_wav(señal: np.ndarray, fs: int, ruta: str):
    """Guarda la señal como archivo WAV int16."""
    int16 = (np.clip(señal, -1, 1) * 32767).astype(np.int16)
    with wave.open(ruta, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(int16.tobytes())
    print(f"[OK] WAV guardado: '{ruta}'")


# ────────────────────────────────────────────────────────────
# RECEPTOR PRINCIPAL
# ────────────────────────────────────────────────────────────

def recibir(puerto: int,
            usar_mascara: bool,
            ruta_musica: str = None,
            timeout: float = 5.0,
            reproducir: bool = False):
    """
    Función principal del receptor.

    Escucha el puerto UDP, recibe todos los paquetes,
    los reensambla en orden, demodula y guarda el resultado.
    """
    print("\n" + "=" * 60)
    print("  RECEPTOR SDR — Escuchando Radio por UDP")
    print("=" * 60)

    # ── Preparar máscara si se va a usar ──────────────────────
    mascara = None
    if usar_mascara:
        print(f"\n[INFO] Preparando máscara musical para descifrado...")
        # Generamos una máscara larga; se recortará según necesidad
        mascara = cargar_mascara_musical(120.0, fs=FS, ruta_wav=ruta_musica)
        print(f"       🔓 MODO DESCIFRADO: Se retirará la música de la señal")
    else:
        print(f"\n[INFO] Máscara musical: DESACTIVADA")

    # ── Abrir socket UDP ──────────────────────────────────────
    print(f"\n[1/4] Abriendo socket UDP en puerto {puerto}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", puerto))
    sock.settimeout(timeout)

    print(f"       ✅ Escuchando en 0.0.0.0:{puerto}")
    print(f"       Timeout: {timeout} s (se cierra si no llegan más paquetes)\n")

    # ── Recibir paquetes ──────────────────────────────────────
    print(f"[2/4] Esperando paquetes...")
    print(f"       (Lanza el transmisor en otra terminal)\n")

    metricas = MetricasRed()
    paquetes_buffer = {}  # {seq: muestras}
    mod_detectado = None
    fc_detectado = FC_DEFAULT
    recibiendo = True

    try:
        while recibiendo:
            try:
                datos, addr = sock.recvfrom(65535)
            except socket.timeout:
                if metricas.paquetes_recibidos > 0:
                    print(f"\n\n[INFO] Timeout de {timeout}s sin paquetes. "
                          f"Fin de la recepción.")
                    recibiendo = False
                    continue
                else:
                    sys.stdout.write(f"\r  ⏳ Esperando primer paquete... "
                                     f"(timeout en {timeout}s)")
                    sys.stdout.flush()
                    continue

            try:
                paquete = desempaquetar(datos)
            except ValueError as e:
                print(f"\n  ⚠️ Paquete descartado: {e}")
                continue

            # Registrar métricas
            metricas.registrar_paquete(paquete, len(datos))

            # Detectar tipo de modulación del primer paquete
            if mod_detectado is None:
                mod_detectado = paquete['mod']
                fc_detectado = paquete['fc']
                tipo = "FM" if mod_detectado == MOD_FM else "AM"
                mask_txt = "🔒 SI" if paquete['mask'] == MASK_ON else "🔓 NO"
                print(f"\r  📡 Primer paquete recibido de {addr[0]}:{addr[1]}")
                print(f"      Modulación: {tipo} | Portadora: {fc_detectado} Hz "
                      f"| Máscara: {mask_txt}")
                print()

            # Retirar máscara si está activa
            muestras = paquete['muestras']
            if paquete['mask'] == MASK_ON and mascara is not None:
                muestras = retirar_mascara(muestras, mascara,
                                            n_seq=paquete['seq'],
                                            bloque=SAMPLES_PER_BLOCK)

            # Guardar en buffer ordenado por secuencia
            paquetes_buffer[paquete['seq']] = muestras

            # Barra de progreso
            n = metricas.paquetes_recibidos
            estado = "🔓" if paquete['mask'] == MASK_ON else "📡"
            sys.stdout.write(
                f"\r  {estado} Recibidos: {n:>5} paquetes | "
                f"Último seq: {paquete['seq']:>5} | "
                f"Desde: {addr[0]}:{addr[1]}  "
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[!] Recepción interrumpida por el usuario.")
    finally:
        sock.close()

    if not paquetes_buffer:
        print("\n[ERROR] No se recibió ningún paquete válido.")
        return

    # ── Reensamblar audio en orden ────────────────────────────
    print(f"\n\n[3/4] Reensamblando audio ({len(paquetes_buffer)} bloques)...")

    # Ordenar por número de secuencia
    seqs_ordenadas = sorted(paquetes_buffer.keys())
    señal_completa = np.concatenate([paquetes_buffer[s] for s in seqs_ordenadas])
    print(f"       Señal reensamblada: {len(señal_completa)} muestras "
          f"({len(señal_completa)/FS:.1f} s)")

    # Detectar huecos
    for i in range(1, len(seqs_ordenadas)):
        if seqs_ordenadas[i] != seqs_ordenadas[i-1] + 1:
            perdidos = seqs_ordenadas[i] - seqs_ordenadas[i-1] - 1
            print(f"       ⚠️ Hueco detectado: paquetes "
                  f"{seqs_ordenadas[i-1]+1}..{seqs_ordenadas[i]-1} "
                  f"({perdidos} perdidos)")

    # ── Demodular ─────────────────────────────────────────────
    print(f"\n[4/4] Demodulando señal recibida...")
    tipo = "FM" if mod_detectado == MOD_FM else "AM"

    if mod_detectado == MOD_FM:
        audio_recuperado = demodular_fm_bloque(señal_completa,
                                                fs=FS, fc=float(fc_detectado))
    else:
        audio_recuperado = demodular_am_bloque(señal_completa,
                                                fs=FS, fc=float(fc_detectado))

    print(f"       Demodulación {tipo} completada. "
          f"Audio: {len(audio_recuperado)} muestras")

    # ── Guardar resultado ─────────────────────────────────────
    guardar_wav(señal_completa, FS, "audio_recibido_modulado.wav")
    guardar_wav(audio_recuperado, FS, "audio_recibido_demodulado.wav")

    # ── Mostrar métricas ──────────────────────────────────────
    print(f"\n{metricas}")

    # ── Reproducir si se pidió ────────────────────────────────
    if reproducir and AUDIO_OK:
        print(f"\n[▶] Reproduciendo audio demodulado...")
        sd.play(audio_recuperado, FS)
        sd.wait()
        print(f"[⏹] Reproducción terminada.")

    print(f"\n{'=' * 60}")
    print(f"  ✅ RECEPCIÓN COMPLETADA")
    print(f"{'=' * 60}")

    return audio_recuperado, metricas


# ────────────────────────────────────────────────────────────
# BLOQUE PRINCIPAL
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Receptor SDR — Recibe voz modulada por UDP"
    )
    parser.add_argument("--puerto", type=int, default=DEFAULT_PORT,
                        help=f"Puerto UDP (por defecto {DEFAULT_PORT})")
    parser.add_argument("--mask", action="store_true",
                        help="Activar descifrado de máscara musical")
    parser.add_argument("--musica", default=None,
                        help="WAV de música para descifrar (opcional)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Segundos sin paquetes antes de cerrar (por defecto 5)")
    parser.add_argument("--play", action="store_true",
                        help="Reproducir audio al terminar")
    args = parser.parse_args()

    recibir(
        puerto=args.puerto,
        usar_mascara=args.mask,
        ruta_musica=args.musica,
        timeout=args.timeout,
        reproducir=args.play
    )
