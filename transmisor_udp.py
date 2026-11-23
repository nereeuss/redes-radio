"""
=============================================================
 TRANSMISOR UDP — Emisor de Radio SDR por Red
=============================================================

Proceso completo:
  1. Carga la señal de voz (WAV o tono sintético)
  2. Modula en FM (o AM si se pide)
  3. Opcionalmente aplica la máscara musical (cifrado acústico)
  4. Trocea en bloques de 10 ms (480 muestras)
  5. Empaqueta cada bloque con cabecera binaria de 22 bytes
  6. Envía cada paquete por socket UDP en tiempo real

Uso:
  python transmisor_udp.py                           # FM, sin máscara, localhost
  python transmisor_udp.py --mod AM                  # Transmitir en AM
  python transmisor_udp.py --mask                    # Activar cifrado musical
  python transmisor_udp.py --ip 192.168.1.20         # Enviar a otro PC
  python transmisor_udp.py --ip 192.168.1.20 --mask  # Cifrado a otro PC

Autor: Equipo SDR — Fase 3
"""

import socket
import time
import sys
import os
import argparse
import numpy as np

# Importar nuestros módulos de fases anteriores
from protocolo_sdr import (
    empaquetar, trocear_señal, cargar_mascara_musical, aplicar_mascara,
    SAMPLES_PER_BLOCK, BLOCK_MS, FS, FC_DEFAULT,
    MOD_AM, MOD_FM, MASK_OFF, MASK_ON,
    DEFAULT_IP, DEFAULT_PORT
)

# Importar moduladores de la Fase 2
from fase2_modulacion import modular_am, modular_fm


# ────────────────────────────────────────────────────────────
# CARGA DE AUDIO
# ────────────────────────────────────────────────────────────

def cargar_audio(ruta: str) -> np.ndarray:
    """
    Carga el archivo WAV procesado en la Fase 1.
    Si no existe, genera un tono de prueba.
    """
    if os.path.isfile(ruta):
        import wave
        with wave.open(ruta, 'rb') as wf:
            raw = wf.readframes(wf.getnframes())
            ancho = wf.getsampwidth()
            n_ch = wf.getnchannels()
        tipos = {1: np.int8, 2: np.int16, 4: np.int32}
        muestras = np.frombuffer(raw, dtype=tipos.get(ancho, np.int16)).astype(np.float32)
        if n_ch == 2:
            muestras = muestras.reshape(-1, 2).mean(axis=1)
        muestras /= np.iinfo(tipos.get(ancho, np.int16)).max
        muestras -= np.mean(muestras)
        pico = np.max(np.abs(muestras))
        if pico > 0:
            muestras = muestras / pico * 0.95
        print(f"[OK] Audio cargado: '{ruta}' ({len(muestras)} muestras, "
              f"{len(muestras)/FS:.1f} s)")
        return muestras.astype(np.float32)
    else:
        print(f"[INFO] '{ruta}' no encontrado. Generando tono de prueba...")
        duracion = 5.0
        t = np.linspace(0, duracion, int(duracion * FS), endpoint=False)
        sig = (0.7 * np.sin(2 * np.pi * 300 * t) +
               0.4 * np.sin(2 * np.pi * 600 * t) +
               0.3 * np.sin(2 * np.pi * 1200 * t))
        sig -= np.mean(sig)
        return (sig / np.max(np.abs(sig)) * 0.95).astype(np.float32)


# ────────────────────────────────────────────────────────────
# TRANSMISOR PRINCIPAL
# ────────────────────────────────────────────────────────────

def transmitir(ip: str, puerto: int,
               ruta_audio: str,
               mod: int, fc: int,
               usar_mascara: bool,
               ruta_musica: str = None):
    """
    Función principal del transmisor.

    Abre un socket UDP, modula la señal, la trocea en bloques
    y envía cada paquete con un temporizador de 10 ms para
    simular transmisión en tiempo real.
    """
    print("\n" + "=" * 60)
    print("  TRANSMISOR SDR — Radio por UDP")
    print("=" * 60)

    # ── Paso 1: Cargar audio ──────────────────────────────────
    voz = cargar_audio(ruta_audio)

    # ── Paso 2: Modular ───────────────────────────────────────
    tipo_mod = "FM" if mod == MOD_FM else "AM"
    print(f"\n[1/5] Modulando en {tipo_mod} (fc = {fc} Hz)...")

    if mod == MOD_FM:
        señal_modulada = modular_fm(voz, fs=FS, fc=float(fc), kf=3500.0)
    else:
        señal_modulada = modular_am(voz, fs=FS, fc=float(fc), ka=0.9)

    print(f"      Señal modulada: {len(señal_modulada)} muestras")

    # ── Paso 3: Preparar máscara musical (si activada) ────────
    mascara = None
    mask_flag = MASK_OFF

    if usar_mascara:
        print(f"\n[2/5] Generando máscara musical (cifrado acústico)...")
        duracion_audio = len(señal_modulada) / FS
        mascara = cargar_mascara_musical(duracion_audio + 1.0, fs=FS,
                                          ruta_wav=ruta_musica)
        mask_flag = MASK_ON
        print(f"      Máscara: {len(mascara)} muestras ({len(mascara)/FS:.1f} s)")
        print(f"      🔒 MODO ESPÍA ACTIVADO: La voz se camuflará en la música")
    else:
        print(f"\n[2/5] Máscara musical: DESACTIVADA (transmisión directa)")

    # ── Paso 4: Trocear en bloques ────────────────────────────
    print(f"\n[3/5] Troceando en bloques de {BLOCK_MS} ms "
          f"({SAMPLES_PER_BLOCK} muestras)...")
    bloques = trocear_señal(señal_modulada, SAMPLES_PER_BLOCK)
    print(f"      Total: {len(bloques)} paquetes a transmitir")

    # Calcular tamaño estimado por paquete
    paquete_ejemplo = empaquetar(bloques[0], 0, mod, fc, mask_flag)
    bytes_por_paquete = len(paquete_ejemplo)
    bytes_totales = bytes_por_paquete * len(bloques)
    print(f"      Tamaño por paquete: {bytes_por_paquete} bytes "
          f"(cabecera 22 + payload {bytes_por_paquete - 22})")
    print(f"      Datos totales a transmitir: {bytes_totales / 1024:.1f} KB")

    # ── Paso 5: Abrir socket y transmitir ─────────────────────
    print(f"\n[4/5] Abriendo socket UDP → {ip}:{puerto}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"\n[5/5] ¡TRANSMITIENDO EN DIRECTO!")
    print(f"      Destino  : {ip}:{puerto}")
    print(f"      Paquetes : {len(bloques)}")
    print(f"      Duración : {len(señal_modulada)/FS:.1f} segundos")
    print(f"      Intervalo: {BLOCK_MS} ms entre paquetes")
    print(f"      Máscara  : {'🔒 ACTIVADA' if usar_mascara else '🔓 DESACTIVADA'}")
    print()

    t_inicio = time.time()
    bytes_enviados = 0

    try:
        for i, bloque in enumerate(bloques):
            t_paquete_inicio = time.time()

            # Aplicar máscara si está activada
            if mascara is not None:
                bloque = aplicar_mascara(bloque, mascara, n_seq=i,
                                         bloque=SAMPLES_PER_BLOCK)

            # Empaquetar
            datagrama = empaquetar(bloque, n_seq=i, mod=mod,
                                    fc=fc, mask=mask_flag)

            # Enviar por UDP
            sock.sendto(datagrama, (ip, puerto))
            bytes_enviados += len(datagrama)

            # Barra de progreso
            progreso = (i + 1) / len(bloques) * 100
            barra = "█" * int(progreso // 5) + "░" * (20 - int(progreso // 5))
            estado_mask = "🔒" if usar_mascara else "📡"
            sys.stdout.write(
                f"\r  {estado_mask} [{barra}] {progreso:5.1f}%  "
                f"Paquete {i+1:>5}/{len(bloques)}  "
                f"({bytes_enviados/1024:.0f} KB)"
            )
            sys.stdout.flush()

            # Esperar para simular tiempo real
            t_transcurrido = time.time() - t_paquete_inicio
            t_espera = max(0, BLOCK_MS / 1000.0 - t_transcurrido)
            time.sleep(t_espera)

    except KeyboardInterrupt:
        print("\n\n[!] Transmisión interrumpida por el usuario.")
    finally:
        sock.close()

    t_total = time.time() - t_inicio

    print(f"\n\n{'=' * 60}")
    print(f"  ✅ TRANSMISIÓN COMPLETADA")
    print(f"{'=' * 60}")
    print(f"  Paquetes enviados  : {len(bloques)}")
    print(f"  Bytes transmitidos : {bytes_enviados / 1024:.1f} KB")
    print(f"  Tiempo real        : {t_total:.2f} s")
    print(f"  Tasa de datos      : {bytes_enviados * 8 / t_total / 1000:.1f} kbps")
    print(f"  Modulación         : {tipo_mod} (fc = {fc} Hz)")
    print(f"  Máscara musical    : {'🔒 ACTIVADA' if usar_mascara else '🔓 DESACTIVADA'}")
    print(f"{'=' * 60}")


# ────────────────────────────────────────────────────────────
# BLOQUE PRINCIPAL
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transmisor SDR — Envía voz modulada por UDP"
    )
    parser.add_argument("--ip", default=DEFAULT_IP,
                        help=f"IP de destino (por defecto {DEFAULT_IP})")
    parser.add_argument("--puerto", type=int, default=DEFAULT_PORT,
                        help=f"Puerto UDP (por defecto {DEFAULT_PORT})")
    parser.add_argument("--entrada", default="audio_original.wav",
                        help="Archivo WAV de entrada")
    parser.add_argument("--mod", choices=["AM", "FM"], default="FM",
                        help="Tipo de modulación (por defecto FM)")
    parser.add_argument("--fc", type=int, default=FC_DEFAULT,
                        help=f"Frecuencia portadora Hz (por defecto {FC_DEFAULT})")
    parser.add_argument("--mask", action="store_true",
                        help="Activar cifrado con máscara musical")
    parser.add_argument("--musica", default=None,
                        help="WAV de música para la máscara (opcional)")
    args = parser.parse_args()

    mod = MOD_FM if args.mod == "FM" else MOD_AM

    transmitir(
        ip=args.ip,
        puerto=args.puerto,
        ruta_audio=args.entrada,
        mod=mod,
        fc=args.fc,
        usar_mascara=args.mask,
        ruta_musica=args.musica
    )
