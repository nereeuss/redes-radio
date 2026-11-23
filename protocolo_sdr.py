"""
=============================================================
 PROTOCOLO SDR — Empaquetado y Desempaquetado de Paquetes
=============================================================

Define el formato binario de cada datagrama UDP:

  ┌──────────┬──────────┬──────────┬────────┬──────────┬────────┬──────────┬───────────┐
  │  MAGIC   │ Nº SEQ   │TIMESTAMP │  MOD   │ PORTADORA│  MASK  │ N_MUESTR │  PAYLOAD  │
  │  4 bytes │ 4 bytes  │ 8 bytes  │ 1 byte │ 2 bytes  │ 1 byte │ 2 bytes  │ variable  │
  │  "SDR1"  │ uint32   │ float64  │ uint8  │ uint16   │ uint8  │ uint16   │ int16[]   │
  └──────────┴──────────┴──────────┴────────┴──────────┴────────┴──────────┴───────────┘

  Cabecera total: 22 bytes
  Payload: N_MUESTRAS × 2 bytes (int16)

  MAGIC       → Identifica paquetes de nuestra app (descarta tráfico ajeno)
  Nº SEQ      → Detecta pérdidas y reordena paquetes
  TIMESTAMP   → Mide retardo y jitter de la red
  MOD         → 0 = AM, 1 = FM
  PORTADORA   → Frecuencia portadora en Hz (ej: 8000)
  MASK        → 0 = sin máscara, 1 = enmascarado con música
  N_MUESTRAS  → Cuántas muestras de audio tiene este paquete
  PAYLOAD     → Las muestras codificadas en int16 Big-Endian

Formato struct (Big-Endian / Network byte order):
  "!" = network byte order (Big-Endian)
  "4s" = 4 chars (MAGIC)
  "I"  = uint32 (secuencia)
  "d"  = float64 (timestamp)
  "B"  = uint8 (modulación)
  "H"  = uint16 (portadora)
  "B"  = uint8 (máscara)
  "H"  = uint16 (n_muestras)

Autor: Equipo SDR — Fase 3
"""

import struct
import time
import numpy as np
import os

# ────────────────────────────────────────────────────────────
# CONSTANTES DEL PROTOCOLO
# ────────────────────────────────────────────────────────────

MAGIC = b"SDR1"                        # 4 bytes de identificación
HEADER_FORMAT = "!4sIdBHBH"            # Formato de la cabecera
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 22 bytes

# Tamaño de cada bloque de audio
BLOCK_MS = 10                          # Milisegundos por paquete (10 ms)
FS = 48_000                            # Frecuencia de muestreo
SAMPLES_PER_BLOCK = int(FS * BLOCK_MS / 1000)  # 480 muestras → 982 bytes < 1500 MTU

# Códigos de modulación
MOD_AM = 0
MOD_FM = 1

# Códigos de máscara
MASK_OFF = 0
MASK_ON = 1

# Puerto UDP por defecto
DEFAULT_PORT = 5005
DEFAULT_IP = "127.0.0.1"               # Localhost para pruebas locales

# Frecuencia portadora por defecto
FC_DEFAULT = 8000


# ────────────────────────────────────────────────────────────
# FUNCIÓN 1: Empaquetar un bloque de audio en un datagrama
# ────────────────────────────────────────────────────────────

def empaquetar(muestras: np.ndarray,
               n_seq: int,
               mod: int = MOD_FM,
               fc: int = FC_DEFAULT,
               mask: int = MASK_OFF) -> bytes:
    """
    Convierte un bloque de muestras float32 en un datagrama binario
    listo para enviar por socket UDP.

    Proceso:
      1. Escalar float32 [-1, 1] → int16 [-32767, 32767]
      2. Construir cabecera de 22 bytes con struct.pack
      3. Concatenar cabecera + payload binario

    Parámetros
    ----------
    muestras : np.ndarray float32 con valores en [-1, 1]
    n_seq    : número de secuencia del paquete (0, 1, 2, ...)
    mod      : tipo de modulación (0=AM, 1=FM)
    fc       : frecuencia portadora en Hz
    mask     : 0=sin máscara, 1=enmascarado con música

    Retorna
    -------
    bytes : datagrama completo (cabecera + payload)
    """
    # Paso 1: Convertir float32 a int16 (codificación PCM)
    # Multiplicamos por 32767 para usar todo el rango dinámico de 16 bits
    int16_data = (np.clip(muestras, -1.0, 1.0) * 32767).astype(np.int16)

    # Paso 2: Construir la cabecera binaria
    # time.time() devuelve los segundos desde epoch con precisión de microsegundos
    timestamp = time.time()
    n_muestras = len(int16_data)

    cabecera = struct.pack(
        HEADER_FORMAT,
        MAGIC,              # 4s: identificador "SDR1"
        n_seq,              # I:  número de secuencia (uint32)
        timestamp,          # d:  timestamp (float64, 8 bytes)
        mod,                # B:  modulación (uint8: 0=AM, 1=FM)
        fc,                 # H:  portadora (uint16)
        mask,               # B:  máscara musical (uint8: 0=off, 1=on)
        n_muestras          # H:  nº de muestras en el payload (uint16)
    )

    # Paso 3: Serializar las muestras a bytes en Big-Endian
    # ">" = Big-Endian (estándar de red), "h" = int16
    payload = struct.pack(f">{n_muestras}h", *int16_data)

    return cabecera + payload


# ────────────────────────────────────────────────────────────
# FUNCIÓN 2: Desempaquetar un datagrama recibido
# ────────────────────────────────────────────────────────────

def desempaquetar(datagrama: bytes) -> dict:
    """
    Decodifica un datagrama binario recibido por socket UDP.

    Proceso:
      1. Extraer cabecera de 22 bytes con struct.unpack
      2. Verificar MAGIC == "SDR1"
      3. Decodificar payload int16 → float32 [-1, 1]

    Retorna
    -------
    dict con claves:
      'magic', 'seq', 'timestamp', 'mod', 'fc', 'mask',
      'n_muestras', 'muestras' (np.ndarray float32)

    Lanza ValueError si el MAGIC no coincide.
    """
    if len(datagrama) < HEADER_SIZE:
        raise ValueError(f"Datagrama demasiado corto: {len(datagrama)} bytes "
                         f"(mínimo {HEADER_SIZE})")

    # Paso 1: Desempaquetar la cabecera
    magic, n_seq, timestamp, mod, fc, mask, n_muestras = struct.unpack(
        HEADER_FORMAT,
        datagrama[:HEADER_SIZE]
    )

    # Paso 2: Verificar que es un paquete nuestro
    if magic != MAGIC:
        raise ValueError(f"MAGIC incorrecto: {magic!r} (esperado {MAGIC!r}). "
                         f"Este paquete no pertenece a nuestra radio.")

    # Paso 3: Decodificar el payload
    payload_bytes = datagrama[HEADER_SIZE:]
    expected_size = n_muestras * 2  # int16 = 2 bytes por muestra

    if len(payload_bytes) < expected_size:
        raise ValueError(f"Payload incompleto: {len(payload_bytes)} bytes "
                         f"(esperados {expected_size})")

    int16_data = struct.unpack(f">{n_muestras}h", payload_bytes[:expected_size])
    muestras = np.array(int16_data, dtype=np.float32) / 32767.0

    return {
        'magic': magic,
        'seq': n_seq,
        'timestamp': timestamp,
        'mod': mod,
        'fc': fc,
        'mask': mask,
        'n_muestras': n_muestras,
        'muestras': muestras
    }


# ────────────────────────────────────────────────────────────
# FUNCIÓN 3: Trocear señal completa en bloques de 20 ms
# ────────────────────────────────────────────────────────────

def trocear_señal(señal: np.ndarray,
                  bloque: int = SAMPLES_PER_BLOCK) -> list:
    """
    Divide un array largo de audio en bloques de tamaño fijo.

    ¿Por qué 20 ms (960 muestras)?
      - 960 muestras × 2 bytes = 1920 bytes de payload
      - Con cabecera de 22 bytes = 1942 bytes total
      - MTU de Ethernet = 1500 bytes...
      - Usamos 480 muestras (10 ms) = 960 + 22 = 982 bytes ✓
      
    Ajustamos a 480 muestras (10 ms) para caber en MTU holgadamente.

    El último bloque se rellena con ceros si no llega a completarse
    (zero-padding, estándar en procesado de señal).
    """
    bloques = []
    for i in range(0, len(señal), bloque):
        fragmento = señal[i:i + bloque]
        # Si el último bloque es más corto, rellenamos con ceros
        if len(fragmento) < bloque:
            fragmento = np.pad(fragmento, (0, bloque - len(fragmento)),
                               mode='constant', constant_values=0.0)
        bloques.append(fragmento.astype(np.float32))
    return bloques


# ────────────────────────────────────────────────────────────
# FUNCIÓN 4: Generar máscara musical (melodía sintética)
# ────────────────────────────────────────────────────────────

def cargar_mascara_musical(duracion_s: float, fs: int = FS,
                           ruta_wav: str = None) -> np.ndarray:
    """
    Carga o genera la máscara musical para el cifrado acústico.

    Si existe un archivo WAV de música, lo usa.
    Si no, genera una melodía sintética con acordes y ritmo
    que suena como una canción simple de 8 bits.

    La máscara se normaliza a amplitud 0.5 para dejar espacio
    a la voz modulada (0.5 voz + 0.5 música = 1.0 máximo).
    """
    if ruta_wav and os.path.exists(ruta_wav):
        import wave
        with wave.open(ruta_wav, 'rb') as wf:
            raw = wf.readframes(wf.getnframes())
            ancho = wf.getsampwidth()
        tipos = {1: np.int8, 2: np.int16, 4: np.int32}
        musica = np.frombuffer(raw, dtype=tipos.get(ancho, np.int16)).astype(np.float32)
        musica /= np.iinfo(tipos.get(ancho, np.int16)).max
    else:
        # Generar melodía sintética (acordes de Do mayor progresivos)
        n_total = int(duracion_s * fs)
        t = np.linspace(0, duracion_s, n_total, endpoint=False)

        # Notas musicales en Hz (escala de Do mayor)
        notas = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        # Progresión de acordes: Do-Mi-Sol, Re-Fa-La, Mi-Sol-Si, Do-Mi-Sol
        acordes = [
            (261.63, 329.63, 392.00),  # Do mayor
            (293.66, 349.23, 440.00),  # Re menor
            (329.63, 392.00, 493.88),  # Mi menor
            (349.23, 440.00, 523.25),  # Fa mayor
        ]

        musica = np.zeros(n_total, dtype=np.float32)
        duracion_acorde = n_total // len(acordes)

        for i, (f1, f2, f3) in enumerate(acordes):
            inicio = i * duracion_acorde
            fin = min(inicio + duracion_acorde, n_total)
            seg_t = t[inicio:fin]

            # Acorde con 3 tonos + un bajo + percusión rítmica
            acorde = (0.3 * np.sin(2 * np.pi * f1 * seg_t) +
                      0.25 * np.sin(2 * np.pi * f2 * seg_t) +
                      0.2 * np.sin(2 * np.pi * f3 * seg_t) +
                      0.15 * np.sin(2 * np.pi * f1 / 2 * seg_t))  # bajo una octava abajo

            # Envolvente de volumen (sube rápido, baja lento = tipo piano)
            n_seg = len(seg_t)
            envolvente = np.exp(-np.linspace(0, 3, n_seg))
            envolvente[:int(0.02 * fs)] *= np.linspace(0, 1, int(0.02 * fs))

            musica[inicio:fin] = acorde * envolvente

    # Normalizar a amplitud 0.5 (deja margen para sumar la voz)
    musica -= np.mean(musica)
    pico = np.max(np.abs(musica))
    if pico > 0:
        musica = musica / pico * 0.5

    # Si la máscara es más corta que la duración pedida, hacer loop
    n_necesario = int(duracion_s * fs)
    if len(musica) < n_necesario:
        repeticiones = (n_necesario // len(musica)) + 1
        musica = np.tile(musica, repeticiones)
    musica = musica[:n_necesario]

    return musica.astype(np.float32)


# ────────────────────────────────────────────────────────────
# FUNCIÓN 5: Aplicar y retirar la máscara musical
# ────────────────────────────────────────────────────────────

def aplicar_mascara(señal: np.ndarray,
                    mascara: np.ndarray,
                    n_seq: int,
                    bloque: int = SAMPLES_PER_BLOCK) -> np.ndarray:
    """
    Suma un fragmento de música a la señal para camuflarla.

    El fragmento de música se selecciona según el nº de secuencia:
      inicio = n_seq × bloque
      fin    = inicio + bloque

    Fórmula:
      señal_camuflada = 0.5 × señal + 0.5 × música[inicio:fin]

    Los coeficientes 0.5 evitan saturación (0.5 + 0.5 = 1.0 máximo).
    """
    inicio = (n_seq * bloque) % len(mascara)
    fin = inicio + len(señal)

    # Extraer el fragmento de música (con wrap-around por si se acaba)
    if fin <= len(mascara):
        fragmento_musica = mascara[inicio:fin]
    else:
        parte1 = mascara[inicio:]
        parte2 = mascara[:fin - len(mascara)]
        fragmento_musica = np.concatenate([parte1, parte2])

    # Sumar señal + música con coeficientes de mezcla
    camuflada = 0.5 * señal + 0.5 * fragmento_musica
    return np.clip(camuflada, -1.0, 1.0).astype(np.float32)


def retirar_mascara(señal_recibida: np.ndarray,
                    mascara: np.ndarray,
                    n_seq: int,
                    bloque: int = SAMPLES_PER_BLOCK) -> np.ndarray:
    """
    Resta el fragmento de música correspondiente para descifrar la voz.

    Fórmula inversa:
      señal_original = (señal_recibida - 0.5 × música[inicio:fin]) / 0.5
                     = 2 × señal_recibida - música[inicio:fin]

    La música se cancela por interferencia destructiva perfecta.
    """
    inicio = (n_seq * bloque) % len(mascara)
    fin = inicio + len(señal_recibida)

    if fin <= len(mascara):
        fragmento_musica = mascara[inicio:fin]
    else:
        parte1 = mascara[inicio:]
        parte2 = mascara[:fin - len(mascara)]
        fragmento_musica = np.concatenate([parte1, parte2])

    # Operación inversa exacta
    descifrada = 2.0 * señal_recibida - fragmento_musica
    return np.clip(descifrada, -1.0, 1.0).astype(np.float32)


# ────────────────────────────────────────────────────────────
# FUNCIÓN 6: Métricas de red
# ────────────────────────────────────────────────────────────

class MetricasRed:
    """
    Calcula estadísticas de transmisión en tiempo real:
      - Paquetes recibidos vs esperados
      - Pérdida de paquetes (%)
      - Retardo medio (ms)
      - Jitter medio (ms) — variación del retardo según RFC 3550
      - Tasa de transferencia (kbps)
    """
    def __init__(self):
        self.paquetes_recibidos = 0
        self.seq_max = -1
        self.seq_recibidos = set()
        self.retardos = []
        self.jitter = 0.0
        self.ultimo_retardo = None
        self.bytes_totales = 0
        self.tiempo_inicio = None

    def registrar_paquete(self, paquete: dict, tamaño_bytes: int):
        """Actualiza las métricas con un nuevo paquete recibido."""
        ahora = time.time()

        if self.tiempo_inicio is None:
            self.tiempo_inicio = ahora

        self.paquetes_recibidos += 1
        self.bytes_totales += tamaño_bytes

        seq = paquete['seq']
        self.seq_recibidos.add(seq)
        if seq > self.seq_max:
            self.seq_max = seq

        # Retardo: diferencia entre el reloj del receptor y el timestamp del emisor
        retardo = ahora - paquete['timestamp']
        self.retardos.append(retardo)

        # Jitter según RFC 3550 (media móvil exponencial)
        if self.ultimo_retardo is not None:
            diferencia = abs(retardo - self.ultimo_retardo)
            self.jitter += (diferencia - self.jitter) / 16.0
        self.ultimo_retardo = retardo

    def resumen(self) -> dict:
        """Devuelve un diccionario con todas las métricas calculadas."""
        esperados = self.seq_max + 1 if self.seq_max >= 0 else 0
        perdidos = esperados - len(self.seq_recibidos)
        pct_perdida = (perdidos / esperados * 100) if esperados > 0 else 0.0

        retardo_medio = np.mean(self.retardos) * 1000 if self.retardos else 0.0

        duracion = time.time() - self.tiempo_inicio if self.tiempo_inicio else 1.0
        kbps = (self.bytes_totales * 8 / 1000) / max(duracion, 0.001)

        return {
            'paquetes_recibidos': self.paquetes_recibidos,
            'paquetes_esperados': esperados,
            'paquetes_perdidos': perdidos,
            'perdida_pct': pct_perdida,
            'retardo_medio_ms': retardo_medio,
            'jitter_ms': self.jitter * 1000,
            'tasa_kbps': kbps,
        }

    def __str__(self):
        r = self.resumen()
        return (
            f"── Métricas de Red ──────────────────────────────\n"
            f"  Paquetes recibidos : {r['paquetes_recibidos']} / {r['paquetes_esperados']}\n"
            f"  Paquetes perdidos  : {r['paquetes_perdidos']} ({r['perdida_pct']:.1f}%)\n"
            f"  Retardo medio      : {r['retardo_medio_ms']:.2f} ms\n"
            f"  Jitter (RFC 3550)  : {r['jitter_ms']:.2f} ms\n"
            f"  Tasa de datos      : {r['tasa_kbps']:.1f} kbps\n"
            f"─────────────────────────────────────────────────"
        )


# ────────────────────────────────────────────────────────────
# TEST: Verificación rápida del protocolo
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  TEST DEL PROTOCOLO SDR")
    print("=" * 55)

    # Crear señal de prueba
    t = np.linspace(0, 0.01, SAMPLES_PER_BLOCK, endpoint=False)
    señal_test = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    print(f"\n[1] Señal de prueba: {len(señal_test)} muestras, "
          f"rango [{señal_test.min():.3f}, {señal_test.max():.3f}]")

    # Empaquetar
    paquete_bytes = empaquetar(señal_test, n_seq=42, mod=MOD_FM,
                               fc=8000, mask=MASK_OFF)
    print(f"[2] Paquete empaquetado: {len(paquete_bytes)} bytes "
          f"(cabecera {HEADER_SIZE} + payload {len(paquete_bytes) - HEADER_SIZE})")

    # Desempaquetar
    resultado = desempaquetar(paquete_bytes)
    print(f"[3] Desempaquetado:")
    print(f"    MAGIC       : {resultado['magic']}")
    print(f"    Secuencia   : {resultado['seq']}")
    print(f"    Modulación  : {'FM' if resultado['mod'] == 1 else 'AM'}")
    print(f"    Portadora   : {resultado['fc']} Hz")
    print(f"    Máscara     : {'ON' if resultado['mask'] else 'OFF'}")
    print(f"    N muestras  : {resultado['n_muestras']}")

    # Verificar integridad
    error = np.max(np.abs(señal_test - resultado['muestras']))
    print(f"[4] Error máximo tras empaquetar/desempaquetar: {error:.6f}")
    # El error viene del redondeo float32→int16→float32 (< 1/32767 ≈ 0.00003)
    assert error < 0.001, "ERROR: La señal no se recuperó correctamente"
    print(f"    ✅ Integridad verificada (error < 0.001)")

    # Test de máscara musical
    print(f"\n[5] Test de máscara musical:")
    mascara = cargar_mascara_musical(1.0)
    print(f"    Máscara generada: {len(mascara)} muestras")

    camuflada = aplicar_mascara(señal_test, mascara, n_seq=0)
    descifrada = retirar_mascara(camuflada, mascara, n_seq=0)
    error_mask = np.max(np.abs(señal_test - descifrada))
    print(f"    Error tras cifrar/descifrar: {error_mask:.6f}")
    assert error_mask < 0.01, "ERROR: La máscara no se canceló correctamente"
    print(f"    ✅ Cancelación de máscara verificada (error < 0.01)")

    # Test de troceado
    señal_larga = np.random.randn(48000).astype(np.float32) * 0.5
    bloques = trocear_señal(señal_larga)
    print(f"\n[6] Troceado: {len(señal_larga)} muestras → {len(bloques)} bloques "
          f"de {SAMPLES_PER_BLOCK} muestras")

    print(f"\n{'=' * 55}")
    print(f"  ✅ TODOS LOS TESTS PASADOS")
    print(f"{'=' * 55}")
