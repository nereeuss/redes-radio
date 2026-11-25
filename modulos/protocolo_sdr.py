"""
=============================================================
 PROTOCOLO SDR — Definición de cabecera binaria UDP
=============================================================
Asignatura: Redes de Ordenadores / Tratamiento de Señales
Fase 4: Transporte de muestras sin modular (Voz y Música)

Estructura de la cabecera (27 bytes, formato de red Big-Endian '!'):
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     MAGIC (4 bytes: "SDRM")                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|    VERSION    |     FLAGS     |    CHANNELS   |   MODULATION  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     DTYPE     |        SAMPLES (muestras/canal)               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                          SAMPLE_RATE                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        SEQUENCE_NUMBER                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      TIMESTAMP (8 bytes, double)              |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       PAYLOAD (muestras PCM)                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
"""

import struct
import time
import os
import sys

# Añadir vendor al path si existe
_vendor = os.path.join(os.path.dirname(__file__), "vendor")
if os.path.isdir(_vendor) and _vendor not in sys.path:
    sys.path.insert(0, _vendor)

import numpy as np
from typing import Tuple, Optional

# Constantes del protocolo
MAGIC_BYTES = b"SDRM"      # Software Defined Radio - Music/Voice
PROTO_VERSION = 1

# Flags de control
FLAG_START = 0x01          # Primer paquete / aviso de inicio
FLAG_DATA  = 0x02          # Paquete ordinario con muestras
FLAG_END   = 0x04          # Fin de transmisión

# Tipos de modulación
MOD_RAW = 0                # Muestras sin modular (Fase 4 - Transporte)
MOD_AM  = 1                # Amplitud Modulada (Fase 5)
MOD_FM  = 2                # Frecuencia Modulada (Fase 5)

# Formato numérico de muestras en el payload
DTYPE_INT16   = 1          # PCM 16-bit con signo (2 bytes/muestra)
DTYPE_FLOAT32 = 2          # IEEE 754 float32 (4 bytes/muestra)

# Formato struct: Big-endian ('!')
# 4s : magic (4B)
# B  : version (1B)
# B  : flags (1B)
# B  : channels (1B: 1=mono, 2=estéreo)
# B  : modulation (1B: 0=RAW, 1=AM, 2=FM)
# B  : dtype (1B: 1=int16, 2=float32)
# H  : samples_per_channel (2B unsigned short)
# I  : sample_rate (4B unsigned int)
# I  : sequence (4B unsigned int)
# d  : timestamp (8B double)
HEADER_FORMAT = "!4sBBBBBHIId"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 27 bytes


class CabeceraSDR:
    """Representa y manipula la cabecera del protocolo de transporte SDR."""

    def __init__(
        self,
        sequence: int = 0,
        samples: int = 512,
        sample_rate: int = 48000,
        channels: int = 1,
        modulation: int = MOD_RAW,
        dtype: int = DTYPE_INT16,
        flags: int = FLAG_DATA,
        timestamp: Optional[float] = None,
        version: int = PROTO_VERSION,
    ):
        self.magic = MAGIC_BYTES
        self.version = version
        self.flags = flags
        self.channels = channels
        self.modulation = modulation
        self.dtype = dtype
        self.samples = samples  # Muestras por canal en este bloque
        self.sample_rate = sample_rate
        self.sequence = sequence
        self.timestamp = timestamp if timestamp is not None else time.time()

    @property
    def is_start(self) -> bool:
        return bool(self.flags & FLAG_START)

    @property
    def is_data(self) -> bool:
        return bool(self.flags & FLAG_DATA)

    @property
    def is_end(self) -> bool:
        return bool(self.flags & FLAG_END)

    @property
    def modulation_name(self) -> str:
        nombres = {MOD_RAW: "RAW (Sin modular)", MOD_AM: "AM", MOD_FM: "FM"}
        return nombres.get(self.modulation, f"DESC({self.modulation})")

    @property
    def dtype_name(self) -> str:
        nombres = {DTYPE_INT16: "int16", DTYPE_FLOAT32: "float32"}
        return nombres.get(self.dtype, f"DESC({self.dtype})")

    @property
    def channels_name(self) -> str:
        return "Mono" if self.channels == 1 else ("Estéreo" if self.channels == 2 else f"{self.channels}ch")

    def pack(self, payload: bytes = b"") -> bytes:
        """Serializa la cabecera y adjunta el payload de audio."""
        header_bytes = struct.pack(
            HEADER_FORMAT,
            self.magic,
            self.version,
            self.flags,
            self.channels,
            self.modulation,
            self.dtype,
            self.samples,
            self.sample_rate,
            self.sequence,
            self.timestamp,
        )
        return header_bytes + payload

    @classmethod
    def unpack(cls, datagram: bytes) -> Tuple["CabeceraSDR", bytes]:
        """
        Deserializa un datagrama UDP en cabecera y payload.
        Lanza ValueError si el paquete es más corto que la cabecera o el MAGIC no coincide.
        """
        if len(datagram) < HEADER_SIZE:
            raise ValueError(
                f"Datagrama demasiado corto ({len(datagram)} B < {HEADER_SIZE} B de cabecera)"
            )

        fields = struct.unpack(HEADER_FORMAT, datagram[:HEADER_SIZE])
        magic = fields[0]

        if magic != MAGIC_BYTES:
            raise ValueError(f"Magic inválido: {magic} (esperado: {MAGIC_BYTES})")

        cabecera = cls(
            version=fields[1],
            flags=fields[2],
            channels=fields[3],
            modulation=fields[4],
            dtype=fields[5],
            samples=fields[6],
            sample_rate=fields[7],
            sequence=fields[8],
            timestamp=fields[9],
        )
        payload = datagram[HEADER_SIZE:]
        return cabecera, payload

    def __repr__(self) -> str:
        flags_str = []
        if self.is_start:
            flags_str.append("START")
        if self.is_data:
            flags_str.append("DATA")
        if self.is_end:
            flags_str.append("END")
        f_repr = "|".join(flags_str) if flags_str else "NONE"

        return (
            f"<CabeceraSDR seq={self.sequence} flg={f_repr} "
            f"ch={self.channels_name} fs={self.sample_rate}Hz "
            f"mod={self.modulation_name} fmt={self.dtype_name} "
            f"samples={self.samples}>"
        )

# ============================================================
# COMPATIBILIDAD CON INTERFAZ GRÁFICA Y FASE 3
# ============================================================
BLOCK_MS = 10
FS = 48_000
SAMPLES_PER_BLOCK = int(FS * BLOCK_MS / 1000)
MASK_OFF = 0
MASK_ON = 1
DEFAULT_PORT = 5005
DEFAULT_IP = "127.0.0.1"
FC_DEFAULT = 8000
MAGIC = b"SDR1"
HEADER_FORMAT_NEREEUSS = "!4sIdBHBH"
HEADER_SIZE_NEREEUSS = 22

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
        HEADER_FORMAT_NEREEUSS,
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
    if len(datagrama) < HEADER_SIZE_NEREEUSS:
        raise ValueError(f"Datagrama demasiado corto: {len(datagrama)} bytes "
                         f"(mínimo {HEADER_SIZE_NEREEUSS})")

    # Paso 1: Desempaquetar la cabecera
    magic, n_seq, timestamp, mod, fc, mask, n_muestras = struct.unpack(
        HEADER_FORMAT_NEREEUSS,
        datagrama[:HEADER_SIZE_NEREEUSS]
    )

    # Paso 2: Verificar que es un paquete nuestro
    if magic != MAGIC:
        raise ValueError(f"MAGIC incorrecto: {magic!r} (esperado {MAGIC!r}). "
                         f"Este paquete no pertenece a nuestra radio.")

    # Paso 3: Decodificar el payload
    payload_bytes = datagrama[HEADER_SIZE_NEREEUSS:]
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
