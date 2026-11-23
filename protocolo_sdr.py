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
