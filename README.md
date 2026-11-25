# 📻 SDR Radio Transmitter — Redes de Ordenadores

**Radio Definida por Software con Python**  
*Grabar · Modular · Transmitir por UDP · Recibir · Demodular*

---

## Estructura del Proyecto

```
redes-radio/
│
├── fase1_audio.py          ← FASE 1: Lectura WAV, FFT y normalización de voz
├── fase2_modulacion.py     ← FASE 2: Modulación AM y FM sobre portadora simulada
├── fase3_comparativa.py    ← FASE 3: Comparativa AM vs FM y resistencia al ruido
├── fase4_transporte.py     ← FASE 4: Protocolo UDP propio, fragmentación y cifrado musical
├── fase5_integracion.py    ← FASE 5: Modulación AM/FM en tiempo real sobre red UDP
│
├── interfaz_grafica.py     ← Interfaz gráfica interactiva (5 pestañas, todo integrado)
│
├── audio_original.wav      ← Voz humana en español (audio de entrada)
├── mascara_musical.wav     ← Melodía 8-bits para esteganografía acústica
│
├── modulos/                ← Módulos auxiliares internos (protocolo, transmisor, receptor)
│   ├── protocolo_sdr.py         Cabecera binaria SDR (27 bytes, Big-Endian)
│   ├── modulador_en_linea.py    Modulación bloque a bloque con continuidad de fase
│   ├── transmisor_udp.py        Emisor UDP por línea de comandos (Fase 4)
│   ├── receptor_udp.py          Receptor UDP por línea de comandos (Fase 4)
│   ├── emisor_modulado_udp.py   Emisor modulado AM/FM por línea de comandos (Fase 5)
│   ├── receptor_modulado_udp.py Receptor modulado AM/FM por línea de comandos (Fase 5)
│   └── interceptor_udp.py       Demostración de espía de red (escucha sin clave → música)
│
├── audios/                 ← Audios generados por las pruebas y validaciones
├── logs/                   ← Registros de transmisión UDP (trazabilidad)
└── plots/                  ← Gráficas e informes de evidencias por fase
```

---

## Fases del Proyecto

| Fase | Archivo | Descripción | Estado |
|:---:|:---|:---|:---:|
| 1 | `fase1_audio.py` | Lectura WAV, eliminación DC, normalización, FFT | ✅ Completada |
| 2 | `fase2_modulacion.py` | Modulación AM y FM, portadora 8 kHz, demodulación local | ✅ Completada |
| 3 | `fase3_comparativa.py` | Comparativa espectral AM vs FM, simulación AWGN | ✅ Completada |
| 4 | `fase4_transporte.py` | Protocolo binario propio, fragmentación UDP, cifrado musical | ✅ Completada |
| 5 | `fase5_integracion.py` | Modulación AM/FM bloque a bloque + UDP + PLC | ✅ Completada |

---

## Cómo Ejecutar

### Interfaz Gráfica (recomendado)
```bash
python interfaz_grafica.py
```

### Scripts de Fase por Consola
```bash
python fase1_audio.py          # Análisis y FFT del audio
python fase2_modulacion.py     # Modulación AM y FM
python fase3_comparativa.py    # Comparativa AM vs FM
python fase4_transporte.py     # Validación de transporte UDP
python fase5_integracion.py    # Cadena completa AM/FM sobre UDP
```

### Demo del Espía de Red (Esteganografía Acústica)
```bash
# En una terminal: lanza la transmisión en la interfaz gráfica (con máscara activada)
# En otra terminal:
python modulos/interceptor_udp.py --play
# → El espía solo escucha música de 8 bits. La voz está oculta.
```

---

## Dependencias

```bash
pip install numpy scipy matplotlib sounddevice
```

---

## Parámetros Técnicos Clave

| Parámetro | Valor | Por qué |
|:---|:---:|:---|
| Frecuencia de muestreo | 48 000 Hz | Margen suficiente para portadora + voz (Nyquist 24 kHz) |
| Portadora fc | 8 000 Hz | Centro del espectro audible simulado |
| Índice AM ka | 0.90 | Modulación profunda sin sobremodulación |
| Desviación FM kf | 3 500 Hz | Ancho de Carson ≈ 2(3500+4000) = 15 kHz |
| Tamaño de bloque | 512 muestras | ≈ 10 ms por paquete, respeta MTU Ethernet (1500 B) |
| Puerto UDP | 5005 / 5018 | Rango libre, sin colisión con puertos del sistema |

---

*Equipo SDR — Asignatura: Redes de Ordenadores / Tratamiento de Señales*
