
import os
from pathlib import Path

# ============================================================================
# RUTAS
# ============================================================================
# Configurables por variable de entorno para no depender de la máquina de
# cada quien. Si no se define la variable, se usa una ruta por defecto
# relativa a este archivo (asume la estructura de carpetas del repo).
#   Windows (PowerShell):
#     $env:RUTA_PARTICIONES = "D:\Datasets\Particiones"
#   Linux/macOS/Colab:
#     export RUTA_PARTICIONES=/content/Particiones

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

# Carpeta raíz que contiene las particiones.
RUTA_PARTICIONES = Path(
    os.environ.get("RUTA_PARTICIONES", RAIZ_PROYECTO / "Datasets" / "Particiones")
)

# Carpeta donde se guardarán modelos entrenados y métricas (se crea sola).
RUTA_RESULTADOS = Path(
    os.environ.get("RUTA_RESULTADOS", RAIZ_PROYECTO / "Resultados")
)

# Subcarpetas de resultados
RUTA_MODELOS = RUTA_RESULTADOS / "modelos"      # pesos .pth
RUTA_METRICAS = RUTA_RESULTADOS / "metricas"    # archivos .json/.csv con métricas


# ============================================================================
# DATOS
# ============================================================================
# Generadores disponibles (nombres = carpetas dentro de Particiones/)
GENERADORES = ["StyleGAN2", "StyleGAN3", "SDXL", "Flux"]

# Nombres de los splits y clases (coinciden con la estructura de carpetas)
SPLITS = ["train", "val", "test"]
CLASES = ["real", "fake"]           # ImageFolder las ordena alfabéticamente:
                                    #   fake -> 0 , real -> 1
# (Nota: 'fake' va antes que 'real' en orden alfabético, así que la clase 0
#  es fake y la clase 1 es real. Tenlo presente al interpretar métricas.)

TAMANO_IMAGEN = 256                 # las imágenes ya vienen a 256x256 PNG

# ----------------------------------------------------------------------------
# MULTICLASE: identifica a qué GENERADOR pertenece un rostro (o si es real),
# en vez de solo fake/real. Usa las particiones balanceadas creadas por
# Particiones/Experimentos/Multiclase/Crear_Particiones_Multiclase.py, donde
# "real" combina FFHQ + CelebA-HQ a partes iguales.
RUTA_PARTICIONES_MULTICLASE = Path(
    os.environ.get(
        "RUTA_PARTICIONES_MULTICLASE",
        RAIZ_PROYECTO / "Datasets" / "Particiones" / "Experimentos" / "Multiclase",
    )
)

# Orden alfabético = orden real que asigna ImageFolder (confirmado con
# ImageFolder(...).class_to_idx): Flux=0, SDXL=1, StyleGAN2=2, StyleGAN3=3,
# real=4. Si se agrega o quita un generador, este orden puede cambiar solo:
# NO hardcodear los índices en otro lado, siempre derivarlos de esta lista
# (sorted(CLASES_MULTICLASE)) o de dataset.class_to_idx directamente.
CLASES_MULTICLASE = ["Flux", "SDXL", "StyleGAN2", "StyleGAN3", "real"]
NUM_CLASES_MULTICLASE = len(CLASES_MULTICLASE)

# Normalización de ImageNet: se aplica AL CARGAR cada imagen (no en disco).
# Debe usarse junto con pesos preentrenados de ImageNet.
IMAGENET_MEDIA = [0.485, 0.456, 0.406]
IMAGENET_DESV = [0.229, 0.224, 0.225]

# ============================================================================
# AUMENTO DE ROBUSTEZ (anti-atajo)
# ============================================================================
# Real y cualquier generador sintético difieren en nitidez/ruido de sensor de
# forma consistente e independiente del generador, así que un clasificador
# puede aprender "¿tiene el nivel de detalle de una foto real?" en vez de
# huellas específicas de un generador. Para evitar que el modelo dependa de
# un nivel FIJO de blur o de compresión como atajo, en cada época de
# entrenamiento se le aplica a cada imagen, con cierta probabilidad, un blur
# gaussiano y/o una recompresión JPEG de intensidad ALEATORIA. Así el modelo
# se ve obligado a ser invariante a esa dimensión y a aprender señales más
# genuinas. Solo se aplica en 'train' (val/test se evalúan sin alterar).
AUMENTO_ROBUSTEZ = False
PROB_BLUR = 0.2
BLUR_SIGMA_RANGO = (0.1, 2.5)
PROB_JPEG = 0.2
JPEG_CALIDAD_RANGO = (30, 95)


# ============================================================================
# REPRODUCIBILIDAD
# ============================================================================

# Semilla para el particionado de datos (la misma de crear_particiones.py).
SEMILLA_PARTICION = 42

# Tres semillas para el ENTRENAMIENTO. Se corre cada configuración con las tres
# y se reporta media ± desviación estándar. Son distintas a propósito: aquí
# SÍ queremos variabilidad, para medir la estabilidad del resultado.
SEMILLAS_ENTRENAMIENTO = [42, 123, 2024]


# ============================================================================
# MODELO
# ============================================================================

# Arquitectura a entrenar localmente. Los nombres siguen la convención de timm.
# (EfficientNet-B0 es el más ligero -> el adecuado para la RTX 3050 de 4 GB.)
MODELO = "efficientnet_b0"

# Transfer learning: partir de pesos preentrenados en ImageNet y hacer
# fine-tuning de TODA la red (no se congelan capas), para que el modelo
# aproveche la base visual pero se especialice en las huellas del generador.
PREENTRENADO = True
CONGELAR_CAPAS = False

NUM_CLASES = 2   # real vs fake


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

# Batch conservador para 4 GB de VRAM. Si aparece "CUDA out of memory",
# baja a 8. Si te sobra memoria (en Colab), puedes subir a 32 o 64.
BATCH_SIZE = 16

# Precisión mixta (AMP): usa float16 donde puede, reduce ~a la mitad el uso
# de memoria y acelera, casi sin costo de precisión. Clave para 4 GB.
USAR_AMP = True

EPOCAS = 20                    # tope de épocas
LEARNING_RATE = 1e-4           # lr moderado, típico para fine-tuning
WEIGHT_DECAY = 1e-4            # regularización L2

# Early stopping: si la métrica de validación no mejora en N épocas seguidas,
# se detiene para no sobreajustar ni perder tiempo.
PACIENCIA_EARLY_STOPPING = 5

# Número de procesos para cargar datos. En Windows conviene 0 o 2 para evitar
# problemas; en Linux/Colab puedes subirlo (4).
NUM_WORKERS = 2


# ============================================================================
# DISPOSITIVO
# ============================================================================

import torch  

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preparar_carpetas() -> None:
    for carpeta in [RUTA_RESULTADOS, RUTA_MODELOS, RUTA_METRICAS]:
        carpeta.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Dispositivo detectado: {DISPOSITIVO}")
    print(f"Particiones en: {RUTA_PARTICIONES}")
    print(f"¿Existe la carpeta de particiones? {RUTA_PARTICIONES.exists()}")
    print(f"Modelo: {MODELO} (preentrenado={PREENTRENADO})")
    print(f"Batch size: {BATCH_SIZE}, AMP: {USAR_AMP}")
    print(f"Semillas de entrenamiento: {SEMILLAS_ENTRENAMIENTO}")