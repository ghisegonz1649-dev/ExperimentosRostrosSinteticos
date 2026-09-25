# Detección de rostros sintéticos 

Este repositorio contiene el código sobre detección de rostros
sintéticos generados por IA. El objetivo es entrenar y evaluar clasificadores
CNN (EfficientNet-B0, ResNet-50 y Xception, vía `timm`) que distinguen rostros
reales de rostros generados por cuatro modelos generativos distintos:
StyleGAN2, StyleGAN3, SDXL y Flux. Además de la clasificación binaria
(real/fake), el proyecto incluye:

- **Experimento A**: entrenamiento y evaluación base por arquitectura sobre
  StyleGAN2.
- **Experimentos B, C y D**: variantes de robustez/generalización sobre las
  mismas arquitecturas.
- **Experimento E**: fine-tuning incremental / evolución del modelo a lo largo
  de varias rondas (curvas de recall, AUC, accuracy y especificidad "visto vs.
  no visto").
- **Experimento multiclase**: en vez de solo real/fake, identifica a qué
  generador pertenece un rostro (Flux, SDXL, StyleGAN2, StyleGAN3 o real,
  donde "real" combina FFHQ y CelebA-HQ a partes iguales).
- **Webapp de demostración**: backend Flask + frontend Next.js que permite
  subir una imagen y ver la clasificación (binaria y multiclase) junto con un
  mapa de calor Grad-CAM.

Cada entrenamiento se corre con tres semillas (42, 123, 2024) y se reporta
media ± desviación estándar para medir la estabilidad de los resultados.

## Estructura

```
Codigos Experimentos/        Config, motor de entrenamiento/evaluación,
                              creación de particiones y notebooks de cada
                              experimento (A, B/C/D, E, multiclase).
Codigos Figuras y Tablas/     Scripts que, a partir de los .json de métricas
                              ya generados, producen las figuras y tablas
                              finales (no reentrenan ni reevalúan nada).
JSON Experimentos/            Métricas crudas (.json) de cada experimento,
                              organizadas por experimento — el archivo de
                              donde salen las figuras y tablas.
Figuras y Tablas/             Figuras (.png) y tablas (.csv/.xlsx) finales,
                              las que se usan en el documento de la tesis.
webapp/                       Backend Flask + frontend Next.js de la página
                              de demostración (clasificación e inferencia
                              multiclase con Grad-CAM).
```

> **Nota:** las carpetas `Datasets/` (imágenes reales y sintéticas) y
> `Resultados/` (pesos `.pth` de los modelos entrenados) no se incluyen en
> este repositorio por su tamaño. Ver la sección [Datasets y modelos
> entrenados](#datasets-y-modelos-entrenados) más abajo.

## Requerimientos computacionales

### Entrenamiento y evaluación (`Codigos Experimentos/`)

- **Python** 3.10+ (probado con notebooks de Jupyter).
- **GPU con CUDA recomendada.** El proyecto se desarrolló y ajustó (batch
  size, uso de AMP) para una **NVIDIA RTX 3050 de 4 GB de VRAM**; también
  corre en CPU o en Google Colab, pero de forma mucho más lenta. Si aparece
  `CUDA out of memory`, bajar `BATCH_SIZE` en `config.py` (por defecto 16);
  con más VRAM (p. ej. en Colab) se puede subir a 32 o 64.
- Se usa precisión mixta (AMP) para reducir el uso de memoria.
- Imágenes de entrada de 256×256 px.
- Espacio en disco: varios GB para las particiones de imágenes (train/val/test
  por generador) más los pesos de los modelos entrenados (uno por
  arquitectura/experimento/semilla).
- Dependencias principales: `torch` + `torchvision` (CUDA 11.8 si hay GPU
  NVIDIA, o CPU-only si no), `timm`, `scikit-learn`, `numpy`, `pandas`,
  `Pillow`.

Instalación sugerida:

```bash
# Con GPU NVIDIA (CUDA 11.8)
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu118

# Sin GPU (CPU)
pip install torch==2.7.1 torchvision

pip install timm scikit-learn numpy pandas Pillow openpyxl
```

Las rutas de datos y resultados son configurables por variable de entorno
(`RUTA_PARTICIONES`, `RUTA_RESULTADOS`, `RUTA_PARTICIONES_MULTICLASE`); por
defecto asumen la estructura `Datasets/Particiones/` y `Resultados/` dentro
del repo (ver `Codigos Experimentos/config.py`).

### Generación de figuras y tablas (`Codigos Figuras y Tablas/`)

Estos scripts solo leen los `.json` de `JSON Experimentos/` y no requieren
GPU. Dependencias: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `openpyxl`
(para exportar `.xlsx`). Reutilizan `timm`/`torch` únicamente en los scripts
que recalculan curvas ROC a partir de un modelo guardado.

### Webapp de demostración (`webapp/`)

- **Backend** (`webapp/requirements.txt`): Python 3.10+, Flask 3.1, flask-cors,
  Pillow, numpy, timm, y `torch` (mismo criterio CUDA/CPU que arriba). No
  necesita GPU para inferencia, pero es más rápido con una.
- **Frontend** (`webapp/frontend/`): Node.js 18+ y npm. Next.js 15, React 19,
  Tailwind CSS 4, Recharts. Instalar con `npm install` y correr con
  `npm run dev` dentro de `webapp/frontend/`.

## Datasets y modelos entrenados

Los datasets de imágenes (reales y generadas por StyleGAN2, StyleGAN3, SDXL y
Flux) usados en los experimentos, así como los pesos de los modelos ya
entrenados, están disponibles en kaggle (no se incluyen en el
repositorio por su tamaño):

- StyleGAN3: https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3
- 130K Real vs Fake Face (SDXL y FLUX.1): https://www.kaggle.com/datasets/shreyanshpatel1/130k-real-vs-fake-face
- FFHQ (rostros reales): https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
- CelebA-HQ 256x256 (rostros reales): https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256

Para reproducir un experimento localmente, descargar las carpetas
correspondientes desde el Drive y ubicarlas según lo esperado por
`Codigos Experimentos/config.py` (o apuntar las variables de entorno
`RUTA_PARTICIONES` / `RUTA_PARTICIONES_MULTICLASE` a donde se hayan
descargado).
