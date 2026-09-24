# Detección de rostros sintéticos — código de la tesis

Este repositorio contiene el código para entrenar y evaluar
clasificadores CNN (EfficientNet-B0, ResNet-50, Xception) que distinguen
rostros reales de rostros generados por StyleGAN2, StyleGAN3, SDXL y Flux, así
como el experimento multiclase (identificar a qué generador pertenece un
rostro) y la webapp de demostración.

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

`Datasets/` y `Resultados/` (imágenes, particiones y pesos `.pth` entrenados)
están fuera del repositorio (ver `.gitignore`): son datos pesados y
específicos de cada máquina, no código. Los scripts los recrean o los leen
desde ahí en tiempo de ejecución.

## Configuración (rutas)

`Codigos Experimentos/config.py` es la fuente única de configuración del
pipeline de entrenamiento/evaluación. Por defecto asume que `Datasets/` y
`Resultados/` viven junto a este repositorio (`ExperimentosRostrosSinteticos/Datasets`,
`ExperimentosRostrosSinteticos/Resultados`), pero cada ruta se puede
sobreescribir con una variable de entorno si tu estructura es distinta:

| Variable | Por defecto | Uso |
|---|---|---|
| `RUTA_DATASETS` | `Datasets/` | Imágenes crudas / carpeta de entrada de `Crear_Particiones*.py` |
| `RUTA_PARTICIONES` | `Datasets/Particiones` | Particiones train/val/test (binario) |
| `RUTA_PARTICIONES_MULTICLASE` | `Datasets/Particiones/Experimentos/Multiclase` | Particiones del experimento multiclase |
| `RUTA_RESULTADOS` | `Resultados/` | Modelos entrenados (`Resultados/modelos`) y métricas (`Resultados/metricas`) |
| `RUTA_RESULTADOS_A` | `Resultados Experimento A/` | Usado por `analizar_resultados.py` si los `.json` del Experimento A se movieron a mano |
| `RUTA_RAIZ_PROYECTO` | raíz del repo | Usado por `analizar_resultados.py` para buscar `.json` sueltos en todo el proyecto |

Ejemplo (PowerShell):
```powershell
$env:RUTA_PARTICIONES = "D:\Datasets\Particiones"
```

## Reproducir las figuras y tablas

Los scripts de `Codigos Figuras y Tablas/` leen sus `.json` de entrada desde
**su propia carpeta** (no desde `JSON Experimentos/` automáticamente). Para
regenerar una figura o tabla:

1. Copia el/los `.json` correspondiente(s) desde `JSON Experimentos/<experimento>/`
   a `Codigos Figuras y Tablas/`.
2. Corre el script (por ejemplo `python generar_tablas_celeba.py`).
3. La salida queda en `Codigos Figuras y Tablas/figuras/...` (no sobreescribe
   `Figuras y Tablas/`, que son las versiones finales ya usadas en la tesis).

Excepción: `generar_matriz_confusion_multiclase.py` sí lee directamente de
`JSON Experimentos/Multiclase/experimentoMulticlase_efficientnet_b0.json`, sin
necesidad de copiarlo.

`generar_curvas_roc.py` (y los scripts que dependen de él) además necesita los
pesos `.pth` del Experimento A y corre inferencia real sobre las particiones
de prueba — no alcanza con los `.json`.

## Pesos entrenados (`.pth`)

Los modelos entrenados **no están en este repositorio**: en total pesan más
de 2 GB (docenas de checkpoints de 16-91 MB cada uno), muy por encima de lo
razonable para un repo de git. Si necesitas los pesos para reproducir
inferencia o la webapp, se comparten aparte (Drive/almacenamiento externo) y
se colocan en `Resultados/modelos/` (o la ruta que indique
`RUTA_RESULTADOS`), que ya está gitignoreada.

## Dependencias

No hay un único `requirements.txt` para todo el pipeline porque cada parte
tiene necesidades distintas:

- **Entrenamiento/evaluación/figuras** (`Codigos Experimentos/`,
  `Codigos Figuras y Tablas/`): `torch`, `torchvision`, `timm`, `numpy`,
  `pandas`, `openpyxl`, `matplotlib`, `scikit-learn`, `Pillow`. Instala
  PyTorch aparte según tengas GPU o no (ver `webapp/requirements.txt` para el
  comando exacto).
- **Generación de dataset StyleGAN2** (`Codigos Figuras y Tablas/generar_dataset_stylegan2.py`):
  ver `Codigos Figuras y Tablas/requirements_stylegan2.txt`.
- **Webapp backend** (`webapp/`): `pip install -r webapp/requirements.txt`.
- **Webapp frontend** (`webapp/frontend/`): Next.js —
  `cd webapp/frontend && npm install`.

## Correr la webapp

```bash
# Backend (puerto 5001)
cd webapp
pip install -r requirements.txt
python app.py

# Frontend
cd webapp/frontend
npm install
npm run dev
```

Requiere el modelo multiclase entrenado en `Resultados/modelos/multiclase/`
(ver "Pesos entrenados" arriba); sin él, el backend levanta igual pero
`/api/analizar` falla.
# ExperimentosRostrosSinteticos
