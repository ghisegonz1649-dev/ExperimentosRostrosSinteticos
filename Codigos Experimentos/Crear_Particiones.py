"""
crear_particiones.py
Crea las particiones train/val/test (70/15/15) para cada generador y deja
las imágenes LISTAS para los experimentos: recorte central cuadrado,
redimensionado a 256x256, RGB y formato PNG uniforme.

Estructura de entrada (p. ej. C:\\ruta\\a\\Datasets):
    Flux/  Reales/  SDXL/  StyleGAN2/  StyleGAN3/   (7,000 imágenes c/u)

Estructura de salida:
    Datasets/Particiones/<Generador>/<split>/<clase>/
    p. ej. Particiones/StyleGAN2/train/fake/  y  Particiones/StyleGAN2/train/real/

Puntos metodológicos clave:
- La partición de REALES se calcula UNA vez con semilla fija y se reutiliza
  igual en los cuatro generadores, para que el único factor que cambie entre
  experimentos sea el generador sintético.
- TODAS las imágenes se guardan en el mismo formato y resolución (PNG, 256x256).
- IGUALACIÓN DE ARTEFACTOS: los generadores difieren en resolución nativa
  (Reales/StyleGAN3: 256x256; StyleGAN2/SDXL/Flux: 1024x1024 con downsampling
  Lanczos) y en compresión de origen (Reales/SDXL/Flux: JPEG; StyleGAN2/
  StyleGAN3: PNG), lo que correlacionaría "historial JPEG" con la etiqueta
  real/fake. Por eso TODAS las imágenes se recomprimen con la misma calidad
  JPEG antes de guardarse como PNG final, de forma simétrica para todas las
  clases. (Se descartó además igualar el historial de RESCALADO subiendo
  Reales/StyleGAN3 a 1024 y bajándolos de nuevo: el reescalado ida-y-vuelta
  deja una firma espectral distinta que reintroduce el atajo, solo que
  invertido. La invarianza a resolución/nitidez se maneja en el
  ENTRENAMIENTO — nucleo.py, blur aleatorio — no en este preprocesamiento).

Requiere: pip install pillow
"""

import io
import os
import random
from pathlib import Path

from PIL import Image

# ----------------------------- Configuración -----------------------------

# Configurable por variable de entorno; por defecto usa Datasets/ junto a
# la raíz del repo (un nivel arriba de Codigos Experimentos/).
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATASETS = Path(os.environ.get("RUTA_DATASETS", RAIZ_PROYECTO / "Datasets"))
RUTA_SALIDA = RUTA_DATASETS / "Particiones"

GENERADORES = ["StyleGAN2", "StyleGAN3", "SDXL", "Flux"]
CARPETA_REALES = "Reales"

SEMILLA = 42  # debe coincidir con la semilla de particionado en config.py

PROPORCIONES = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = ["train", "val", "test"]

EXTENSIONES_VALIDAS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

TAMANO_OBJETIVO = 256  # resolución final: 256x256 (debe coincidir con config.py)
FORMATO_SALIDA = ".png"  # formato uniforme sin pérdida para ambas clases

# Calidad JPEG común aplicada a TODAS las imágenes (incluidas las que nacen
# PNG sin compresión) para igualar el historial de compresión entre clases.
CALIDAD_JPEG_COMUN = 90


# ------------------------------- Funciones -------------------------------

def listar_imagenes(carpeta: Path) -> list[Path]:
    """Lista las imágenes de una carpeta en orden determinista (sorted)."""
    if not carpeta.exists():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")
    archivos = sorted(
        p for p in carpeta.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONES_VALIDAS
    )
    if not archivos:
        raise ValueError(f"No se encontraron imágenes en: {carpeta}")
    return archivos


def dividir(archivos: list[Path], semilla: int) -> dict[str, list[Path]]:
    """Baraja con semilla fija y divide en train/val/test según PROPORCIONES."""
    rng = random.Random(semilla)
    barajados = archivos.copy()
    rng.shuffle(barajados)

    n = len(barajados)
    n_train = int(n * PROPORCIONES["train"])
    n_val = int(n * PROPORCIONES["val"])

    return {
        "train": barajados[:n_train],
        "val": barajados[n_train:n_train + n_val],
        "test": barajados[n_train + n_val:],
    }


def procesar_imagen(origen: Path, destino: Path) -> None:
    """Abre la imagen, la convierte a RGB, aplica recorte central cuadrado
    (para no deformar caras si la imagen no fuera cuadrada), la redimensiona
    a TAMANO_OBJETIVO x TAMANO_OBJETIVO, IGUALA su historial de compresión
    frente a las demás clases, y la guarda como PNG."""
    with Image.open(origen) as img:
        img = img.convert("RGB")

        # Recorte central cuadrado (si ya es cuadrada, no cambia nada)
        ancho, alto = img.size
        lado = min(ancho, alto)
        izq = (ancho - lado) // 2
        arriba = (alto - lado) // 2
        img = img.crop((izq, arriba, izq + lado, arriba + lado))

        # Redimensionado con filtro Lanczos (mejor calidad para reducir).
        # Cada imagen pasa por UN solo redimensionado, directo desde su
        # resolución nativa (1024 o 256) hasta 256: no se sube de resolución
        # artificialmente (ver nota en el docstring del módulo sobre por qué
        # se descartó esa idea).
        img = img.resize((TAMANO_OBJETIVO, TAMANO_OBJETIVO), Image.LANCZOS)

        # Igualación de artefactos de compresión: se recodifica TODA imagen
        # (incluidas las que nacieron PNG sin compresión, como StyleGAN2 y
        # StyleGAN3) con la misma calidad JPEG, para que "no tener historial
        # de compresión" tampoco sea una pista trivial. El resultado se
        # vuelve a decodificar y se guarda como PNG (contenedor sin pérdida
        # adicional, pero con los artefactos JPEG ya incrustados en los
        # píxeles). Esta operación SÍ es simétrica: se aplica igual sin
        # importar la resolución nativa de origen.
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=CALIDAD_JPEG_COMUN)
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")

        img.save(destino, format="PNG")


def archivo_valido(ruta: Path) -> bool:
    """Verifica que un archivo ya existente sea PNG de 256x256.
    Leer el tamaño solo carga la cabecera del archivo, así que es rápido."""
    try:
        with Image.open(ruta) as img:
            return (img.format == "PNG"
                    and img.size == (TAMANO_OBJETIVO, TAMANO_OBJETIVO))
    except Exception:
        return False  # archivo corrupto o ilegible -> reprocesar


def procesar_y_copiar(archivos: list[Path], destino: Path) -> int:
    """Procesa (256x256, RGB, PNG) y guarda la lista de archivos en destino.
    Si el archivo destino ya existe pero NO cumple (formato o tamaño
    incorrectos, p. ej. de una corrida anterior sin procesamiento),
    se reprocesa desde el origen. Devuelve cuántas imágenes procesó."""
    destino.mkdir(parents=True, exist_ok=True)

    # Limpieza: eliminar residuos de corridas anteriores que no sean PNG
    # (p. ej. .jpg copiados sin procesar por una versión previa del script)
    for residuo in destino.iterdir():
        if residuo.is_file() and residuo.suffix.lower() != FORMATO_SALIDA:
            residuo.unlink()

    procesadas = 0
    for archivo in archivos:
        ruta_destino = destino / (archivo.stem + FORMATO_SALIDA)
        if not (ruta_destino.exists() and archivo_valido(ruta_destino)):
            procesar_imagen(archivo, ruta_destino)
            procesadas += 1
    return procesadas


def guardar_listado(particion: dict[str, list[Path]], ruta_txt: Path) -> None:
    """Guarda un registro en texto de qué archivo quedó en qué split
    (útil como evidencia de reproducibilidad en la tesis)."""
    ruta_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_txt, "w", encoding="utf-8") as f:
        for split in SPLITS:
            for archivo in particion[split]:
                f.write(f"{split}\t{archivo.name}\n")


# --------------------------------- Main ----------------------------------

def main() -> None:
    print(f"Semilla: {SEMILLA}")
    print(f"Proporciones: {PROPORCIONES}\n")

    # 0) Conteo inicial de imágenes por dataset (antes de particionar)
    print("=== Imágenes por dataset ===")
    total_general = 0
    for carpeta in [CARPETA_REALES] + GENERADORES:
        n = len(listar_imagenes(RUTA_DATASETS / carpeta))
        total_general += n
        print(f"  {carpeta}: {n:,} imágenes")
    print(f"  TOTAL: {total_general:,} imágenes\n")

    # 1) Partición de reales: se calcula UNA vez y se comparte entre todos
    reales = listar_imagenes(RUTA_DATASETS / CARPETA_REALES)
    particion_reales = dividir(reales, SEMILLA)
    guardar_listado(particion_reales, RUTA_SALIDA / "listados" / "reales.txt")
    print(f"Reales: {len(reales)} imágenes -> "
          f"train={len(particion_reales['train'])}, "
          f"val={len(particion_reales['val'])}, "
          f"test={len(particion_reales['test'])}\n")

    # 2) Para cada generador: partir las falsas y copiar ambas clases
    for generador in GENERADORES:
        print(f"--- {generador} ---")
        falsas = listar_imagenes(RUTA_DATASETS / generador)
        particion_falsas = dividir(falsas, SEMILLA)
        guardar_listado(
            particion_falsas,
            RUTA_SALIDA / "listados" / f"{generador.lower()}.txt",
        )

        for split in SPLITS:
            base = RUTA_SALIDA / generador / split
            procesar_y_copiar(particion_falsas[split], base / "fake")
            procesar_y_copiar(particion_reales[split], base / "real")
            print(f"  {split}: fake={len(particion_falsas[split])}, "
                  f"real={len(particion_reales[split])} "
                  f"(procesadas a {TAMANO_OBJETIVO}x{TAMANO_OBJETIVO} PNG)")
        print()

    # 3) Verificación final de conteos
    print("=== Verificación ===")
    for generador in GENERADORES:
        for split in SPLITS:
            base = RUTA_SALIDA / generador / split
            n_fake = len(list((base / "fake").iterdir()))
            n_real = len(list((base / "real").iterdir()))
            print(f"{generador}/{split}: fake={n_fake}, real={n_real}")

    # 4) Resumen: total de imágenes por dataset
    print("\n=== Resumen por dataset ===")
    total_general = 0
    for generador in GENERADORES:
        total_fake = 0
        total_real = 0
        for split in SPLITS:
            base = RUTA_SALIDA / generador / split
            total_fake += len(list((base / "fake").iterdir()))
            total_real += len(list((base / "real").iterdir()))
        total_dataset = total_fake + total_real
        total_general += total_dataset
        print(f"{generador}: fake={total_fake} + real={total_real} "
              f"= {total_dataset} imágenes")
    print(f"\nTotal general en Particiones: {total_general} imágenes")

    print("\nParticiones creadas en:", RUTA_SALIDA)


if __name__ == "__main__":
    main()