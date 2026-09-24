"""
crear_particiones_multiclase.py
Crea las particiones train/val/test para el experimento MULTICLASE
(Flux vs SDXL vs StyleGAN2 vs StyleGAN3 vs real), donde la clase "real"
combina FFHQ + CelebA-HQ para no depender de las huellas de una sola
fuente de rostros reales.

Estructura de entrada (p. ej. C:\\ruta\\a\\Datasets):
    Flux/  SDXL/  StyleGAN2/  StyleGAN3/   7,000 imágenes c/u (sintéticas)
    Reales/            7,000 imágenes (FFHQ)
    Reales_Celeb_HQ/   3,500 imágenes (CelebA-HQ)

Estructura de salida:
    Particiones/Experimentos/Multiclase/<split>/<clase>/
    clases = Flux, SDXL, StyleGAN2, StyleGAN3, real

Balance: cada clase queda en 7,000 imágenes en total (4,900/1,050/1,050
train/val/test), incluida "real", que se arma con 3,500 FFHQ (submuestreadas
de las 7,000 disponibles, con la misma semilla) + 3,500 CelebA-HQ (todas),
mitad y mitad, para que el tamaño de "real" coincida con el de cada
generador sin necesidad de pesos de clase.

Dentro de real/, los archivos se prefijan (ffhq_ / celeba_) porque ambas
fuentes reutilizan el mismo esquema de nombres (00001.jpg, 00002.jpg, ...)
y colisionarían si se copiaran tal cual a la misma carpeta.

Mismo preprocesamiento y misma semilla que crear_particiones.py: recorte
central cuadrado, resize Lanczos a 256x256, e igualación de historial de
compresión (recompresión JPEG común) antes de guardar como PNG. Ver el
docstring de crear_particiones.py para la justificación metodológica
completa de ese preprocesamiento.

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
RUTA_SALIDA = RUTA_DATASETS / "Particiones" / "Experimentos" / "Multiclase"

GENERADORES = ["Flux", "SDXL", "StyleGAN2", "StyleGAN3"]
CARPETA_FFHQ = "Reales"
CARPETA_CELEBA = "Reales_Celeb_HQ"

SEMILLA = 42  # debe coincidir con la semilla de particionado en config.py

PROPORCIONES = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLITS = ["train", "val", "test"]

EXTENSIONES_VALIDAS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

TAMANO_OBJETIVO = 256
FORMATO_SALIDA = ".png"
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

def submuestrear(archivos: list[Path], objetivo: int, semilla: int) -> list[Path]:
    rng = random.Random(semilla)
    barajados = archivos.copy()
    rng.shuffle(barajados)
    return barajados[:objetivo]

def procesar_imagen(origen: Path, destino: Path) -> None:
    with Image.open(origen) as img:
        img = img.convert("RGB")

        ancho, alto = img.size
        lado = min(ancho, alto)
        izq = (ancho - lado) // 2
        arriba = (alto - lado) // 2
        img = img.crop((izq, arriba, izq + lado, arriba + lado))

        img = img.resize((TAMANO_OBJETIVO, TAMANO_OBJETIVO), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=CALIDAD_JPEG_COMUN)
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")

        img.save(destino, format="PNG")

def archivo_valido(ruta: Path) -> bool:
    """Verifica que un archivo ya existente sea PNG de 256x256."""
    try:
        with Image.open(ruta) as img:
            return (img.format == "PNG"
                    and img.size == (TAMANO_OBJETIVO, TAMANO_OBJETIVO))
    except Exception:
        return False

def procesar_y_copiar(archivos: list[Path], destino: Path, prefijo: str = "") -> int:
    """Procesa (256x256, RGB, PNG) y guarda cada archivo en destino, con un
    prefijo opcional en el nombre (para mezclar dos fuentes -FFHQ/CelebA-HQ-
    en la misma carpeta 'real' sin que los nombres choquen). Si el destino
    ya existe y es válido, no lo reprocesa. Devuelve cuántas procesó."""
    destino.mkdir(parents=True, exist_ok=True)

    procesadas = 0
    for archivo in archivos:
        nombre = f"{prefijo}{archivo.stem}"
        ruta_destino = destino / (nombre + FORMATO_SALIDA)
        if not (ruta_destino.exists() and archivo_valido(ruta_destino)):
            procesar_imagen(archivo, ruta_destino)
            procesadas += 1
    return procesadas


def limpiar_residuos(destino: Path, nombres_esperados: set[str]) -> None:
    if not destino.exists():
        return
    for residuo in destino.iterdir():
        if residuo.is_file() and residuo.name not in nombres_esperados:
            residuo.unlink()


def guardar_listado(particion: dict[str, list[Path]], ruta_txt: Path, prefijo: str = "") -> None:
    ruta_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_txt, "w", encoding="utf-8") as f:
        for split in SPLITS:
            for archivo in particion[split]:
                f.write(f"{split}\t{prefijo}{archivo.name}\n")


# --------------------------------- Main ----------------------------------
def main() -> None:
    print(f"Semilla: {SEMILLA}")
    print(f"Proporciones: {PROPORCIONES}\n")

    print("=== Imágenes disponibles por fuente ===")
    for carpeta in [CARPETA_FFHQ, CARPETA_CELEBA] + GENERADORES:
        n = len(listar_imagenes(RUTA_DATASETS / carpeta))
        print(f"  {carpeta}: {n:,} imágenes")
    print()

    # 1) Clase "real": FFHQ submuestreado al tamaño de CelebA-HQ + CelebA-HQ completo
    celeba = listar_imagenes(RUTA_DATASETS / CARPETA_CELEBA)
    ffhq_completo = listar_imagenes(RUTA_DATASETS / CARPETA_FFHQ)
    ffhq = submuestrear(ffhq_completo, len(celeba), SEMILLA)

    particion_ffhq = dividir(ffhq, SEMILLA)
    particion_celeba = dividir(celeba, SEMILLA)

    guardar_listado(particion_ffhq, RUTA_SALIDA / "listados" / "real_ffhq.txt", prefijo="ffhq_")
    guardar_listado(particion_celeba, RUTA_SALIDA / "listados" / "real_celeba.txt", prefijo="celeba_")

    print(f"--- real (FFHQ + CelebA-HQ) ---")
    print(f"  FFHQ: {len(ffhq)} (submuestreadas de {len(ffhq_completo)} disponibles)")
    print(f"  CelebA-HQ: {len(celeba)} (todas las disponibles)")
    for split in SPLITS:
        n_ffhq = len(particion_ffhq[split])
        n_celeba = len(particion_celeba[split])
        destino = RUTA_SALIDA / split / "real"
        esperados = {f"ffhq_{a.stem}{FORMATO_SALIDA}" for a in particion_ffhq[split]}
        esperados |= {f"celeba_{a.stem}{FORMATO_SALIDA}" for a in particion_celeba[split]}
        limpiar_residuos(destino, esperados)
        procesar_y_copiar(particion_ffhq[split], destino, prefijo="ffhq_")
        procesar_y_copiar(particion_celeba[split], destino, prefijo="celeba_")
        print(f"  {split}: ffhq={n_ffhq}, celeba={n_celeba}, total={n_ffhq + n_celeba}")
    print()

    # 2) Clases de generadores: cada una en su propia carpeta, solo sintéticas
    for generador in GENERADORES:
        print(f"--- {generador} ---")
        archivos = listar_imagenes(RUTA_DATASETS / generador)
        particion = dividir(archivos, SEMILLA)
        guardar_listado(particion, RUTA_SALIDA / "listados" / f"{generador.lower()}.txt")

        for split in SPLITS:
            destino = RUTA_SALIDA / split / generador
            esperados = {f"{a.stem}{FORMATO_SALIDA}" for a in particion[split]}
            limpiar_residuos(destino, esperados)
            procesar_y_copiar(particion[split], destino)
            print(f"  {split}: {len(particion[split])} imágenes")
        print()

    # 3) Verificación final de conteos
    print("=== Verificación final ===")
    clases = GENERADORES + ["real"]
    total_general = 0
    for split in SPLITS:
        conteos = []
        for clase in clases:
            n = len(list((RUTA_SALIDA / split / clase).iterdir()))
            total_general += n
            conteos.append(f"{clase}={n}")
        print(f"{split}: " + ", ".join(conteos))
    print(f"\nTotal general en Multiclase: {total_general} imágenes")

    print("\nParticiones multiclase creadas en:", RUTA_SALIDA)


if __name__ == "__main__":
    main()
