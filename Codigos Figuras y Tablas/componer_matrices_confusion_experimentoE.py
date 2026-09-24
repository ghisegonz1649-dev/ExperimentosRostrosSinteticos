"""
componer_matrices_confusion_experimentoE.py
Genera UNA sola figura con las 12 matrices de confusión del Experimento E
tras FT-3 (modelo final): filas = arquitecturas (EfficientNet-B0, ResNet-50,
Xception), columnas = generadores (StyleGAN2, StyleGAN3, SDXL, Flux).

Equivale a juntar las tres imágenes matriz_FT3_<arq>.png que produce
generar_figuras_experimentoE.py, pero sin repetir el título ni los rótulos:
los ejes sólo se rotulan en el borde de la rejilla y el nombre de cada
generador aparece únicamente en la fila superior.

No vuelve a evaluar: reutiliza los mismos .json (suma de las 3 semillas,
igual que las matrices individuales).

Salida en figuras/experimentoE/matrices/matriz_FT3_arquitecturas.png

Uso:
    python componer_matrices_confusion_experimentoE.py
"""

import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
RUTA_METRICAS = BASE
RUTA_SALIDA = BASE / "figuras" / "experimentoE" / "matrices"

# ---------------------------------------------------------------------------
# NOMENCLATURA (misma que generar_figuras_experimentoE.py)
# ---------------------------------------------------------------------------
NB = {"efficientnet_b0": "EfficientNet-B0",
      "resnet50": "ResNet-50",
      "legacy_xception": "Xception"}
ARQS = ["efficientnet_b0", "resnet50", "legacy_xception"]
CMAP_ARQ = {"efficientnet_b0": "Blues", "resnet50": "Oranges",
            "legacy_xception": "Greens"}

GENS = ["StyleGAN2_CelebA", "StyleGAN3_CelebA", "SDXL_CelebA", "Flux_CelebA"]
GEN_LBL = {"StyleGAN2_CelebA": "StyleGAN2", "StyleGAN3_CelebA": "StyleGAN3",
           "SDXL_CelebA": "SDXL", "Flux_CelebA": "Flux"}


# ---------------------------------------------------------------------------
# LECTURA
# ---------------------------------------------------------------------------
def cargar():
    """DATA[arq][semilla][etapa][generador] = dict de metricas."""
    DATA = {}
    for f in glob.glob(str(RUTA_METRICAS / "experimentoE_*.json")):
        name = Path(f).stem            # experimentoE_<arq>_semilla<s>
        resto = name.replace("experimentoE_", "")
        if "_semilla" not in resto:     # resumenes agregados: se ignoran
            continue
        arq, sem = resto.rsplit("_semilla", 1)
        sem = int(sem)
        d = json.load(open(f, encoding="utf-8"))
        etapas = {"Inicial": d["evaluacion_inicial"]}
        for fa in d["fases"]:
            etapas[fa["fase"]] = fa["evaluacion"]
        DATA.setdefault(arq, {})[sem] = {"etapas": etapas}
    return DATA


DATA = cargar()
SEMILLAS = sorted(next(iter(DATA.values())).keys())


def matriz_sumada(arq, gen):
    """Suma de las matrices de las 3 semillas (FT-3) y su versión por filas."""
    M = sum(np.array(DATA[arq][s]["etapas"]["FT-3"][gen]["matriz_confusion"])
            for s in SEMILLAS)
    tot = M.sum(axis=1, keepdims=True)
    prop = np.divide(M, tot, out=np.zeros_like(M, dtype=float), where=tot > 0)
    return M, prop


def dibujar_panel(ax, arq, gen, primera_fila, primera_columna, ultima_fila):
    M, prop = matriz_sumada(arq, gen)
    ax.imshow(prop, cmap=CMAP_ARQ[arq], vmin=0, vmax=1)

    ax.set_xticks([0, 1]); ax.set_xticklabels(["fake", "real"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["fake", "real"])
    if ultima_fila:
        ax.set_xlabel("Predicho")
    if primera_columna:
        ax.set_ylabel(f"{NB[arq]}\n\nReal", fontsize=11)
    if primera_fila:
        ax.set_title(GEN_LBL[gen], fontsize=12)

    for a in range(2):
        for b in range(2):
            c = "white" if prop[a, b] > 0.5 else "black"
            ax.text(b, a, f"{int(M[a, b])}\n({prop[a, b] * 100:.1f} %)",
                    ha="center", va="center", color=c,
                    fontweight="bold", fontsize=11, linespacing=1.4)


if __name__ == "__main__":
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(ARQS), len(GENS),
                             figsize=(4.2 * len(GENS), 4.3 * len(ARQS)))
    for i, arq in enumerate(ARQS):
        for j, gen in enumerate(GENS):
            dibujar_panel(axes[i, j], arq, gen,
                          primera_fila=(i == 0),
                          primera_columna=(j == 0),
                          ultima_fila=(i == len(ARQS) - 1))

    fig.suptitle("Matrices de confusión tras FT-3 (modelo final) — Exp. E "
                 "(suma de 3 semillas)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    nombre_salida = "matriz_FT3_arquitecturas.png"
    fig.savefig(RUTA_SALIDA / nombre_salida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> experimentoE/matrices/{nombre_salida}")
