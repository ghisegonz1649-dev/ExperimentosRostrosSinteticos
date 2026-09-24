"""
componer_heatmap_confusion_ABCD.py
Misma información que componer_matrices_confusion_ABCD.py, pero con
presentación de MAPA DE CALOR: una sola escala de color compartida por las
12 matrices (0-100 % por fila) y una barra de color única a la derecha, en
lugar de un colormap distinto por experimento.

Filas = arquitecturas (EfficientNet-B0, ResNet-50, Xception).
Columnas = experimentos (A/StyleGAN2, B/StyleGAN3, C/SDXL, D/Flux).

No vuelve a evaluar: reutiliza los mismos .json que lee
generar_figuras_celeba.py (suma de las 3 semillas).

Uso:
    python componer_heatmap_confusion_ABCD.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import generar_figuras_celeba as base
from componer_matrices_confusion_ABCD import matriz_sumada

ARQS = base.ARQS
EXPS = base.EXPS
NB = base.NB
GEN = base.GEN

RUTA_SALIDA = base.RUTA_MATRICES
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

CMAP = "RdYlBu_r"   # escala única para todas las celdas


def dibujar_panel(ax, arq, exp, primera_fila, primera_columna, ultima_fila):
    M, prop = matriz_sumada(arq, exp)
    im = ax.imshow(prop * 100, cmap=CMAP, vmin=0, vmax=100)

    ax.set_xticks([0, 1]); ax.set_xticklabels(["fake", "real"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["fake", "real"])
    if ultima_fila:
        ax.set_xlabel("Predicho")
    if primera_columna:
        ax.set_ylabel(f"{NB[arq]}\n\nReal", fontsize=11)
    if primera_fila:
        ax.set_title(f"Experimento {exp}\n({GEN[exp]})", fontsize=12)

    # separación tipo mapa de calor entre celdas
    ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
    ax.set_yticks([-0.5, 0.5, 1.5], minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)

    for a in range(2):
        for b in range(2):
            v = prop[a, b] * 100
            c = "white" if v > 65 or v < 12 else "black"
            ax.text(b, a, f"{int(M[a, b])}\n({v:.1f} %)",
                    ha="center", va="center", color=c,
                    fontweight="bold", fontsize=11, linespacing=1.4)
    return im


if __name__ == "__main__":
    fig, axes = plt.subplots(len(ARQS), len(EXPS),
                             figsize=(4.2 * len(EXPS) + 0.8, 4.3 * len(ARQS)))
    for i, arq in enumerate(ARQS):
        for j, exp in enumerate(EXPS):
            im = dibujar_panel(axes[i, j], arq, exp,
                               primera_fila=(i == 0),
                               primera_columna=(j == 0),
                               ultima_fila=(i == len(ARQS) - 1))

    fig.suptitle("Matrices de confusión — Experimentos A-D "
                 "(entrenamiento en StyleGAN2, suma de 3 semillas)",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 0.93, 0.97))

    cax = fig.add_axes((0.945, 0.08, 0.014, 0.82))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Porcentaje de la clase real (%)", fontsize=11)

    nombre_salida = "matriz_general_ABCD_heatmap.png"
    fig.savefig(RUTA_SALIDA / nombre_salida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> matrices/{nombre_salida}")
