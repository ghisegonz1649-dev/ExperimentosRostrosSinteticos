"""
componer_matrices_confusion_ABCD.py
Genera UNA sola figura general con las 12 matrices de confusión de los
experimentos A-D: filas = arquitecturas (EfficientNet-B0, ResNet-50,
Xception), columnas = experimentos (A/StyleGAN2, B/StyleGAN3, C/SDXL,
D/Flux).

A diferencia de generar_figuras_celeba.py (que produce las 12 matrices
individuales y las 4 comparativas por experimento), esta versión es SOLO
para el panel general: rótulos de eje sólo en el borde de la rejilla y
título de columna únicamente en la fila superior. No modifica ni depende
de las imágenes ya generadas.

No vuelve a evaluar: reutiliza los mismos .json que lee
generar_figuras_celeba.py (suma de las 3 semillas, igual que las otras
matrices).

Uso:
    python componer_matrices_confusion_ABCD.py
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

ARQS = base.ARQS
EXPS = base.EXPS
NB = base.NB
GEN = base.GEN
CMAP = base.CMAP

RUTA_SALIDA = base.RUTA_MATRICES
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)


def matriz_sumada(arq, exp):
    """Suma de las matrices de las 3 semillas y su versión por filas."""
    dic = base.fuente(arq, exp)
    M = sum(np.array(m["matriz_confusion"]) for m in dic.values())
    tot = M.sum(axis=1, keepdims=True)
    prop = np.divide(M, tot, out=np.zeros_like(M, dtype=float), where=tot > 0)
    return M, prop


def dibujar_panel(ax, arq, exp, primera_fila, primera_columna, ultima_fila):
    M, prop = matriz_sumada(arq, exp)
    ax.imshow(prop, cmap=CMAP[exp], vmin=0, vmax=1)

    ax.set_xticks([0, 1]); ax.set_xticklabels(["fake", "real"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["fake", "real"])
    if ultima_fila:
        ax.set_xlabel("Predicho")
    if primera_columna:
        ax.set_ylabel(f"{NB[arq]}\n\nReal", fontsize=11)
    if primera_fila:
        ax.set_title(f"Experimento {exp}\n({GEN[exp]})", fontsize=12)

    for a in range(2):
        for b in range(2):
            c = "white" if prop[a, b] > 0.5 else "black"
            ax.text(b, a, f"{int(M[a, b])}\n({prop[a, b] * 100:.1f} %)",
                    ha="center", va="center", color=c,
                    fontweight="bold", fontsize=11, linespacing=1.4)


if __name__ == "__main__":
    fig, axes = plt.subplots(len(ARQS), len(EXPS),
                             figsize=(4.2 * len(EXPS), 4.3 * len(ARQS)))
    for i, arq in enumerate(ARQS):
        for j, exp in enumerate(EXPS):
            dibujar_panel(axes[i, j], arq, exp,
                          primera_fila=(i == 0),
                          primera_columna=(j == 0),
                          ultima_fila=(i == len(ARQS) - 1))

    fig.suptitle("Matrices de confusión — Experimentos A-D "
                 "(entrenamiento en StyleGAN2, suma de 3 semillas)",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    nombre_salida = "matriz_general_experimentos_ABCD.png"
    fig.savefig(RUTA_SALIDA / nombre_salida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> matrices/{nombre_salida}")
