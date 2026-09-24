"""
generar_curva_roc_estructura.py
Genera una figura didactica que muestra la ESTRUCTURA de una curva ROC,
al estilo de las figuras conceptuales tipicas (lineas gruesas, colores
planos, sin caja de leyenda, flechas "Mejor"/"Peor"). Es un esquema
ilustrativo para el capitulo de Metodologia, no contiene datos reales de
ningun experimento.

Muestra:
    - Tres curvas ilustrativas con distinto desempeno (mejor -> peor)
    - La diagonal del clasificador aleatorio (AUC = 0.5)
    - El punto del clasificador perfecto (FPR=0, TPR=1)
    - Flechas indicando la direccion de "Mejor" / "Peor" desempeno

Salida: figuras/diagramas/curva_roc_estructura.png

Uso:
    python generar_curva_roc_estructura.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RUTA_SALIDA = BASE / "figuras" / "diagramas"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)


def main():
    fpr = np.linspace(0, 1, 200)
    # Curvas ilustrativas (no datos reales): TPR = FPR^(1/k), a mayor k, mas
    # se arquea la curva hacia la esquina superior izquierda (mejor modelo).
    curva_azul = fpr ** (1 / 6.0)   # excelente
    curva_verde = fpr ** (1 / 2.6)  # buena
    curva_naranja = fpr ** (1 / 1.5)  # mediocre

    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    # Trayectoria del clasificador perfecto: sube recto por FPR=0 hasta
    # TPR=1 y luego avanza recto por TPR=1 hasta FPR=1 (forma de "escuadra").
    ax.plot([0, 0, 1], [0, 1, 1], color="#2f8fe0", linewidth=2,
             linestyle=(0, (2, 2)), zorder=4, clip_on=False)

    ax.plot(fpr, curva_azul, color="#2f8fe0", linewidth=4.5, solid_capstyle="round")
    ax.plot(fpr, curva_verde, color="#2ecc71", linewidth=4.5, solid_capstyle="round")
    ax.plot(fpr, curva_naranja, color="#f5a623", linewidth=4.5, solid_capstyle="round")
    ax.plot([0, 1], [0, 1], color="#c0392b", linestyle="--", linewidth=3,
             dashes=(6, 4))

    # Clasificador perfecto
    ax.plot(0, 1, marker="o", markersize=14, color="#2f8fe0", zorder=5,
             clip_on=False)
    ax.text(-0.02, 1.05, "Clasificador\nperfecto", fontsize=16, color="#2f8fe0",
             ha="left", va="bottom", fontweight="bold", linespacing=1.05)

    ax.text(0.66, 1.10, "Curva ROC", fontsize=22, color="black",
             ha="center", va="center", fontweight="bold")

    # Flecha "Mejor"/"Peor" perpendicular a la diagonal, con las etiquetas
    # pegadas a cada punta para que quede claro a qué extremo corresponden.
    p_mejor = (0.50, 0.88)
    p_peor = (0.80, 0.52)
    ax.annotate("", xy=p_mejor, xytext=p_peor,
                 arrowprops=dict(arrowstyle="<->", color="black", lw=2.5))
    ax.text(p_mejor[0] - 0.03, p_mejor[1] + 0.03, "Mejor", fontsize=18,
             color="black", ha="right", va="bottom", fontweight="bold")
    ax.text(p_peor[0] + 0.03, p_peor[1] - 0.03, "Peor", fontsize=18,
             color="black", ha="left", va="top", fontweight="bold")

    ax.text(0.46, 0.46, "Clasificador\naleatorio", fontsize=13, color="#c0392b",
             ha="center", va="center", rotation=45, fontweight="bold")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.22)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0.0", "0.5", "1.0"], fontsize=17)
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0.0", "0.5", "1.0"], fontsize=17)
    ax.set_xlabel("Tasa de falsos positivos (FPR)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)", fontsize=14, fontweight="bold")
    # Centrar la etiqueta del eje Y sobre el rango de datos 0-1 (no sobre
    # el ylim completo, que incluye espacio extra arriba para el título).
    ax.yaxis.set_label_coords(-0.14, 0.5 / 1.22)
    ax.grid(color="0.85", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")

    fig.tight_layout()

    salida = RUTA_SALIDA / "curva_roc_estructura.png"
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f"Guardada: {salida}")


if __name__ == "__main__":
    main()
