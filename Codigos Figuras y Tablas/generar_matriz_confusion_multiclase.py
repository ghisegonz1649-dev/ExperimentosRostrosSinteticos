"""
generar_matriz_confusion_multiclase.py
Genera la matriz de confusion del experimento multiclase (atribucion de
generador a 5 clases: Flux, SDXL, StyleGAN2, StyleGAN3, real) a partir de
JSON Experimentos/Multiclase/experimentoMulticlase_efficientnet_b0.json.

Sigue la misma convencion visual que las matrices de los experimentos A-D
(Datasets/graficos_metricas.py): imshow con colormap Blues, texto blanco/negro
segun intensidad, ejes "Predicho"/"Real".

Salida: figuras/multiclase/matriz_confusion_multiclase_efficientnet_b0.png

Uso:
    python generar_matriz_confusion_multiclase.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RUTA_JSON = (
    BASE.parent / "JSON Experimentos" / "Multiclase"
    / "experimentoMulticlase_efficientnet_b0.json"
)
RUTA_SALIDA = BASE / "figuras" / "multiclase"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)


def _dibujar_matriz(ax, m: np.ndarray, clases: list[str], titulo: str):
    ax.imshow(m, cmap="Blues")
    n = len(clases)
    etiquetas = [c if c != "real" else "Real" for c in clases]
    ax.set_xticks(range(n)); ax.set_xticklabels(etiquetas, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(etiquetas)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=11)
    vmax = m.max()
    for i in range(n):
        for j in range(n):
            color = "white" if m[i, j] > vmax / 2 else "black"
            ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center",
                     color=color, fontsize=11, fontweight="bold")


def main():
    d = json.load(open(RUTA_JSON, encoding="utf-8"))
    r = d["resultados_por_semilla"][0]
    metricas = r["metricas_test"]
    clases = metricas["clases"]
    m = np.array(metricas["matriz_confusion"])
    acc = metricas["accuracy"]

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    _dibujar_matriz(ax, m, clases,
                     f"EfficientNet-B0 — semilla {r['semilla']} "
                     f"(accuracy = {acc:.2%})")
    fig.suptitle("Experimento multiclase — atribución de generador (5 clases)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    salida = RUTA_SALIDA / "matriz_confusion_multiclase_efficientnet_b0.png"
    fig.savefig(salida, dpi=130)
    plt.close(fig)
    print(f"Guardada: {salida}")


if __name__ == "__main__":
    main()
