"""
generar_curvas_loss_comparativa.py
Una sola figura comparando la curva de pérdida (loss) de entrenamiento y
validación de las 3 arquitecturas (Experimento A, StyleGAN2_CelebA), en vez
de las figuras individuales por arquitectura que ya genera
generar_figuras_celeba.py (figuras/curvas/curvas_<arq>.png).

Para cada arquitectura se promedia la pérdida entre las 3 semillas, época a
época, truncando en la época mínima alcanzada por cualquiera de las 3 (el
early stopping detiene a cada semilla en una época distinta).

No vuelve a entrenar ni evaluar: solo lee los .json ya generados.

Uso:
    python generar_curvas_loss_comparativa.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RUTA_SALIDA = BASE / "figuras" / "curvas"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

GENERADOR_ENTRENAMIENTO = "StyleGAN2_CelebA"
ARQS = ["efficientnet_b0", "resnet50", "legacy_xception"]
NB = {"efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet-50",
      "legacy_xception": "Xception"}
COL = {"efficientnet_b0": "tab:blue", "resnet50": "tab:orange",
       "legacy_xception": "tab:green"}


def perdida_media(arq):
    """Pérdida media train/val entre las 3 semillas, truncada a la época
    mínima común (evita extrapolar más allá de una semilla que ya paró por
    early stopping)."""
    ruta = BASE / f"experimentoA_{arq}_{GENERADOR_ENTRENAMIENTO}.json"
    d = json.load(open(ruta, encoding="utf-8"))
    historiales = [r["historial"] for r in d["resultados_por_semilla"]]
    n_min = min(len(h) for h in historiales)
    train = np.array([[h[i]["perdida_train"] for h in historiales]
                       for i in range(n_min)])
    val = np.array([[h[i]["perdida_val"] for h in historiales]
                     for i in range(n_min)])
    epocas = np.arange(1, n_min + 1)
    return epocas, train.mean(axis=1), val.mean(axis=1)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for arq in ARQS:
        epocas, train_m, val_m = perdida_media(arq)
        color = COL[arq]
        ax.plot(epocas, train_m, "-", color=color, lw=2,
                label=f"{NB[arq]} — train")
        ax.plot(epocas, val_m, "--", color=color, lw=2,
                label=f"{NB[arq]} — val")

    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida (loss)")
    ax.set_title("Curvas de pérdida por arquitectura — Experimento A "
                  "(media de 3 semillas)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    nombre = "curvas_loss_comparativa_arquitecturas.png"
    fig.savefig(RUTA_SALIDA / nombre, dpi=200)
    plt.close(fig)
    print(f"-> curvas/{nombre}")
