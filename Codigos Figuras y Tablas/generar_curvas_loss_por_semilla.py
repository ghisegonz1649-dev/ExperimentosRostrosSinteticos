"""
generar_curvas_loss_por_semilla.py
Una sola imagen con el panel de Loss (train sólida / val punteada, una
línea por semilla) de las 3 arquitecturas puestas lado a lado — el mismo
dato que el panel izquierdo de figuras/curvas/curvas_<arq>.png (generadas
por generar_figuras_celeba.py), pero sin el panel de Accuracy y con las 3
arquitecturas juntas en una sola figura en vez de 3 archivos separados.

No vuelve a entrenar ni evaluar: solo lee los .json ya generados.

Uso:
    python generar_curvas_loss_por_semilla.py
"""

import json
from pathlib import Path

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


def dibujar_panel(ax, arq):
    ruta = BASE / f"experimentoA_{arq}_{GENERADOR_ENTRENAMIENTO}.json"
    d = json.load(open(ruta, encoding="utf-8"))
    col = plt.cm.tab10.colors

    for k, r in enumerate(d["resultados_por_semilla"]):
        h = r["historial"]
        ep = [x["epoca"] for x in h]
        c = col[k]
        s = r["semilla"]
        ax.plot(ep, [x["perdida_train"] for x in h], "-", color=c,
                label=f"train s{s}")
        ax.plot(ep, [x["perdida_val"] for x in h], "--", color=c,
                label=f"val s{s}")

    ax.set_title(f"{NB[arq]} — Experimento A")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, arq in zip(axes, ARQS):
        dibujar_panel(ax, arq)
    fig.tight_layout()

    nombre = "curvas_loss_por_semilla_arquitecturas.png"
    fig.savefig(RUTA_SALIDA / nombre, dpi=200)
    plt.close(fig)
    print(f"-> curvas/{nombre}")
