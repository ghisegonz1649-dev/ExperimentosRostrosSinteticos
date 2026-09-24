"""
componer_curvas_experimentoE.py
Genera UNA sola imagen por fase del Experimento E (FT-1, FT-2, FT-3) con las
tres arquitecturas juntas: filas = arquitectura, columnas = Loss / Accuracy.

Lee los mismos .json que generar_figuras_experimentoE.py (no reentrena nada).

Salida en figuras/experimentoE/curvas/curvas_E_<fase>_arquitecturas.png

Uso:
    python componer_curvas_experimentoE.py
"""

import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
RUTA_METRICAS = BASE
RUTA_CURVAS = BASE / "figuras" / "experimentoE" / "curvas"

# ---------------------------------------------------------------------------
# NOMENCLATURA
# ---------------------------------------------------------------------------
NB = {"efficientnet_b0": "EfficientNet-B0",
      "resnet50": "ResNet-50",
      "legacy_xception": "Xception"}
ARQS = ["efficientnet_b0", "resnet50", "legacy_xception"]
FASES = ["FT-1", "FT-2", "FT-3"]
# Generador que entra al entrenamiento en cada fase (para el subtitulo)
NUEVO_GEN = {"FT-1": "StyleGAN3", "FT-2": "SDXL", "FT-3": "Flux"}


# ---------------------------------------------------------------------------
# LECTURA
# ---------------------------------------------------------------------------
def cargar():
    """DATA[arq][semilla]['fases'] = lista de fases con su historial."""
    DATA = {}
    for f in glob.glob(str(RUTA_METRICAS / "experimentoE_*.json")):
        name = Path(f).stem            # experimentoE_<arq>_semilla<s>
        resto = name.replace("experimentoE_", "")
        if "_semilla" not in resto:     # resumenes agregados: no llevan historial
            continue
        arq, sem = resto.rsplit("_semilla", 1)
        sem = int(sem)
        d = json.load(open(f, encoding="utf-8"))
        DATA.setdefault(arq, {})[sem] = {"fases": d["fases"]}
    return DATA


DATA = cargar()
SEMILLAS = sorted(next(iter(DATA.values())).keys())


# ===========================================================================
# UNA IMAGEN POR FASE CON LAS 3 ARQUITECTURAS
# ===========================================================================
def figura_fase_arquitecturas(fase):
    RUTA_CURVAS.mkdir(parents=True, exist_ok=True)
    col = plt.cm.tab10.colors
    fig, axes = plt.subplots(len(ARQS), 2, figsize=(13, 13.5))

    for i, arq in enumerate(ARQS):
        ax_loss, ax_acc = axes[i]
        for k, s in enumerate(SEMILLAS):
            fa = next(f for f in DATA[arq][s]["fases"] if f["fase"] == fase)
            h = fa["historial"]
            ep = [x["epoca"] for x in h]
            c = col[k]
            ax_loss.plot(ep, [x["perdida_train"] for x in h], "-", color=c)
            ax_loss.plot(ep, [x["perdida_val"] for x in h], "--", color=c)
            ax_acc.plot(ep, [x["acc_train"] for x in h], "-", color=c)
            ax_acc.plot(ep, [x["acc_val"] for x in h], "--", color=c)

        ax_loss.set_xlabel("Época"); ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"{NB[arq]} — Loss")
        ax_loss.grid(alpha=0.3)
        ax_acc.set_xlabel("Época"); ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title(f"{NB[arq]} — Accuracy")
        ax_acc.grid(alpha=0.3)

    # Leyenda unica para toda la figura: color = semilla, estilo = train/val
    handles = [Line2D([], [], color=col[k], lw=2, label=f"Semilla {s}")
               for k, s in enumerate(SEMILLAS)]
    handles += [Line2D([], [], color="gray", lw=2, ls="-", label="Train"),
                Line2D([], [], color="gray", lw=2, ls="--", label="Validación")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(f"Curvas de aprendizaje — Exp. E — {fase} "
                 f"(entra {NUEVO_GEN[fase]})\n"
                 "Sólida = train, punteada = val (una línea por semilla)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    nombre = f"curvas_E_{fase}_arquitecturas.png"
    fig.savefig(RUTA_CURVAS / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> experimentoE/curvas/{nombre}")


def main():
    print("Curvas de aprendizaje Exp. E (una imagen por fase, 3 arquitecturas):")
    for fase in FASES:
        figura_fase_arquitecturas(fase)
    print(f"\nListo. Todo en: {RUTA_CURVAS}")


if __name__ == "__main__":
    main()
