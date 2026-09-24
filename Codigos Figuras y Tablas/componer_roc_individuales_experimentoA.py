"""
componer_roc_individuales_experimentoA.py
Genera una sola figura con las 3 curvas ROC individuales (EfficientNet-B0,
ResNet-50, Xception) de un experimento, una al lado de la otra.

A diferencia de generar_curvas_roc_individuales.py (que produce las 12
figuras estándar arquitectura x experimento con título largo y ejes
rotulados), esta versión es SOLO para el compuesto: título corto
("Arquitectura — Experimento A") y sin etiquetas de ejes. No modifica ni
depende de las imágenes individuales ya generadas.

No vuelve a correr inferencia: reutiliza el caché en roc_cache/*.npz que ya
generó generar_curvas_roc.py (si no existe, ejecútalo primero).

Uso:
    python componer_roc_individuales_experimentoA.py [A|B|C|D]   (por defecto A)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import generar_curvas_roc as base

RUTA_CACHE = base.RUTA_CACHE
ARQS = base.ARQS
NB = base.NB
COL = base.COL
SEMILLAS = base.SEMILLAS

RUTA_SALIDA = base.RUTA_SALIDA / "individuales"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

EXP = sys.argv[1].upper() if len(sys.argv) > 1 else "A"


def cargar_cache(arquitectura, semilla, exp):
    cache_f = RUTA_CACHE / f"{arquitectura}_semilla{semilla}_exp{exp}.npz"
    if not cache_f.exists():
        raise FileNotFoundError(
            f"Falta {cache_f.name}. Corre primero generar_curvas_roc.py."
        )
    d = np.load(cache_f)
    return d["y_true"], d["p_fake"]


def dibujar_panel(ax, arquitectura, tabla_auc):
    curvas, aucs = [], []
    for semilla in SEMILLAS:
        y_true, p_fake = cargar_cache(arquitectura, semilla, EXP)
        fpr, tpr, _ = roc_curve(y_true, p_fake)
        curvas.append((fpr, tpr))
        aucs.append(sk_auc(fpr, tpr))
    malla, tpr_m, _ = base.curva_media(curvas)
    auc_texto = tabla_auc[(NB[arquitectura], EXP)]["texto"]

    color = COL[arquitectura]
    for (fpr, tpr), semilla, auc_s in zip(curvas, SEMILLAS, aucs):
        ax.plot(fpr, tpr, color=color, lw=1, alpha=0.45,
                label=f"Semilla {semilla} (AUC = {auc_s:.3f})")
    ax.plot(malla, tpr_m, color=color, lw=2.5,
            label=f"Media (AUC = {auc_texto})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.500)")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title(f"{NB[arquitectura]} — Experimento {EXP}")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    tabla_auc = base.cargar_auc_tabla_resumen()

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, arquitectura in zip(axes, ARQS):
        dibujar_panel(ax, arquitectura, tabla_auc)
    fig.tight_layout()

    nombre_salida = f"roc_individuales_experimento{EXP}_arquitecturas.png"
    fig.savefig(RUTA_SALIDA / nombre_salida, dpi=200)
    plt.close(fig)
    print(f"-> roc/individuales/{nombre_salida}")
