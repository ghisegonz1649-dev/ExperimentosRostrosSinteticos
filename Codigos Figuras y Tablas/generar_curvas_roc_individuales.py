"""
generar_curvas_roc_individuales.py
Una figura POR CADA combinación (arquitectura, experimento) — 12 en total
(3 arquitecturas x 4 experimentos A-D) — en vez de las vistas superpuestas de
generar_curvas_roc.py (por experimento) y generar_curvas_roc_por_arquitectura.py
(por arquitectura). Cada figura muestra las 3 curvas de semilla individuales
(línea fina) más la curva media (línea gruesa), sin banda sombreada.

No vuelve a correr inferencia: reutiliza el caché en roc_cache/*.npz que ya
generó generar_curvas_roc.py (si no existe, ejecútalo primero).

Uso:
    C:\\Users\\Maria\\AppData\\Local\\Programs\\Python\\Python310\\python.exe generar_curvas_roc_individuales.py
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
EXPS = base.EXPS
TITULO_EXP = base.TITULO_EXP

RUTA_SALIDA = base.RUTA_SALIDA / "individuales"
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)


def cargar_cache(arquitectura, semilla, exp):
    cache_f = RUTA_CACHE / f"{arquitectura}_semilla{semilla}_exp{exp}.npz"
    if not cache_f.exists():
        raise FileNotFoundError(
            f"Falta {cache_f.name}. Corre primero generar_curvas_roc.py."
        )
    d = np.load(cache_f)
    return d["y_true"], d["p_fake"]


def figura_individual(arquitectura, exp, tabla_auc):
    curvas, aucs = [], []
    for semilla in SEMILLAS:
        y_true, p_fake = cargar_cache(arquitectura, semilla, exp)
        fpr, tpr, _ = roc_curve(y_true, p_fake)
        curvas.append((fpr, tpr))
        aucs.append(sk_auc(fpr, tpr))
    malla, tpr_m, _ = base.curva_media(curvas)
    auc_texto = tabla_auc[(NB[arquitectura], exp)]["texto"]

    fig, ax = plt.subplots(figsize=(5, 5))
    color = COL[arquitectura]
    for (fpr, tpr), semilla, auc_s in zip(curvas, SEMILLAS, aucs):
        ax.plot(fpr, tpr, color=color, lw=1, alpha=0.45,
                label=f"Semilla {semilla} (AUC = {auc_s:.3f})")
    ax.plot(malla, tpr_m, color=color, lw=2.5,
            label=f"Media (AUC = {auc_texto})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.500)")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad, clase sintética)")
    ax.set_title(f"{NB[arquitectura]} — {TITULO_EXP[exp]}")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    nombre = f"roc_{arquitectura}_experimento_{exp}.png"
    fig.savefig(RUTA_SALIDA / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> roc/individuales/{nombre}")


if __name__ == "__main__":
    print("Generando las 12 curvas ROC individuales (arquitectura x experimento)...")
    tabla_auc = base.cargar_auc_tabla_resumen()
    for arquitectura in ARQS:
        for exp in EXPS:
            figura_individual(arquitectura, exp, tabla_auc)
    print("Listo.")
