"""
generar_curvas_roc_por_arquitectura.py
Mismo dato que generar_curvas_roc.py (curvas ROC de los Experimentos A, B, C
y D), pero organizado al revés: UN panel por ARQUITECTURA, con los cuatro
experimentos superpuestos. Muestra cómo se degrada la curva de una misma
arquitectura conforme se aleja del generador de entrenamiento (StyleGAN2).

No vuelve a correr inferencia: reutiliza el caché en roc_cache/*.npz que ya
generó generar_curvas_roc.py (si no existe, ejecútalo primero).

Uso:
    C:\\Users\\Maria\\AppData\\Local\\Programs\\Python\\Python310\\python.exe generar_curvas_roc_por_arquitectura.py
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
RUTA_SALIDA = base.RUTA_SALIDA
ARQS = base.ARQS
NB = base.NB
SEMILLAS = base.SEMILLAS
EXPS = base.EXPS
TITULO_EXP = base.TITULO_EXP
NOMBRE_GEN_CORTO = base.NOMBRE_GEN_CORTO

# Un color por experimento, en degradado de "fácil/cercano" a "difícil/lejano"
# (A = mismo dominio de entrenamiento -> D = generador más distante).
COL_EXP = {"A": "tab:blue", "B": "tab:green", "C": "tab:orange", "D": "tab:red"}


def cargar_cache(arquitectura, semilla, exp):
    cache_f = RUTA_CACHE / f"{arquitectura}_semilla{semilla}_exp{exp}.npz"
    if not cache_f.exists():
        raise FileNotFoundError(
            f"Falta {cache_f.name}. Corre primero generar_curvas_roc.py."
        )
    d = np.load(cache_f)
    return d["y_true"], d["p_fake"]


def curva_media_exp(arquitectura, exp):
    curvas, aucs = [], []
    for semilla in SEMILLAS:
        y_true, p_fake = cargar_cache(arquitectura, semilla, exp)
        fpr, tpr, _ = roc_curve(y_true, p_fake)
        curvas.append((fpr, tpr))
        aucs.append(sk_auc(fpr, tpr))
    malla, tpr_m, tpr_sd = base.curva_media(curvas)
    return malla, tpr_m, tpr_sd, float(np.mean(aucs)), float(np.std(aucs))


def figura_arquitectura(arquitectura):
    fig, ax = plt.subplots(figsize=(5, 5))
    for exp in EXPS:
        malla, tpr_m, tpr_sd, auc_m, auc_sd = curva_media_exp(arquitectura, exp)
        etiqueta = f"{exp} — {NOMBRE_GEN_CORTO[exp]} (AUC = {auc_m:.3f} ± {auc_sd:.3f})"
        ax.plot(malla, tpr_m, color=COL_EXP[exp], lw=2, label=etiqueta)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.500)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad, clase sintética)")
    ax.set_title(f"{NB[arquitectura]} — degradación de la curva ROC "
                f"(A → B → C → D)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    nombre = f"roc_por_arquitectura_{arquitectura}.png"
    fig.savefig(RUTA_SALIDA / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> roc/{nombre}")


def figura_combinada():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    for ax, arquitectura in zip(axes, ARQS):
        for exp in EXPS:
            malla, tpr_m, _, auc_m, _ = curva_media_exp(arquitectura, exp)
            ax.plot(malla, tpr_m, color=COL_EXP[exp], lw=1.8,
                    label=f"{exp} — {NOMBRE_GEN_CORTO[exp]} ({auc_m:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_title(NB[arquitectura])
        ax.set_xlabel("Tasa de falsos positivos")
        ax.grid(alpha=0.3)
        if arquitectura == ARQS[0]:
            ax.set_ylabel("Tasa de verdaderos positivos")
            ax.legend(loc="lower right", fontsize=8, title="Experimento (AUC medio)")
    fig.suptitle("Curvas ROC por arquitectura — degradación A → B → C → D")
    fig.tight_layout()
    nombre = "roc_por_arquitectura_todas.png"
    fig.savefig(RUTA_SALIDA / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> roc/{nombre}")


if __name__ == "__main__":
    print("Generando curvas ROC por arquitectura (A, B, C, D superpuestas)...")
    for arquitectura in ARQS:
        figura_arquitectura(arquitectura)
    figura_combinada()
    print("Listo.")
