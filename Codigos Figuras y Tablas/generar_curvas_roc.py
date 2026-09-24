"""
generar_curvas_roc.py
Genera las curvas ROC reales (FPR vs TPR) de los Experimentos A, B, C y D,
que faltaban: los .json de resultados solo guardan el valor escalar de AUC,
no las probabilidades por imagen necesarias para trazar la curva punto a punto.

Vuelve a cargar los 9 modelos ya entrenados en el Experimento A (3 arquitecturas
x 3 semillas) y corre SOLO inferencia (no reentrena nada) sobre los cuatro
conjuntos de prueba (StyleGAN2, StyleGAN3, SDXL, Flux), reutilizando nucleo.py
para que la evaluación (softmax, fake=clase positiva) sea idéntica a la que ya
produjo los .json existentes. Las probabilidades se cachean en .npz para no
tener que re-inferir cada vez que se quiera ajustar el estilo de la figura.

Nota: el Experimento E (adaptación progresiva) NO se incluye aquí porque los
pesos intermedios de las fases FT-1/FT-2/FT-3 no se conservaron localmente
(solo sus métricas agregadas en experimentoE_*.json). Para tener sus curvas
ROC habría que recuperar esos .pth de Drive o repetir el fine-tuning.

Uso (con el intérprete de Python donde estén instalados torch/timm/sklearn):
    python generar_curvas_roc.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve

BASE = Path(__file__).resolve().parent          # Codigos Figuras y Tablas/
RAIZ = BASE.parent                               # raíz del repo (ExperimentosRostrosSinteticos/)
sys.path.insert(0, str(RAIZ / "Codigos Experimentos"))
import config
import nucleo

# --- Rutas reales en disco (distintas de las que trae config.py por defecto) ---
config.RUTA_PARTICIONES = RAIZ / "Datasets" / "Particiones" / "Experimentos"
config.RUTA_MODELOS = BASE
config.NUM_WORKERS = 0   # evita problemas de multiprocessing en Windows

RUTA_CACHE = BASE / "roc_cache"
RUTA_SALIDA = BASE / "figuras" / "roc"
RUTA_TABLA_RESUMEN = BASE / "figuras" / "tablas" / "tabla_resumen.csv"
RUTA_CACHE.mkdir(parents=True, exist_ok=True)
RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

ARQS = ["efficientnet_b0", "resnet50", "legacy_xception"]
NB = {"efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet-50",
      "legacy_xception": "Xception"}
COL = {"efficientnet_b0": "tab:blue", "resnet50": "tab:orange",
       "legacy_xception": "tab:green"}
SEMILLAS = config.SEMILLAS_ENTRENAMIENTO        # [42, 123, 2024]
GENERADOR_ENTRENAMIENTO = "StyleGAN2_CelebA"
EXPS = ["A", "B", "C", "D"]

# Carpeta REAL en disco para cada experimento. Ojo: SDXL y Flux llevan guion
# bajo antes de la "A" ("SDXL_Celeb_A", "Flux_Celeb_A"), a diferencia de
# StyleGAN2/3 que no lo llevan ("StyleGAN2_CelebA"). Verificado contra
# Datasets/Particiones/Experimentos/ — no es un error de tipeo.
CARPETA_GENERADOR = {
    "A": "StyleGAN2_CelebA",
    "B": "StyleGAN3_CelebA",
    "C": "SDXL_Celeb_A",
    "D": "Flux_Celeb_A",
}
NOMBRE_GEN_CORTO = {"A": "StyleGAN2", "B": "StyleGAN3", "C": "SDXL", "D": "FLUX.1"}
TITULO_EXP = {
    "A": "Experimento A — Línea base (StyleGAN2)",
    "B": "Experimento B — StyleGAN3 (sin reentrenar)",
    "C": "Experimento C — SDXL (sin reentrenar)",
    "D": "Experimento D — FLUX.1 (sin reentrenar)",
}


# ---------------------------------------------------------------------------
# INFERENCIA
# ---------------------------------------------------------------------------
@torch.no_grad()
def obtener_scores(modelo, loader):
    """Igual que nucleo.evaluar(), pero devuelve los arreglos crudos (etiqueta
    binaria fake=1 / probabilidad de fake) en vez de métricas agregadas, que
    es lo que necesita roc_curve()."""
    modelo.eval()
    etiquetas, probas_fake = [], []
    for imagenes, y in loader:
        imagenes = imagenes.to(config.DISPOSITIVO, non_blocking=True)
        with torch.autocast(device_type=config.DISPOSITIVO.type,
                            enabled=config.USAR_AMP):
            salidas = modelo(imagenes)
        p = torch.softmax(salidas.float(), dim=1)
        etiquetas.extend(y.numpy())
        probas_fake.extend(p[:, 0].cpu().numpy())
    y_true = np.array(etiquetas)
    y_true_fake = (y_true == 0).astype(int)   # fake=0 en ImageFolder -> positiva=1
    return y_true_fake, np.array(probas_fake)


def cargar_modelo(arquitectura, semilla):
    nombre = f"{arquitectura}_{GENERADOR_ENTRENAMIENTO}_semilla{semilla}.pth"
    ruta = config.RUTA_MODELOS / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"No encuentro {ruta}")
    modelo = timm.create_model(arquitectura, pretrained=False,
                               num_classes=config.NUM_CLASES)
    modelo.load_state_dict(torch.load(ruta, map_location=config.DISPOSITIVO))
    return modelo.to(config.DISPOSITIVO)


def obtener_todas_las_probas(arquitectura, semilla):
    """Para una (arquitectura, semilla), devuelve {exp: (y_true, p_fake)} para
    los 4 experimentos, usando cache en disco cuando ya existe y cargando el
    modelo UNA sola vez para lo que falte."""
    resultados = {}
    faltan = []
    for exp in EXPS:
        cache_f = RUTA_CACHE / f"{arquitectura}_semilla{semilla}_exp{exp}.npz"
        if cache_f.exists():
            d = np.load(cache_f)
            resultados[exp] = (d["y_true"], d["p_fake"])
        else:
            faltan.append(exp)

    if faltan:
        print(f"  [{arquitectura} | semilla {semilla}] calculando: {faltan}")
        modelo = cargar_modelo(arquitectura, semilla)
        for exp in faltan:
            loader = nucleo.crear_dataloader(CARPETA_GENERADOR[exp], "test",
                                             semilla, barajar=False)
            y_true, p_fake = obtener_scores(modelo, loader)
            cache_f = RUTA_CACHE / f"{arquitectura}_semilla{semilla}_exp{exp}.npz"
            np.savez(cache_f, y_true=y_true, p_fake=p_fake)
            resultados[exp] = (y_true, p_fake)
        del modelo
        if config.DISPOSITIVO.type == "cuda":
            torch.cuda.empty_cache()

    return resultados


def cargar_auc_tabla_resumen():
    """Lee AUC_ROC_media/sd/texto desde tabla_resumen.csv (fuente oficial de
    las métricas agregadas del capítulo de resultados), para que las figuras
    muestren exactamente los mismos valores que la tabla en vez de un AUC
    recalculado aquí a partir de las curvas ROC cacheadas."""
    tabla = {}
    with open(RUTA_TABLA_RESUMEN, encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f):
            clave = (fila["Arquitectura"], fila["Experimento"])
            tabla[clave] = {
                "media": float(fila["AUC_ROC_media"]),
                "sd": float(fila["AUC_ROC_sd"]),
                "texto": fila["AUC_ROC_texto"],
            }
    return tabla


# ---------------------------------------------------------------------------
# CURVA ROC MEDIA ENTRE SEMILLAS
# ---------------------------------------------------------------------------
def curva_media(curvas_fpr_tpr, malla=None):
    """Promedio vertical de varias curvas ROC (una por semilla): interpola el
    TPR de cada curva sobre una malla común de FPR y promedia. Es el método
    estándar para reportar una curva ROC 'media' entre varias corridas."""
    if malla is None:
        malla = np.linspace(0, 1, 200)
    tprs_interp = [np.interp(malla, fpr, tpr) for fpr, tpr in curvas_fpr_tpr]
    tprs_interp = np.array(tprs_interp)
    return malla, tprs_interp.mean(axis=0), tprs_interp.std(axis=0)


def procesar(arquitectura, exp, cache_probas):
    """Curva ROC media ± sd entre las 3 semillas para (arquitectura, exp)."""
    curvas, aucs = [], []
    for semilla in SEMILLAS:
        y_true, p_fake = cache_probas[(arquitectura, semilla)][exp]
        fpr, tpr, _ = roc_curve(y_true, p_fake)
        curvas.append((fpr, tpr))
        aucs.append(sk_auc(fpr, tpr))
    malla, tpr_media, tpr_sd = curva_media(curvas)
    return malla, tpr_media, tpr_sd, float(np.mean(aucs)), float(np.std(aucs))


# ---------------------------------------------------------------------------
# FIGURAS
# ---------------------------------------------------------------------------
def figura_por_experimento(exp, cache_probas, tabla_auc):
    fig, ax = plt.subplots(figsize=(5, 5))
    for arq in ARQS:
        malla, tpr_m, tpr_sd, _, _ = procesar(arq, exp, cache_probas)
        auc_texto = tabla_auc[(NB[arq], exp)]["texto"]
        ax.plot(malla, tpr_m, color=COL[arq], lw=2,
                label=f"{NB[arq]} (AUC = {auc_texto})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.500)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad, clase sintética)")
    ax.set_title(TITULO_EXP[exp])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    nombre = f"roc_experimento_{exp}.png"
    fig.savefig(RUTA_SALIDA / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> roc/{nombre}")


def figura_combinada(cache_probas, tabla_auc):
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, exp in zip(axes.flat, EXPS):
        for arq in ARQS:
            malla, tpr_m, _, _, _ = procesar(arq, exp, cache_probas)
            auc_media = tabla_auc[(NB[arq], exp)]["media"]
            ax.plot(malla, tpr_m, color=COL[arq], lw=1.8,
                    label=f"{NB[arq]} ({auc_media:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_title(f"Experimento {exp} — {NOMBRE_GEN_CORTO[exp]}")
        ax.grid(alpha=0.3)
        if exp == "A":
            ax.legend(loc="lower right", fontsize=8, title="Arquitectura (AUC medio)")
    fig.supxlabel("Tasa de falsos positivos")
    fig.supylabel("Tasa de verdaderos positivos")
    fig.suptitle("Curvas ROC — media de 3 semillas por arquitectura y experimento")
    fig.tight_layout()
    nombre = "roc_todos_experimentos.png"
    fig.savefig(RUTA_SALIDA / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> roc/{nombre}")


# ---------------------------------------------------------------------------
# VERIFICACIÓN: el AUC recalculado aquí debe coincidir con el de los .json
# ---------------------------------------------------------------------------
def verificacion_contra_json(cache_probas):
    print("\nVerificación contra los AUC ya reportados en experimentoA_*.json:")
    ok_total = True
    for arq in ARQS:
        ruta = BASE / f"experimentoA_{arq}_{GENERADOR_ENTRENAMIENTO}.json"
        if not ruta.exists():
            print(f"  (no encontrado: {ruta.name})")
            continue
        d = json.load(open(ruta, encoding="utf-8"))
        for r in d["resultados_por_semilla"]:
            semilla = r["semilla"]
            auc_json = r["metricas_test"]["auc"]
            y_true, p_fake = cache_probas[(arq, semilla)]["A"]
            fpr, tpr, _ = roc_curve(y_true, p_fake)
            auc_recalc = sk_auc(fpr, tpr)
            diff = abs(auc_json - auc_recalc)
            ok = diff < 1e-3
            ok_total &= ok
            marca = "OK" if ok else "*** REVISAR ***"
            print(f"  {arq:16} semilla {semilla}: json={auc_json:.4f} "
                  f"recalculado={auc_recalc:.4f}  {marca}")
    if not ok_total:
        print("\n  AVISO: alguna verificación no coincide — revisar antes de "
              "confiar en las figuras generadas.")
    return ok_total


if __name__ == "__main__":
    print(f"Dispositivo: {config.DISPOSITIVO}")

    print("\nCargando/calculando probabilidades para las 3 arquitecturas x "
          "3 semillas x 4 experimentos...")
    cache_probas = {}
    for arq in ARQS:
        for semilla in SEMILLAS:
            cache_probas[(arq, semilla)] = obtener_todas_las_probas(arq, semilla)

    verificacion_contra_json(cache_probas)

    tabla_auc = cargar_auc_tabla_resumen()

    print("\nGenerando figuras ROC por experimento...")
    for exp in EXPS:
        figura_por_experimento(exp, cache_probas, tabla_auc)

    print("\nGenerando figura combinada...")
    figura_combinada(cache_probas, tabla_auc)

    print("\nListo.")
