"""
generar_tablas_experimentoE.py
Genera tablas (CSV + Excel) con las metricas del Experimento E (aprendizaje
incremental / continual learning), a partir de los mismos 9 .json que usa
generar_figuras_experimentoE.py. No reentrena ni reevalua: solo lee y organiza.

Estructura del Exp. E: un modelo base (Exp. A, entrenado en StyleGAN2) que se
va afinando por fases -- FT-1 (+StyleGAN3), FT-2 (+SDXL), FT-3 (+Flux) -- y en
cada etapa se evalua sobre el test de los 4 generadores.

Metricas por etapa/generador/arquitectura/semilla:
  exactitud (accuracy), precision, sensibilidad (recall), especificidad,
  F1-Score, AUC-ROC y matriz de confusion (VP, FN, FP, VN).
La columna 'Visto' indica si ese generador ya habia entrado al entrenamiento
al llegar a esa etapa (True) o si aun era zero-shot (False).

Salidas en figuras/experimentoE/tablas:
  - tabla_por_semilla.csv  -> una fila por (arq, etapa, generador, semilla)
  - tabla_resumen.csv      -> media +/- desviacion estandar (3 semillas)
  - tabla_matrices.csv     -> matriz de confusion sumada (3 semillas)
  - tablas_experimentoE.xlsx -> las tres tablas en hojas separadas

Uso:
    python generar_tablas_experimentoE.py
"""

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
RUTA_METRICAS = BASE
RUTA_SALIDA = BASE / "figuras" / "experimentoE" / "tablas"

# ---------------------------------------------------------------------------
# NOMENCLATURA (coherente con generar_figuras_experimentoE.py)
# ---------------------------------------------------------------------------
NB = {"efficientnet_b0": "EfficientNet-B0",
      "resnet50": "ResNet-50",
      "legacy_xception": "Xception"}
ARQS = ["efficientnet_b0", "resnet50", "legacy_xception"]

GENS = ["StyleGAN2_CelebA", "StyleGAN3_CelebA", "SDXL_CelebA", "Flux_CelebA"]
GEN_LBL = {"StyleGAN2_CelebA": "StyleGAN2", "StyleGAN3_CelebA": "StyleGAN3",
           "SDXL_CelebA": "SDXL", "Flux_CelebA": "Flux"}

ETAPAS = ["Inicial", "FT-1", "FT-2", "FT-3"]
# Generadores que YA se han visto en entrenamiento al llegar a cada etapa
VISTOS = {
    "Inicial": {"StyleGAN2_CelebA"},
    "FT-1":    {"StyleGAN2_CelebA", "StyleGAN3_CelebA"},
    "FT-2":    {"StyleGAN2_CelebA", "StyleGAN3_CelebA", "SDXL_CelebA"},
    "FT-3":    {"StyleGAN2_CelebA", "StyleGAN3_CelebA", "SDXL_CelebA",
                "Flux_CelebA"},
}


# ---------------------------------------------------------------------------
# LECTURA  (misma logica que generar_figuras_experimentoE.py)
# ---------------------------------------------------------------------------
def cargar():
    """DATA[arq][semilla][etapa][generador] = dict de metricas."""
    DATA = {}
    for f in glob.glob(str(RUTA_METRICAS / "experimentoE_*.json")):
        name = Path(f).stem
        resto = name.replace("experimentoE_", "")
        arq, sem = resto.rsplit("_semilla", 1)
        sem = int(sem)
        d = json.load(open(f, encoding="utf-8"))
        etapas = {"Inicial": d["evaluacion_inicial"]}
        for fa in d["fases"]:
            etapas[fa["fase"]] = fa["evaluacion"]
        DATA.setdefault(arq, {})[sem] = etapas
    return DATA


DATA = cargar()
SEMILLAS = sorted(next(iter(DATA.values())).keys())


def desglose_matriz(mc):
    """Matriz [[fake->fake, fake->real],[real->fake, real->real]].
    Devuelve VP (fake ok), FN (fake->real), FP (real->fake), VN (real ok)."""
    vp, fn = mc[0][0], mc[0][1]
    fp, vn = mc[1][0], mc[1][1]
    return vp, fn, fp, vn


def especificidad(mc):
    _, _, fp, vn = desglose_matriz(mc)
    return vn / (vn + fp) if (vn + fp) else float("nan")


# ---------------------------------------------------------------------------
# CONSTRUCCION DE FILAS
# ---------------------------------------------------------------------------
def filas_por_semilla():
    filas = []
    for arq in ARQS:
        for etapa in ETAPAS:
            for gen in GENS:
                for semilla in SEMILLAS:
                    m = DATA[arq][semilla][etapa][gen]
                    mc = m["matriz_confusion"]
                    vp, fn, fp, vn = desglose_matriz(mc)
                    filas.append({
                        "Arquitectura": NB[arq],
                        "Etapa": etapa,
                        "Generador": GEN_LBL[gen],
                        "Visto": gen in VISTOS[etapa],
                        "Semilla": semilla,
                        "Exactitud": m["accuracy"],
                        "Precision": m["precision_fake"],
                        "Sensibilidad_Recall": m["recall_fake"],
                        "Especificidad": especificidad(mc),
                        "F1_Score": m["f1_fake"],
                        "AUC_ROC": m["auc"],
                        "VP_fake_fake": vp,
                        "FN_fake_real": fn,
                        "FP_real_fake": fp,
                        "VN_real_real": vn,
                    })
    return pd.DataFrame(filas)


METRICAS = ["Exactitud", "Precision", "Sensibilidad_Recall",
            "Especificidad", "F1_Score", "AUC_ROC"]


def tabla_resumen(df):
    """Media +/- desviacion estandar (ddof=1) por arq/etapa/generador."""
    filas = []
    for arq in ARQS:
        for etapa in ETAPAS:
            for gen in GENS:
                sub = df[(df["Arquitectura"] == NB[arq]) &
                         (df["Etapa"] == etapa) &
                         (df["Generador"] == GEN_LBL[gen])]
                fila = {"Arquitectura": NB[arq], "Etapa": etapa,
                        "Generador": GEN_LBL[gen],
                        "Visto": gen in VISTOS[etapa],
                        "N_semillas": len(sub)}
                for met in METRICAS:
                    fila[f"{met}_media"] = sub[met].mean()
                    fila[f"{met}_sd"] = sub[met].std(ddof=1)
                    fila[f"{met}_texto"] = (f"{sub[met].mean():.4f} "
                                            f"± {sub[met].std(ddof=1):.4f}")
                filas.append(fila)
    return pd.DataFrame(filas)


def tabla_matrices(df):
    """Matriz de confusion sumada sobre las 3 semillas por arq/etapa/gen."""
    filas = []
    for arq in ARQS:
        for etapa in ETAPAS:
            for gen in GENS:
                sub = df[(df["Arquitectura"] == NB[arq]) &
                         (df["Etapa"] == etapa) &
                         (df["Generador"] == GEN_LBL[gen])]
                filas.append({
                    "Arquitectura": NB[arq],
                    "Etapa": etapa,
                    "Generador": GEN_LBL[gen],
                    "Visto": gen in VISTOS[etapa],
                    "VP_fake_fake": int(sub["VP_fake_fake"].sum()),
                    "FN_fake_real": int(sub["FN_fake_real"].sum()),
                    "FP_real_fake": int(sub["FP_real_fake"].sum()),
                    "VN_real_real": int(sub["VN_real_real"].sum()),
                })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    df_semilla = filas_por_semilla()
    df_resumen = tabla_resumen(df_semilla)
    df_matrices = tabla_matrices(df_semilla)

    # CSVs (UTF-8 con BOM para que Excel abra bien los acentos)
    df_semilla.to_csv(RUTA_SALIDA / "tabla_por_semilla.csv",
                      index=False, encoding="utf-8-sig")
    df_resumen.to_csv(RUTA_SALIDA / "tabla_resumen.csv",
                      index=False, encoding="utf-8-sig")
    df_matrices.to_csv(RUTA_SALIDA / "tabla_matrices.csv",
                       index=False, encoding="utf-8-sig")

    # Excel con las tres tablas en hojas separadas
    xlsx = RUTA_SALIDA / "tablas_experimentoE.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df_semilla.to_excel(w, sheet_name="Por semilla", index=False)
        df_resumen.to_excel(w, sheet_name="Resumen (media±sd)", index=False)
        df_matrices.to_excel(w, sheet_name="Matrices confusion", index=False)

    print("Tablas generadas en:", RUTA_SALIDA)
    print("  -> tabla_por_semilla.csv    (", len(df_semilla), "filas )")
    print("  -> tabla_resumen.csv        (", len(df_resumen), "filas )")
    print("  -> tabla_matrices.csv       (", len(df_matrices), "filas )")
    print("  -> tablas_experimentoE.xlsx ( 3 hojas )")


if __name__ == "__main__":
    main()
