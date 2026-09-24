"""
generar_tablas_celeba.py
Genera tablas (CSV + Excel) con las métricas de todos los experimentos de la
tanda CelebA-HQ, a partir de los mismos .json que usa generar_figuras_celeba.py.
No reentrena ni reevalúa: solo lee y organiza.

Métricas por experimento/arquitectura/semilla:
  exactitud (accuracy), precisión, sensibilidad (recall), especificidad,
  F1-Score, AUC-ROC y matriz de confusión (VP, FN, FP, VN).

Salidas en RUTA_SALIDA/tablas:
  - tabla_por_semilla.csv     -> una fila por (arquitectura, experimento, semilla)
  - tabla_resumen.csv         -> media ± desviación estándar (3 semillas)
  - tabla_matrices.csv        -> matriz de confusión sumada (3 semillas)
  - tablas_celeba.xlsx        -> las tres tablas en hojas separadas

Uso:
    python generar_tablas_celeba.py
"""

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# RUTAS  (los .json están en la misma carpeta que este script)
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
RUTA_METRICAS = BASE
RUTA_SALIDA = BASE / "figuras" / "tablas"

NB = {"efficientnet_b0": "EfficientNet-B0",
      "resnet50": "ResNet-50",
      "legacy_xception": "Xception"}
ARQS = ["efficientnet_b0", "resnet50", "legacy_xception"]
EXPS = ["A", "B", "C", "D"]
GEN = {"A": "StyleGAN2", "B": "StyleGAN3", "C": "SDXL", "D": "Flux"}


# ---------------------------------------------------------------------------
# LECTURA  (misma lógica que generar_figuras_celeba.py)
# ---------------------------------------------------------------------------
def cargar():
    A, BCD = {}, {}
    for f in glob.glob(str(RUTA_METRICAS / "experimentoA_*.json")):
        if "_semilla" in Path(f).name:        # ignora los .json por-semilla
            continue
        d = json.load(open(f, encoding="utf-8"))
        A[d["modelo"]] = {r["semilla"]: r["metricas_test"]
                          for r in d["resultados_por_semilla"]}
    for f in glob.glob(str(RUTA_METRICAS / "experimentosBCD_*.json")):
        d = json.load(open(f, encoding="utf-8"))
        for e in d["evaluaciones"]:
            (BCD.setdefault(e["arquitectura"], {})
                .setdefault(e["experimento"], {})[e["semilla"]]) = e["metricas"]
    return A, BCD


A, BCD = cargar()


def fuente(arq, exp):
    return A[arq] if exp == "A" else BCD[arq][exp]


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
# CONSTRUCCIÓN DE FILAS
# ---------------------------------------------------------------------------
def filas_por_semilla():
    filas = []
    for arq in ARQS:
        for exp in EXPS:
            for semilla, m in sorted(fuente(arq, exp).items()):
                mc = m["matriz_confusion"]
                vp, fn, fp, vn = desglose_matriz(mc)
                filas.append({
                    "Arquitectura": NB[arq],
                    "Experimento": exp,
                    "Generador": GEN[exp],
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
    """Media ± desviación estándar (ddof=1) por arquitectura y experimento."""
    filas = []
    for arq in ARQS:
        for exp in EXPS:
            sub = df[(df["Arquitectura"] == NB[arq]) &
                     (df["Experimento"] == exp)]
            fila = {"Arquitectura": NB[arq], "Experimento": exp,
                    "Generador": GEN[exp], "N_semillas": len(sub)}
            for met in METRICAS:
                fila[f"{met}_media"] = sub[met].mean()
                fila[f"{met}_sd"] = sub[met].std(ddof=1)
                fila[f"{met}_texto"] = (f"{sub[met].mean():.4f} "
                                        f"± {sub[met].std(ddof=1):.4f}")
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_matrices(df):
    """Matriz de confusión sumada sobre las 3 semillas por arq/experimento."""
    filas = []
    for arq in ARQS:
        for exp in EXPS:
            sub = df[(df["Arquitectura"] == NB[arq]) &
                     (df["Experimento"] == exp)]
            filas.append({
                "Arquitectura": NB[arq],
                "Experimento": exp,
                "Generador": GEN[exp],
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
    xlsx = RUTA_SALIDA / "tablas_celeba.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df_semilla.to_excel(w, sheet_name="Por semilla", index=False)
        df_resumen.to_excel(w, sheet_name="Resumen (media±sd)", index=False)
        df_matrices.to_excel(w, sheet_name="Matrices confusion", index=False)

    print("Tablas generadas en:", RUTA_SALIDA)
    print("  -> tabla_por_semilla.csv  (", len(df_semilla), "filas )")
    print("  -> tabla_resumen.csv      (", len(df_resumen), "filas )")
    print("  -> tabla_matrices.csv     (", len(df_matrices), "filas )")
    print("  -> tablas_celeba.xlsx     ( 3 hojas )")


if __name__ == "__main__":
    main()
