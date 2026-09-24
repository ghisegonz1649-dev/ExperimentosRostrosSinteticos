"""
generar_curvas_roc_experimentoE.py
Extiende generar_curvas_roc.py al Experimento E (adaptación progresiva):
curvas ROC reales para el modelo base y para cada fase de fine-tuning
(FT-1, FT-2, FT-3), evaluadas sobre los cuatro generadores.

Requiere los .pth de cada fase, nombrados como los guarda experimento_E.py /
el notebook de Colab: "<arquitectura>_E_FT-<n>_semilla<semilla>.pth", en la
misma carpeta que este script. Si faltan para alguna arquitectura, esa
arquitectura simplemente se omite (se avisa por consola) — vuelve a correr
este script cuando los descargues, no hace falta tocar nada más.

La curva de la "Fase 0" (modelo del Experimento A, antes de cualquier
fine-tuning) NO se recalcula: es el mismo modelo y los mismos datos que ya
se procesaron en generar_curvas_roc.py, así que se reutiliza su caché
(roc_cache/<arq>_semilla<semilla>_exp{A,B,C,D}.npz).

Uso:
    C:\\Users\\Maria\\AppData\\Local\\Programs\\Python\\Python310\\python.exe generar_curvas_roc_experimentoE.py
"""

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

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
# Reutiliza rutas, helpers y convenciones ya definidos y probados en el
# script de A-D (config.RUTA_PARTICIONES/RUTA_MODELOS, obtener_scores,
# curva_media, CARPETA_GENERADOR, colores, etc.)
import generar_curvas_roc as base

config = base.config
nucleo = base.nucleo

RUTA_CACHE = base.RUTA_CACHE
RUTA_SALIDA = base.RUTA_SALIDA
ARQS = base.ARQS
NB = base.NB
COL = base.COL
SEMILLAS = base.SEMILLAS
CARPETA_GENERADOR = base.CARPETA_GENERADOR   # {"A":"StyleGAN2_CelebA", "B":"StyleGAN3_CelebA", ...}

# Las claves de generador dentro de experimentoE_*.json (nombres "lógicos",
# no las carpetas reales en disco) -> letra de experimento A/B/C/D, para
# reutilizar CARPETA_GENERADOR y así llegar a la ruta real.
LETRA_DE_GENERADOR_JSON = {
    "StyleGAN2_CelebA": "A",
    "StyleGAN3_CelebA": "B",
    "SDXL_CelebA": "C",
    "Flux_CelebA": "D",
}
FASES = ["FT-1", "FT-2", "FT-3"]
COL_GEN = {"StyleGAN2_CelebA": "tab:blue", "StyleGAN3_CelebA": "tab:green",
          "SDXL_CelebA": "tab:red", "Flux_CelebA": "tab:purple"}
NB_GEN = {"StyleGAN2_CelebA": "StyleGAN2", "StyleGAN3_CelebA": "StyleGAN3",
         "SDXL_CelebA": "SDXL", "Flux_CelebA": "FLUX.1"}


def arquitecturas_disponibles():
    """Solo las arquitecturas con los 9 checkpoints de fine-tuning (3 fases x
    3 semillas) presentes en disco."""
    disponibles = []
    for arq in ARQS:
        faltantes = [f"{arq}_E_{fase}_semilla{s}.pth"
                    for fase in FASES for s in SEMILLAS
                    if not (BASE / f"{arq}_E_{fase}_semilla{s}.pth").exists()]
        if faltantes:
            print(f"  [omitido] {arq}: faltan {len(faltantes)}/9 checkpoints "
                  f"de Experimento E (p. ej. {faltantes[0]})")
        else:
            disponibles.append(arq)
    return disponibles


def cargar_modelo_fase(arquitectura, fase, semilla):
    nombre = f"{arquitectura}_E_{fase}_semilla{semilla}.pth"
    ruta = BASE / nombre
    modelo = timm.create_model(arquitectura, pretrained=False,
                               num_classes=config.NUM_CLASES)
    modelo.load_state_dict(torch.load(ruta, map_location=config.DISPOSITIVO))
    return modelo.to(config.DISPOSITIVO)


def probas_fase(arquitectura, fase, semilla):
    """{generador_json: (y_true, p_fake)} para una (arquitectura, fase,
    semilla), evaluando sobre los 4 generadores. Cachea en disco."""
    resultados = {}
    faltan = []
    for gen_json in LETRA_DE_GENERADOR_JSON:
        cache_f = RUTA_CACHE / f"{arquitectura}_semilla{semilla}_E-{fase}_{gen_json}.npz"
        if cache_f.exists():
            d = np.load(cache_f)
            resultados[gen_json] = (d["y_true"], d["p_fake"])
        else:
            faltan.append(gen_json)

    if faltan:
        modelo = cargar_modelo_fase(arquitectura, fase, semilla)
        for gen_json in faltan:
            letra = LETRA_DE_GENERADOR_JSON[gen_json]
            carpeta_disco = CARPETA_GENERADOR[letra]
            loader = nucleo.crear_dataloader(carpeta_disco, "test", semilla,
                                             barajar=False)
            y_true, p_fake = base.obtener_scores(modelo, loader)
            cache_f = RUTA_CACHE / f"{arquitectura}_semilla{semilla}_E-{fase}_{gen_json}.npz"
            np.savez(cache_f, y_true=y_true, p_fake=p_fake)
            resultados[gen_json] = (y_true, p_fake)
        del modelo
        if config.DISPOSITIVO.type == "cuda":
            torch.cuda.empty_cache()

    return resultados


def probas_fase0(arquitectura, semilla):
    """Fase 0 = modelo del Experimento A, ya cacheado por generar_curvas_roc.py
    bajo las claves de letra A/B/C/D. Se remapea a claves de nombre de
    generador para tener el mismo formato que las fases FT-*."""
    resultados = {}
    for gen_json, letra in LETRA_DE_GENERADOR_JSON.items():
        cache_f = RUTA_CACHE / f"{arquitectura}_semilla{semilla}_exp{letra}.npz"
        if not cache_f.exists():
            raise FileNotFoundError(
                f"Falta {cache_f.name}: corre primero generar_curvas_roc.py "
                f"para {arquitectura}."
            )
        d = np.load(cache_f)
        resultados[gen_json] = (d["y_true"], d["p_fake"])
    return resultados


def curva_media_generador(arquitectura, fase, gen_json, cache_por_semilla):
    curvas, aucs = [], []
    for semilla in SEMILLAS:
        y_true, p_fake = cache_por_semilla[semilla][gen_json]
        fpr, tpr, _ = roc_curve(y_true, p_fake)
        curvas.append((fpr, tpr))
        aucs.append(sk_auc(fpr, tpr))
    malla, tpr_m, tpr_sd = base.curva_media(curvas)
    return malla, tpr_m, tpr_sd, float(np.mean(aucs)), float(np.std(aucs))


def verificacion_contra_json(arquitectura, cache_todas_fases):
    print(f"\nVerificación ({arquitectura}) contra experimentoE_*.json:")
    ok_total = True
    for semilla in SEMILLAS:
        ruta = BASE / f"experimentoE_{arquitectura}_semilla{semilla}.json"
        if not ruta.exists():
            print(f"  (no encontrado: {ruta.name})")
            continue
        d = json.load(open(ruta, encoding="utf-8"))
        for fase_dict in d["fases"]:
            fase = fase_dict["fase"]
            for gen_json, m in fase_dict["evaluacion"].items():
                auc_json = m["auc"]
                y_true, p_fake = cache_todas_fases[(fase, semilla)][gen_json]
                fpr, tpr, _ = roc_curve(y_true, p_fake)
                auc_recalc = sk_auc(fpr, tpr)
                ok = abs(auc_json - auc_recalc) < 1e-3
                ok_total &= ok
                marca = "OK" if ok else "*** REVISAR ***"
                if not ok:
                    print(f"  {fase} | semilla {semilla} | {gen_json}: "
                          f"json={auc_json:.4f} recalc={auc_recalc:.4f}  {marca}")
    print(f"  {'Todo coincide (< 0.001 de diferencia).' if ok_total else 'Hay diferencias — revisar arriba.'}")
    return ok_total


def figura_arquitectura(arquitectura, cache_fase0, cache_todas_fases):
    columnas = ["Fase 0 (línea base)"] + FASES
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, col in zip(axes, columnas):
        for gen_json in LETRA_DE_GENERADOR_JSON:
            if col == "Fase 0 (línea base)":
                cache_por_semilla = {s: cache_fase0[s] for s in SEMILLAS}
                malla, tpr_m, _, auc_m, auc_sd = curva_media_generador(
                    arquitectura, None, gen_json, cache_por_semilla)
            else:
                cache_por_semilla = {s: cache_todas_fases[(col, s)] for s in SEMILLAS}
                malla, tpr_m, _, auc_m, auc_sd = curva_media_generador(
                    arquitectura, col, gen_json, cache_por_semilla)
            ax.plot(malla, tpr_m, color=COL_GEN[gen_json], lw=1.8,
                    label=f"{NB_GEN[gen_json]} ({auc_m:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_title(col)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Tasa de falsos positivos")
        if col == columnas[0]:
            ax.set_ylabel("Tasa de verdaderos positivos")
            ax.legend(loc="lower right", fontsize=8, title="Generador evaluado (AUC medio)")
    fig.suptitle(f"Experimento E — {NB[arquitectura]} — evolución de la curva ROC "
                f"por generador a lo largo del fine-tuning incremental")
    fig.tight_layout()
    nombre = f"roc_E_{arquitectura}.png"
    fig.savefig(RUTA_SALIDA / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> roc/{nombre}")


if __name__ == "__main__":
    print(f"Dispositivo: {config.DISPOSITIVO}")

    print("\nComprobando checkpoints de fine-tuning disponibles...")
    arqs_ok = arquitecturas_disponibles()
    if not arqs_ok:
        print("\nNo hay ninguna arquitectura con los 9 checkpoints completos. Nada que hacer.")
        sys.exit(0)
    print(f"\nArquitecturas con Experimento E completo: {arqs_ok}")

    for arquitectura in arqs_ok:
        print(f"\n=== {arquitectura} ===")
        cache_fase0 = {s: probas_fase0(arquitectura, s) for s in SEMILLAS}
        cache_todas_fases = {}
        for fase in FASES:
            for semilla in SEMILLAS:
                print(f"  {fase} | semilla {semilla}: calculando/cargando...")
                cache_todas_fases[(fase, semilla)] = probas_fase(arquitectura, fase, semilla)

        verificacion_contra_json(arquitectura, cache_todas_fases)
        figura_arquitectura(arquitectura, cache_fase0, cache_todas_fases)

    print("\nListo.")
    faltan_arqs = [a for a in ARQS if a not in arqs_ok]
    if faltan_arqs:
        print(f"\nRecuerda: aún faltan los checkpoints de Experimento E para {faltan_arqs}. "
              f"Vuelve a correr este script cuando los descargues.")
