"""
generar_figuras_celeba.py
Genera TODAS las figuras de la tanda CelebA-HQ a partir de los 6 .json
(3 de Experimento A + 3 de B/C/D). No reentrena ni reevalúa: solo lee.

Separa las dos historias de la tesis:
  - RECALL (fake)      -> generalización cross-generador (detección de deepfakes)
  - ESPECIFICIDAD (real) -> robustez al dataset real (¿reconoce CelebA-HQ?)

Salidas en RUTA_SALIDA. Las matrices de confusión y las curvas de aprendizaje
se guardan en imágenes INDIVIDUALES (una por arquitectura/experimento).

Uso:
    python generar_figuras_celeba.py
"""

import json
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# RUTAS  (los .json están en la misma carpeta que este script)
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent       # carpeta donde vive este script
RUTA_METRICAS = BASE                          # donde están los .json
RUTA_SALIDA = BASE / "figuras"                # donde se guardan las figuras
RUTA_MATRICES = RUTA_SALIDA / "matrices"     # subcarpeta: matrices individuales
RUTA_CURVAS = RUTA_SALIDA / "curvas"         # subcarpeta: curvas individuales

GENERADOR_ENTRENAMIENTO = "StyleGAN2_CelebA"

NB = {"efficientnet_b0": "EfficientNet-B0",
      "resnet50": "ResNet-50",
      "legacy_xception": "Xception"}
ARQS = ["efficientnet_b0", "resnet50", "legacy_xception"]
EXPS = ["A", "B", "C", "D"]
GEN = {"A": "StyleGAN2", "B": "StyleGAN3", "C": "SDXL", "D": "Flux"}
COL = {"efficientnet_b0": "tab:blue", "resnet50": "tab:orange",
       "legacy_xception": "tab:green"}
# Colormap por experimento (se usa en las matrices individuales y agrupadas)
CMAP = {"A": "Blues", "B": "Greens", "C": "Oranges", "D": "Purples"}


# ---------------------------------------------------------------------------
# LECTURA
# ---------------------------------------------------------------------------
def especificidad(m):
    """Especificidad = reales bien clasificadas / reales totales.
    Matriz [[fake->fake, fake->real],[real->fake, real->real]]."""
    mc = m["matriz_confusion"]
    vn, fp = mc[1][1], mc[1][0]
    return vn / (vn + fp) if (vn + fp) else float("nan")


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


def agg(arq, exp, fn):
    dic = fuente(arq, exp)
    vals = [fn(m) if callable(fn) else m[fn] for m in dic.values()]
    return np.mean(vals), np.std(vals, ddof=1)


XT = [f"{e}\n({GEN[e]})" for e in EXPS]


# ===========================================================================
# FIGURAS DE LÍNEA (comparativas, las 3 arquitecturas juntas)
# ===========================================================================
def figura_lineas(fn, marker, ylabel, titulo, nombre, ylim=(0, 1.05),
                  linea_azar=True):
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arq in ARQS:
        ms = [agg(arq, e, fn)[0] for e in EXPS]
        ss = [agg(arq, e, fn)[1] for e in EXPS]
        ax.errorbar(range(4), ms, yerr=ss, marker=marker, capsize=4,
                    lw=2, ms=8, label=NB[arq], color=COL[arq])
    ax.set_xticks(range(4)); ax.set_xticklabels(XT)
    ax.set_ylim(*ylim)
    if linea_azar:
        ax.axhline(0.5, ls=":", c="gray")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(RUTA_SALIDA / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> {nombre}")


def figura_dos_historias():
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    paneles = [("recall_fake", "Recall (fake) — generalización", "o"),
               (especificidad, "Especificidad (real) — robustez", "s")]
    for ax, (fn, tit, mk) in zip(axes, paneles):
        for arq in ARQS:
            ms = [agg(arq, e, fn)[0] for e in EXPS]
            ss = [agg(arq, e, fn)[1] for e in EXPS]
            ax.errorbar(range(4), ms, yerr=ss, marker=mk, capsize=4,
                        lw=2, ms=7, label=NB[arq], color=COL[arq])
        ax.set_xticks(range(4)); ax.set_xticklabels(XT)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, ls=":", c="gray")
        ax.set_title(tit); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Las dos caras del problema: el modelo falla en ejes "
                 "distintos según la arquitectura")
    fig.tight_layout()
    fig.savefig(RUTA_SALIDA / "fig_dos_historias.png", dpi=200)
    plt.close(fig)
    print("  -> fig_dos_historias.png")


# ===========================================================================
# MATRICES DE CONFUSIÓN — UNA IMAGEN POR ARQUITECTURA Y EXPERIMENTO
# ===========================================================================
def figura_matriz_individual(arq, exp):
    RUTA_MATRICES.mkdir(parents=True, exist_ok=True)
    dic = fuente(arq, exp)
    M = sum(np.array(m["matriz_confusion"]) for m in dic.values())
    tot = M.sum(axis=1, keepdims=True)
    prop = np.divide(M, tot, out=np.zeros_like(M, dtype=float), where=tot > 0)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(prop, cmap=CMAP[exp], vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["fake", "real"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["fake", "real"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    for a in range(2):
        for b in range(2):
            c = "white" if prop[a, b] > 0.5 else "black"
            ax.text(b, a, f"{int(M[a, b])}",
                    ha="center", va="center", color=c,
                    fontweight="bold", fontsize=12)
    ax.set_title(f"{NB[arq]} — Exp. {exp} ({GEN[exp]})\n"
                 f"suma de 3 semillas", fontsize=11)
    fig.tight_layout()
    nombre = f"matriz_{arq}_{exp}.png"
    fig.savefig(RUTA_MATRICES / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> matrices/{nombre}")


# ===========================================================================
# MATRICES DE CONFUSIÓN — UNA IMAGEN POR EXPERIMENTO (las 3 arquitecturas)
# ===========================================================================
def figura_matrices_por_experimento(exp):
    RUTA_MATRICES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(ARQS), figsize=(4.5 * len(ARQS), 4.6))
    for ax, arq in zip(axes, ARQS):
        dic = fuente(arq, exp)
        M = sum(np.array(m["matriz_confusion"]) for m in dic.values())
        tot = M.sum(axis=1, keepdims=True)
        prop = np.divide(M, tot, out=np.zeros_like(M, dtype=float),
                         where=tot > 0)
        ax.imshow(prop, cmap=CMAP[exp], vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["fake", "real"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["fake", "real"])
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        for a in range(2):
            for b in range(2):
                c = "white" if prop[a, b] > 0.5 else "black"
                ax.text(b, a, f"{int(M[a, b])}",
                        ha="center", va="center", color=c,
                        fontweight="bold", fontsize=12)
        ax.set_title(NB[arq], fontsize=12)
    fig.suptitle(f"Matrices de confusión — Exp. {exp} ({GEN[exp]})  "
                 f"(suma de 3 semillas)", fontsize=13)
    fig.tight_layout()
    nombre = f"matriz_comparativa_exp_{exp}.png"
    fig.savefig(RUTA_MATRICES / nombre, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> matrices/{nombre}")


# ===========================================================================
# CURVAS DE APRENDIZAJE — UNA IMAGEN POR ARQUITECTURA
# ===========================================================================
def figura_curvas_individual(arq):
    RUTA_CURVAS.mkdir(parents=True, exist_ok=True)
    ruta = RUTA_METRICAS / f"experimentoA_{arq}_{GENERADOR_ENTRENAMIENTO}.json"
    d = json.load(open(ruta, encoding="utf-8"))
    col = plt.cm.tab10.colors

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    for k, r in enumerate(d["resultados_por_semilla"]):
        h = r["historial"]
        ep = [x["epoca"] for x in h]
        c = col[k]
        s = r["semilla"]
        ax_loss.plot(ep, [x["perdida_train"] for x in h], "-", color=c,
                     label=f"train s{s}")
        ax_loss.plot(ep, [x["perdida_val"] for x in h], "--", color=c,
                     label=f"val s{s}")
        ax_acc.plot(ep, [x["acc_train"] for x in h], "-", color=c)
        ax_acc.plot(ep, [x["acc_val"] for x in h], "--", color=c)

    ax_loss.set_xlabel("Época"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss"); ax_loss.grid(alpha=0.3); ax_loss.legend(fontsize=8)
    ax_acc.set_xlabel("Época"); ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy"); ax_acc.grid(alpha=0.3)

    fig.suptitle(f"Curvas de aprendizaje — {NB[arq]} — Experimento A "
                 f"({GENERADOR_ENTRENAMIENTO}). Sólida=train, punteada=val")
    fig.tight_layout()
    nombre = f"curvas_{arq}.png"
    fig.savefig(RUTA_CURVAS / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> curvas/{nombre}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("Figuras comparativas (3 arquitecturas juntas):")
    figura_dos_historias()
    figura_lineas("recall_fake", "o", "Recall (fake) — media ± sd",
                  "Generalización cross-generador: detección de deepfakes\n"
                  "(entrenado en StyleGAN2, sin reentrenar)",
                  "fig_recall_crossgen.png")
    figura_lineas(especificidad, "s", "Especificidad (real) — media ± sd",
                  "Robustez al dataset real: ¿reconoce caras reales de "
                  "CelebA-HQ?\n(constante entre experimentos: las reales no "
                  "cambian)", "fig_especificidad_robustez.png")
    figura_lineas("accuracy", "^", "Accuracy global — media ± sd",
                  "Accuracy global por experimento (mezcla las dos métricas)",
                  "fig_accuracy_global.png")
    figura_lineas("auc", "D", "AUC-ROC — media ± sd",
                  "AUC-ROC por experimento",
                  "fig_auc.png", ylim=(0.4, 1.02))

    print("\nMatrices de confusión (una imagen por arquitectura y experimento):")
    for arq in ARQS:
        for exp in EXPS:
            figura_matriz_individual(arq, exp)

    print("\nMatrices de confusión comparativas (una imagen por experimento, "
          "las 3 arquitecturas):")
    for exp in EXPS:
        figura_matrices_por_experimento(exp)

    print("\nCurvas de aprendizaje (una imagen por arquitectura):")
    for arq in ARQS:
        figura_curvas_individual(arq)

    print(f"\nListo. Todo en: {RUTA_SALIDA}")
    print(f"  matrices individuales -> {RUTA_MATRICES}")
    print(f"  curvas individuales   -> {RUTA_CURVAS}")


if __name__ == "__main__":
    main()
