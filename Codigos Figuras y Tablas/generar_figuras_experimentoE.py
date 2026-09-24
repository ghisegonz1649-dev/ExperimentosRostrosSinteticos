"""
generar_figuras_experimentoE.py
Genera las figuras del Experimento E (aprendizaje incremental / continual
learning) a partir de los 9 .json:  3 arquitecturas x 3 semillas.
No reentrena ni reevalua: solo lee los .json ya generados.

Estructura de cada .json (experimentoE_<arq>_semilla<s>.json):
  - evaluacion_inicial : modelo base (Exp. A, entrenado solo en StyleGAN2)
                         evaluado sobre los 4 generadores.
  - fases[]            : FT-1 (+StyleGAN3), FT-2 (+SDXL), FT-3 (+Flux).
      cada fase tiene 'historial' (curvas por epoca) y 'evaluacion'
      (metricas sobre los 4 generadores tras esa fase).

Historia del experimento:
  Al ir anadiendo generadores al entrenamiento, se sigue la deteccion de
  cada generador etapa a etapa. Un generador se detecta mal (zero-shot)
  hasta que ENTRA al entrenamiento; a partir de ahi salta. Tambien se vigila
  si detectar lo nuevo degrada lo viejo (olvido) o las caras reales.

Salidas en figuras/experimentoE/.

Uso:
    python generar_figuras_experimentoE.py
"""

import json
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
RUTA_METRICAS = BASE
RUTA_SALIDA = BASE / "figuras" / "experimentoE"
RUTA_CURVAS = RUTA_SALIDA / "curvas"
RUTA_MATRICES = RUTA_SALIDA / "matrices"

# ---------------------------------------------------------------------------
# NOMENCLATURA
# ---------------------------------------------------------------------------
NB = {"efficientnet_b0": "EfficientNet-B0",
      "resnet50": "ResNet-50",
      "legacy_xception": "Xception"}
ARQS = ["efficientnet_b0", "resnet50", "legacy_xception"]
COL_ARQ = {"efficientnet_b0": "tab:blue", "resnet50": "tab:orange",
           "legacy_xception": "tab:green"}
# Colormap distinto por arquitectura para las matrices de confusion
CMAP_ARQ = {"efficientnet_b0": "Blues", "resnet50": "Oranges",
            "legacy_xception": "Greens"}

# Generadores (orden en que se van incorporando) y etiqueta corta
GENS = ["StyleGAN2_CelebA", "StyleGAN3_CelebA", "SDXL_CelebA", "Flux_CelebA"]
GEN_LBL = {"StyleGAN2_CelebA": "StyleGAN2", "StyleGAN3_CelebA": "StyleGAN3",
           "SDXL_CelebA": "SDXL", "Flux_CelebA": "Flux"}
COL_GEN = {"StyleGAN2_CelebA": "tab:blue", "StyleGAN3_CelebA": "tab:green",
           "SDXL_CelebA": "tab:orange", "Flux_CelebA": "tab:purple"}

# Etapas del experimento (x de las graficas de evolucion)
ETAPAS = ["Inicial", "FT-1", "FT-2", "FT-3"]
# Generadores que YA se han visto en entrenamiento al llegar a cada etapa
VISTOS = {
    "Inicial": {"StyleGAN2_CelebA"},
    "FT-1":    {"StyleGAN2_CelebA", "StyleGAN3_CelebA"},
    "FT-2":    {"StyleGAN2_CelebA", "StyleGAN3_CelebA", "SDXL_CelebA"},
    "FT-3":    {"StyleGAN2_CelebA", "StyleGAN3_CelebA", "SDXL_CelebA",
                "Flux_CelebA"},
}
# Etapa en la que cada generador ENTRA por primera vez al entrenamiento
ENTRA_EN = {"StyleGAN2_CelebA": "Inicial", "StyleGAN3_CelebA": "FT-1",
            "SDXL_CelebA": "FT-2", "Flux_CelebA": "FT-3"}


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
    """DATA[arq][semilla][etapa][generador] = dict de metricas."""
    DATA = {}
    for f in glob.glob(str(RUTA_METRICAS / "experimentoE_*.json")):
        name = Path(f).stem            # experimentoE_<arq>_semilla<s>
        resto = name.replace("experimentoE_", "")
        arq, sem = resto.rsplit("_semilla", 1)
        sem = int(sem)
        d = json.load(open(f, encoding="utf-8"))
        etapas = {"Inicial": d["evaluacion_inicial"]}
        for fa in d["fases"]:
            etapas[fa["fase"]] = fa["evaluacion"]
        DATA.setdefault(arq, {})[sem] = {"etapas": etapas, "fases": d["fases"]}
    return DATA


DATA = cargar()
SEMILLAS = sorted(next(iter(DATA.values())).keys())


def valor(arq, etapa, gen, fn):
    """Media y sd (ddof=1) sobre semillas de una metrica (str) o funcion."""
    vals = []
    for s in SEMILLAS:
        m = DATA[arq][s]["etapas"][etapa][gen]
        vals.append(fn(m) if callable(fn) else m[fn])
    return np.mean(vals), np.std(vals, ddof=1)


# ===========================================================================
# EVOLUCION DE UNA METRICA POR GENERADOR (3 paneles = 3 arquitecturas)
# ===========================================================================
def figura_evolucion(fn, ylabel, titulo, nombre, ylim=(0, 1.05),
                     marcar_entrada=True, linea_azar=True):
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    x = range(len(ETAPAS))
    for ax, arq in zip(axes, ARQS):
        for gen in GENS:
            ms = [valor(arq, e, gen, fn)[0] for e in ETAPAS]
            ss = [valor(arq, e, gen, fn)[1] for e in ETAPAS]
            ax.errorbar(x, ms, yerr=ss, marker="o", capsize=3, lw=2, ms=6,
                        color=COL_GEN[gen], label=GEN_LBL[gen])
            if marcar_entrada:
                ie = ETAPAS.index(ENTRA_EN[gen])
                ax.scatter([ie], [ms[ie]], s=220, marker="*",
                           color=COL_GEN[gen], edgecolor="black",
                           zorder=5)
        if linea_azar:
            ax.axhline(0.5, ls=":", c="gray")
        ax.set_xticks(list(x)); ax.set_xticklabels(ETAPAS)
        ax.set_ylim(*ylim)
        ax.set_title(NB[arq])
        ax.grid(alpha=0.3)
        ax.set_xlabel("Etapa de entrenamiento incremental")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(title="Generador (test)", loc="lower right", fontsize=9)
    fig.suptitle(titulo + "   (media ± sd sobre 3 semillas; "
                 "★ = etapa en que el generador entra al entrenamiento)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(RUTA_SALIDA / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> experimentoE/{nombre}")


# ===========================================================================
# VISTOS vs NO-VISTOS: promedio de recall segun si el generador ya entro
# ===========================================================================
def figura_visto_vs_novisto():
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(ETAPAS))
    for arq in ARQS:
        vistos_m, novistos_m = [], []
        for e in ETAPAS:
            vis = [valor(arq, e, g, "recall_fake")[0]
                   for g in GENS if g in VISTOS[e]]
            nov = [valor(arq, e, g, "recall_fake")[0]
                   for g in GENS if g not in VISTOS[e]]
            vistos_m.append(np.mean(vis) if vis else np.nan)
            novistos_m.append(np.mean(nov) if nov else np.nan)
        ax.plot(x, vistos_m, "-o", lw=2, ms=7, color=COL_ARQ[arq],
                label=f"{NB[arq]} — vistos")
        ax.plot(x, novistos_m, "--s", lw=2, ms=6, color=COL_ARQ[arq],
                alpha=0.7, label=f"{NB[arq]} — no vistos")
    ax.axhline(0.5, ls=":", c="gray")
    ax.set_xticks(list(x)); ax.set_xticklabels(ETAPAS)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Etapa de entrenamiento incremental")
    ax.set_ylabel("Recall (fake) — promedio de generadores")
    ax.set_title("Generadores vistos vs. no vistos en cada etapa\n"
                 "(sólida = ya en entrenamiento; punteada = aún zero-shot)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(RUTA_SALIDA / "fig_E_visto_vs_novisto.png", dpi=200)
    plt.close(fig)
    print("  -> experimentoE/fig_E_visto_vs_novisto.png")


# ===========================================================================
# HEATMAP etapa x generador (una imagen por arquitectura)
# ===========================================================================
def figura_heatmap(arq, fn=("recall_fake"), etiqueta="Recall (fake)"):
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    M = np.array([[valor(arq, e, g, fn)[0] for g in GENS] for e in ETAPAS])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(GENS)))
    ax.set_xticklabels([GEN_LBL[g] for g in GENS])
    ax.set_yticks(range(len(ETAPAS))); ax.set_yticklabels(ETAPAS)
    ax.set_xlabel("Generador (test)")
    ax.set_ylabel("Etapa de entrenamiento")
    for i, e in enumerate(ETAPAS):
        for j, g in enumerate(GENS):
            visto = g in VISTOS[e]
            c = "white" if M[i, j] < 0.6 else "black"
            ax.text(j, i, f"{M[i, j]:.2f}" + ("\n(visto)" if visto else ""),
                    ha="center", va="center", color=c, fontsize=9,
                    fontweight="bold" if visto else "normal")
    fig.colorbar(im, ax=ax, label=etiqueta + " (media 3 semillas)")
    ax.set_title(f"{NB[arq]} — {etiqueta} por etapa y generador\n"
                 f"Exp. E (aprendizaje incremental)")
    fig.tight_layout()
    nombre = f"heatmap_recall_{arq}.png"
    fig.savefig(RUTA_SALIDA / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> experimentoE/{nombre}")


# ===========================================================================
# CURVAS DE APRENDIZAJE — UNA IMAGEN POR ARQUITECTURA Y FASE (FT-1/2/3),
# cada imagen con su panel de loss y su panel de accuracy.
# ===========================================================================
def figura_curvas_fase(arq, fase):
    RUTA_CURVAS.mkdir(parents=True, exist_ok=True)
    col = plt.cm.tab10.colors
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    for k, s in enumerate(SEMILLAS):
        fa = next(f for f in DATA[arq][s]["fases"] if f["fase"] == fase)
        h = fa["historial"]
        ep = [x["epoca"] for x in h]
        c = col[k]
        ax_loss.plot(ep, [x["perdida_train"] for x in h], "-", color=c,
                     label=f"train s{s}")
        ax_loss.plot(ep, [x["perdida_val"] for x in h], "--", color=c,
                     label=f"val s{s}")
        ax_acc.plot(ep, [x["acc_train"] for x in h], "-", color=c)
        ax_acc.plot(ep, [x["acc_val"] for x in h], "--", color=c)
    ax_loss.set_xlabel("Época"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss"); ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=8)
    ax_acc.set_xlabel("Época"); ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy"); ax_acc.grid(alpha=0.3)
    fig.suptitle(f"Curvas de aprendizaje — {NB[arq]} — Exp. E — {fase}\n"
                 "Sólida = train, punteada = val (una línea por semilla)",
                 fontsize=13)
    fig.tight_layout()
    nombre = f"curvas_E_{arq}_{fase}.png"
    fig.savefig(RUTA_CURVAS / nombre, dpi=200)
    plt.close(fig)
    print(f"  -> experimentoE/curvas/{nombre}")


# ===========================================================================
# MATRICES DE CONFUSION EN LA ETAPA FINAL (FT-3): arq x generador
# ===========================================================================
def figura_matrices_finales(arq):
    RUTA_MATRICES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(GENS), figsize=(4.3 * len(GENS), 4.5))
    for ax, gen in zip(axes, GENS):
        M = sum(np.array(DATA[arq][s]["etapas"]["FT-3"][gen]["matriz_confusion"])
                for s in SEMILLAS)
        tot = M.sum(axis=1, keepdims=True)
        prop = np.divide(M, tot, out=np.zeros_like(M, dtype=float),
                         where=tot > 0)
        ax.imshow(prop, cmap=CMAP_ARQ[arq], vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["fake", "real"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["fake", "real"])
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        for a in range(2):
            for b in range(2):
                c = "white" if prop[a, b] > 0.5 else "black"
                ax.text(b, a, f"{int(M[a, b])}", ha="center", va="center",
                        color=c, fontweight="bold", fontsize=12)
        ax.set_title(GEN_LBL[gen], fontsize=12)
    fig.suptitle(f"Matrices de confusión tras FT-3 (modelo final) — {NB[arq]}\n"
                 f"Exp. E — suma de 3 semillas", fontsize=13)
    fig.tight_layout()
    nombre = f"matriz_FT3_{arq}.png"
    fig.savefig(RUTA_MATRICES / nombre, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> experimentoE/matrices/{nombre}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("Evolución de métricas por generador (Exp. E):")
    figura_evolucion("recall_fake", "Recall (fake) — detección de deepfakes",
                     "Experimento E — evolución del Recall (fake) por generador",
                     "fig_E_recall_evolucion.png")
    figura_evolucion("auc", "AUC-ROC",
                     "Experimento E — evolución del AUC-ROC por generador",
                     "fig_E_auc_evolucion.png", ylim=(0.4, 1.02),
                     linea_azar=False)
    figura_evolucion("accuracy", "Accuracy global",
                     "Experimento E — evolución de la Accuracy por generador",
                     "fig_E_accuracy_evolucion.png")
    figura_evolucion(especificidad, "Especificidad (real)",
                     "Experimento E — especificidad (reales) por etapa",
                     "fig_E_especificidad_evolucion.png", marcar_entrada=False)

    print("\nVistos vs. no vistos:")
    figura_visto_vs_novisto()

    print("\nHeatmaps recall (una imagen por arquitectura):")
    for arq in ARQS:
        figura_heatmap(arq)

    print("\nCurvas de aprendizaje (una imagen por arquitectura y fase):")
    for arq in ARQS:
        for fase in ["FT-1", "FT-2", "FT-3"]:
            figura_curvas_fase(arq, fase)

    print("\nMatrices de confusión finales FT-3 (una imagen por arquitectura):")
    for arq in ARQS:
        figura_matrices_finales(arq)

    print(f"\nListo. Todo en: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
