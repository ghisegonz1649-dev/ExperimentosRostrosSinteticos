"""
analizar_resultados.py
Convierte los .json ya guardados (experimentos.py y experimentos_bcd.py) en
las TABLAS y FIGURAS COMPARATIVAS. No reentrena ni reevalúa
nada: solo lee, agrega y dibuja.

Diferencia con graficos_metricas.py:
    graficos_metricas.py  -> figuras de UNA arquitectura (diagnóstico).
    analizar_resultados.py -> tablas formales y figuras que CRUZAN las tres
                              arquitecturas (lo que va en la tesis).

Genera en RUTA_RESULTADOS/analisis/:

    TABLAS (cada una en .csv y en .md, para pegar en el documento)
        tabla_experimento_A.csv/.md      Rendimiento de las 3 CNN en A
        tabla_experimento_B.csv/.md      Rendimiento cross-dataset en B
        tabla_degradacion_A_B.csv/.md    Degradación A -> B
        tabla_experimento_C.csv/.md      Rendimiento cross-family en C
        tabla_degradacion_A_C.csv/.md    Degradación A -> C
        tabla_experimento_D.csv/.md      Rendimiento en D
        tabla_degradacion_A_D.csv/.md    Degradación A -> D

    FIGURAS
        figura_curvas_aprendizaje.png    Curvas comparativas (3 CNN), Exp A
        figura_matrices_A.png            Matrices de confusión, Exp A
        figura_matrices_B.png            Matrices de confusión, Exp B
        figura_matrices_C.png            Matrices de confusión, Exp C
        figura_matrices_D.png            Matrices de confusión, Exp D
        figura_degradacion.png           Trayectoria A->B->C->D por CNN

Convención de clases (igual que nucleo.py): fake=0 (positiva), real=1.
Matriz de confusión: fila 0 = verdadero fake, columna 0 = predicho fake.

La ESPECIFICIDAD (recall de la clase real) no viene en los .json: se deriva
aquí de la matriz de confusión. Es la métrica que muestra que la degradación
cross-generador viene toda de falsos negativos (falsos clasificados como
reales) y no de falsos positivos.

Dispersión: se reporta la DESVIACIÓN ESTÁNDAR MUESTRAL (ddof=1) sobre las
3 semillas, que es la convención al reportar n corridas independientes.

"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DEL ANÁLISIS
# ---------------------------------------------------------------------------
# Configurables por variable de entorno; por defecto, relativas a la raíz
# del repo (un nivel arriba de Codigos Experimentos/), por si los .json se
# movieron a mano (igual que en experimentos_bcd.py).
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

RUTA_RESULTADOS_A = Path(
    os.environ.get("RUTA_RESULTADOS_A", RAIZ_PROYECTO / "Resultados Experimento A")
)

RUTA_RAIZ_PROYECTO = Path(os.environ.get("RUTA_RAIZ_PROYECTO", RAIZ_PROYECTO))
CARPETA_EXCLUIDA_BUSQUEDA = "Datasets"


def _buscar_en_todo_el_proyecto(nombre: str) -> Path | None:
    if not RUTA_RAIZ_PROYECTO.exists():
        return None
    for carpeta in RUTA_RAIZ_PROYECTO.iterdir():
        if carpeta.is_dir() and carpeta.name != CARPETA_EXCLUIDA_BUSQUEDA:
            coincidencias = list(carpeta.rglob(nombre))
            if coincidencias:
                return coincidencias[0]
    return None

RUTA_ANALISIS = config.RUTA_RESULTADOS / "analisis"

GENERADOR_ENTRENAMIENTO = "StyleGAN2"

ARQUITECTURAS = ["efficientnet_b0", "resnet50", "legacy_xception"]

ARQUITECURA = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
    "legacy_xception": "Xception",
}

# Experimento -> generador de evaluación (igual que experimentos_bcd.py)
GENERADOR_DE_EXP = {
    "A": "StyleGAN2",
    "B": "StyleGAN3",
    "C": "SDXL",
    "D": "Flux",
}

# Métricas que van en las tablas, en orden
METRICAS = [
    ("accuracy", "Accuracy"),
    ("precision_fake", "Precision (fake)"),
    ("recall_fake", "Recall (fake)"),
    ("especificidad", "Especificidad (real)"),
    ("f1_fake", "F1 (fake)"),
    ("auc", "AUC-ROC"),
]

# ===========================================================================
# LECTURA DE LOS .JSON
# ===========================================================================
def _con_especificidad(metricas: dict) -> dict:
    m = metricas.get("matriz_confusion")
    if not m:
        return metricas
    vn = m[1][1]
    fp = m[1][0]
    resultado = dict(metricas)
    resultado["especificidad"] = vn / (vn + fp) if (vn + fp) > 0 else float("nan")
    return resultado

def _buscar_json_a(arquitectura: str) -> Path | None:
    nombre = f"experimentoA_{arquitectura}_{GENERADOR_ENTRENAMIENTO}.json"
    candidato = config.RUTA_METRICAS / nombre
    if candidato.exists():
        return candidato
    if RUTA_RESULTADOS_A.exists():
        coincidencias = list(RUTA_RESULTADOS_A.rglob(nombre))
        if coincidencias:
            return coincidencias[0]
    return _buscar_en_todo_el_proyecto(nombre)


def _buscar_json_bcd(arquitectura: str) -> Path | None:
    propio = config.RUTA_METRICAS / f"experimentosBCD_{arquitectura}.json"
    if propio.exists():
        return propio
    combinado = config.RUTA_METRICAS / "experimentosBCD.json"
    if combinado.exists():
        return combinado
    encontrado = _buscar_en_todo_el_proyecto(f"experimentosBCD_{arquitectura}.json")
    if encontrado:
        return encontrado
    return _buscar_en_todo_el_proyecto("experimentosBCD.json")


def cargar_todo(arquitecturas: list[str]) -> dict:

    datos: dict[str, dict] = {}

    for arq in arquitecturas:
        ruta_a = _buscar_json_a(arq)
        if ruta_a is None:
            print(f"  [falta] {arq}: sin .json del Experimento A -> se omite")
            continue

        with open(ruta_a, encoding="utf-8") as f:
            json_a = json.load(f)

        entrada = {"historial": {}, "metricas": {"A": {}}}
        for r in json_a["resultados_por_semilla"]:
            entrada["historial"][r["semilla"]] = r["historial"]
            entrada["metricas"]["A"][r["semilla"]] = _con_especificidad(
                r["metricas_test"])
        print(f"  [ok] {arq}: Experimento A <- {ruta_a.name}")

        ruta_bcd = _buscar_json_bcd(arq)
        if ruta_bcd is None:
            print(f"  [falta] {arq}: sin .json de B/C/D -> solo tendrá A")
        else:
            with open(ruta_bcd, encoding="utf-8") as f:
                json_bcd = json.load(f)
            for e in json_bcd["evaluaciones"]:
                # El .json combinado trae las 3 arquitecturas mezcladas
                if e["arquitectura"] != arq:
                    continue
                exp = e["experimento"]
                entrada["metricas"].setdefault(exp, {})[e["semilla"]] = (
                    _con_especificidad(e["metricas"]))
            presentes = [x for x in "BCD" if entrada["metricas"].get(x)]
            print(f"  [ok] {arq}: Experimentos {', '.join(presentes)} "
                  f"<- {ruta_bcd.name}")

        datos[arq] = entrada

    if not datos:
        raise FileNotFoundError(
            "No encontré ningún .json. Corre primero experimentos.py "
            f"(y experimentos_bcd.py). Busqué en {config.RUTA_METRICAS} "
            f"y bajo {RUTA_RESULTADOS_A}."
        )
    return datos


# ===========================================================================
# AGREGACIÓN: MEDIA ± DESVIACIÓN MUESTRAL
# ===========================================================================
def media_sd(valores: list[float]) -> tuple[float, float]:
    """Media y desviación estándar MUESTRAL (ddof=1). Con una sola semilla
    la sd no está definida y se devuelve 0.0."""
    arr = np.asarray(valores, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _formatear(media: float, sd: float) -> str:
    return f"{media:.4f} ± {sd:.4f}"


def agregar_experimento(datos: dict, arq: str, exp: str) -> dict | None:
    """{clave_metrica: (media, sd)} de un experimento y arquitectura, o None
    si no hay datos para esa combinación."""
    por_semilla = datos[arq]["metricas"].get(exp)
    if not por_semilla:
        return None
    return {
        clave: media_sd([m[clave] for m in por_semilla.values()])
        for clave, _ in METRICAS
    }

# ===========================================================================
# ESCRITURA DE TABLAS (.csv y .md)
# ===========================================================================
def escribir_tabla(nombre_base: str, encabezados: list[str],
                   filas: list[list[str]], titulo: str) -> list[Path]:
    """Escribe la misma tabla en .csv (para Excel) y .md (para pegar en Word
    o convertir). Devuelve las rutas escritas."""
    RUTA_ANALISIS.mkdir(parents=True, exist_ok=True)
    salidas = []

    ruta_csv = RUTA_ANALISIS / f"{nombre_base}.csv"
    with open(ruta_csv, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(encabezados)
        escritor.writerows(filas)
    salidas.append(ruta_csv)

    ruta_md = RUTA_ANALISIS / f"{nombre_base}.md"
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write(f"**{titulo}**\n\n")
        f.write("| " + " | ".join(encabezados) + " |\n")
        f.write("|" + "|".join(["---"] * len(encabezados)) + "|\n")
        for fila in filas:
            f.write("| " + " | ".join(fila) + " |\n")
    salidas.append(ruta_md)

    return salidas

def tabla_rendimiento(datos: dict, exp: str) -> list[Path] | None:
    """Tabla de rendimiento: una fila por arquitectura, una columna
    por métrica, con media ± sd de las 3 semillas."""
    filas = []
    for arq in datos:
        agregado = agregar_experimento(datos, arq, exp)
        if agregado is None:
            continue
        filas.append([ARQUITECURA.get(arq, arq)]
                     + [_formatear(*agregado[c]) for c, _ in METRICAS])
    if not filas:
        return None

    generador = GENERADOR_DE_EXP[exp]
    if exp == "A":
        titulo = (f"Rendimiento de las tres arquitecturas en el "
                  f"Experimento A ({generador}). Media ± desviación estándar "
                  f"sobre {len(config.SEMILLAS_ENTRENAMIENTO)} semillas.")
    else:
        titulo = (f"Rendimiento cross-generador de las tres "
                  f"arquitecturas en el Experimento {exp} ({generador}), sin "
                  f"reentrenamiento. Media ± desviación estándar sobre "
                  f"{len(config.SEMILLAS_ENTRENAMIENTO)} semillas.")

    encabezados = ["Arquitectura"] + [e for _, e in METRICAS]
    return escribir_tabla(f"tabla_experimento_{exp}",
                          encabezados, filas, titulo)


def tabla_degradacion(datos: dict, exp: str) -> list[Path] | None:
    filas = []
    for arq in datos:
        m_a = datos[arq]["metricas"].get("A")
        m_x = datos[arq]["metricas"].get(exp)
        if not m_a or not m_x:
            continue

        semillas_comunes = sorted(set(m_a) & set(m_x))
        if not semillas_comunes:
            continue

        fila = [ARQUITECURA.get(arq, arq)]
        for clave, _ in METRICAS:
            caidas = [m_a[s][clave] - m_x[s][clave] for s in semillas_comunes]
            media, sd = media_sd(caidas)
            # Signo explícito: positivo = empeoró al cambiar de generador
            fila.append(f"{media:+.4f} ± {sd:.4f}")
        filas.append(fila)

    if not filas:
        return None

    generador = GENERADOR_DE_EXP[exp]
    titulo = (f"Degradación del rendimiento (A → {exp}) por "
              f"arquitectura, de {GENERADOR_ENTRENAMIENTO} a {generador}. "
              f"Valores positivos indican caída respecto a la línea base. "
              f"Diferencia calculada por semilla y promediada.")

    encabezados = ["Arquitectura"] + [f"Δ {e}" for _, e in METRICAS]
    return escribir_tabla(f"tabla_degradacion_A_{exp}",
                          encabezados, filas, titulo)


# ===========================================================================
# FIGURA — CURVAS DE APRENDIZAJE COMPARATIVAS
# ===========================================================================
def figura_curvas_aprendizaje(datos: dict) -> Path:
    """Una columna por arquitectura, dos filas (pérdida y accuracy). Cada
    semilla es una línea: sólida = train, punteada = validación."""
    arqs = list(datos)
    fig, axes = plt.subplots(2, len(arqs), figsize=(5.2 * len(arqs), 8),
                             squeeze=False)
    colores = plt.cm.tab10.colors

    for col, arq in enumerate(arqs):
        ax_loss = axes[0][col]
        ax_acc = axes[1][col]

        for i, (semilla, hist) in enumerate(sorted(datos[arq]["historial"].items())):
            epocas = [h["epoca"] for h in hist]
            color = colores[i % len(colores)]
            ax_loss.plot(epocas, [h["perdida_train"] for h in hist], "-",
                         color=color, label=f"train (s{semilla})")
            ax_loss.plot(epocas, [h["perdida_val"] for h in hist], "--",
                         color=color, label=f"val (s{semilla})")
            ax_acc.plot(epocas, [h["acc_train"] for h in hist], "-",
                        color=color, label=f"train (s{semilla})")
            ax_acc.plot(epocas, [h["acc_val"] for h in hist], "--",
                        color=color, label=f"val (s{semilla})")

        ax_loss.set_title(ARQUITECURA.get(arq, arq))
        ax_loss.set_xlabel("Época")
        ax_loss.set_ylabel("Pérdida (CrossEntropy)")
        ax_loss.grid(alpha=0.3)
        ax_loss.legend(fontsize=7)

        ax_acc.set_xlabel("Época")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.grid(alpha=0.3)
        ax_acc.legend(fontsize=7, loc="lower right")

    # Mismo eje Y en toda la fila: si no, las tres curvas no son comparables
    # a simple vista (que es justo el propósito de la figura).
    for fila in axes:
        limites = [ax.get_ylim() for ax in fila]
        minimo = min(l[0] for l in limites)
        maximo = max(l[1] for l in limites)
        for ax in fila:
            ax.set_ylim(minimo, maximo)

    fig.suptitle(f"Curvas de aprendizaje comparativas — "
                 f"Experimento A ({GENERADOR_ENTRENAMIENTO}). "
                 f"Sólida = entrenamiento, punteada = validación.",
                 fontsize=12, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    RUTA_ANALISIS.mkdir(parents=True, exist_ok=True)
    salida = RUTA_ANALISIS / "figura_curvas_aprendizaje.png"
    fig.savefig(salida, dpi=200)
    plt.close(fig)
    return salida


# ===========================================================================
# FIGURAS — MATRICES DE CONFUSIÓN COMPARATIVAS
# ===========================================================================
def _dibujar_matriz(ax, m: np.ndarray, titulo: str) -> None:
    # Normalizamos el color por fila (por clase verdadera): así el color
    # significa "qué porcentaje de esta clase fue a parar aquí", que es lo
    # que se quiere leer, y no depende de cuántas imágenes haya.
    totales = m.sum(axis=1, keepdims=True)
    proporciones = np.divide(m, totales, out=np.zeros_like(m, dtype=float),
                             where=totales > 0)
    ax.imshow(proporciones, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["fake", "real"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["fake", "real"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=10)
    for i in range(2):
        for j in range(2):
            color = "white" if proporciones[i, j] > 0.5 else "black"
            ax.text(j, i, f"{int(m[i, j])}\n({proporciones[i, j]*100:.1f}%)",
                    ha="center", va="center", color=color,
                    fontsize=10, fontweight="bold")


def figura_matrices(datos: dict, exp: str) -> Path | None:
    """Matrices de confusión de las tres arquitecturas para UN experimento.
    Se suman las 3 semillas: es la matriz agregada que va en la tesis (las
    matrices por semilla ya las genera graficos_metricas.py)."""
    paneles = []
    for arq in datos:
        por_semilla = datos[arq]["metricas"].get(exp)
        if not por_semilla:
            continue
        total = sum(np.array(m["matriz_confusion"])
                    for m in por_semilla.values())
        paneles.append((ARQUITECURA.get(arq, arq), total, len(por_semilla)))

    if not paneles:
        return None

    fig, axes = plt.subplots(1, len(paneles), figsize=(4.5 * len(paneles), 4.4),
                             squeeze=False)
    for ax, (nombre, m, n_semillas) in zip(axes[0], paneles):
        _dibujar_matriz(ax, m, nombre)

    generador = GENERADOR_DE_EXP[exp]
    n = paneles[0][2]
    fig.suptitle(f"Matrices de confusión — Experimento {exp} "
                 f"({generador}). Conteos acumulados de las {n} semillas; "
                 f"el color indica la proporción dentro de cada clase real.",
                 fontsize=11, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    RUTA_ANALISIS.mkdir(parents=True, exist_ok=True)
    salida = RUTA_ANALISIS / f"figura_matrices_{exp}.png"
    fig.savefig(salida, dpi=200)
    plt.close(fig)
    return salida


def figura_degradacion(datos: dict) -> Path | None:
    """Accuracy y AUC de cada arquitectura a lo largo de A -> B -> C -> D.
    Es la figura que cuenta la historia central del "aprendo -> fallo":
    una línea por arquitectura, con barras de error de las 3 semillas."""
    experimentos = ["A", "B", "C", "D"]
    # Solo los experimentos que tengan datos en alguna arquitectura
    experimentos = [e for e in experimentos
                    if any(datos[a]["metricas"].get(e) for a in datos)]
    if len(experimentos) < 2:
        return None

    fig, (ax_acc, ax_auc) = plt.subplots(1, 2, figsize=(13, 5))
    marcadores = ["o", "s", "^", "D"]

    for i, arq in enumerate(datos):
        for ax, clave, etiqueta_y in [(ax_acc, "accuracy", "Accuracy"),
                                      (ax_auc, "auc", "AUC-ROC")]:
            xs, medias, sds = [], [], []
            for j, exp in enumerate(experimentos):
                agregado = agregar_experimento(datos, arq, exp)
                if agregado is None:
                    continue
                xs.append(j)
                medias.append(agregado[clave][0])
                sds.append(agregado[clave][1])
            if not xs:
                continue
            ax.errorbar(xs, medias, yerr=sds, marker=marcadores[i % 4],
                        capsize=4, linewidth=2, markersize=7,
                        label=ARQUITECURA.get(arq, arq))
            ax.set_ylabel(etiqueta_y)

    for ax in (ax_acc, ax_auc):
        ax.set_xticks(range(len(experimentos)))
        ax.set_xticklabels([f"{e}\n({GENERADOR_DE_EXP[e]})"
                            for e in experimentos])
        ax.set_ylim(0.4, 1.02)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.text(0.02, 0.505, "azar", color="gray", fontsize=8,
                transform=ax.get_yaxis_transform())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")

    ax_acc.set_title("Accuracy por experimento")
    ax_auc.set_title("AUC-ROC por experimento")
    fig.suptitle("Degradación del rendimiento al cambiar de "
                 "generador (media ± sd de las semillas). Modelos entrenados "
                 f"en {GENERADOR_ENTRENAMIENTO}, sin reentrenamiento.",
                 fontsize=12, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    RUTA_ANALISIS.mkdir(parents=True, exist_ok=True)
    salida = RUTA_ANALISIS / "figura_degradacion.png"
    fig.savefig(salida, dpi=200)
    plt.close(fig)
    return salida

def ejecutar(arquitecturas: list[str]) -> None:
    print("=" * 66)
    print("ANÁLISIS DE RESULTADOS — tablas y figuras del Capítulo V")
    print("=" * 66)
    print("\nLeyendo .json:")
    datos = cargar_todo(arquitecturas)

    print("\nTablas:")
    for exp in ["A", "B", "C", "D"]:
        rutas = tabla_rendimiento(datos, exp)
        if rutas:
            print(f"  -> {rutas[0].name} / {rutas[1].name}")
        if exp != "A":
            rutas = tabla_degradacion(datos, exp)
            if rutas:
                print(f"  -> {rutas[0].name} / {rutas[1].name}")

    print("\nFiguras:")
    print(f"  -> {figura_curvas_aprendizaje(datos).name}")
    for exp in ["A", "B", "C", "D"]:
        ruta = figura_matrices(datos, exp)
        if ruta:
            print(f"  -> {ruta.name}")
    ruta = figura_degradacion(datos)
    if ruta:
        print(f"  -> {ruta.name}")

    print(f"\nListo. Todo en: {RUTA_ANALISIS}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tablas y figuras comparativas del Capítulo V a partir "
                    "de los .json ya guardados")
    ap.add_argument("--arquitecturas", nargs="+", default=ARQUITECTURAS,
                    choices=ARQUITECTURAS,
                    help="Arquitecturas a incluir (por defecto, las tres).")
    args = ap.parse_args()
    ejecutar(args.arquitecturas)


if __name__ == "__main__":
    main()