from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ARCHITECTURA_LABELS = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
    "legacy_xception": "Xception",
}

# Sólo las cuatro métricas que se reportan en el capítulo de resultados de la
# tesis. Precisión y F1 se retiraron del análisis, así que no se exponen aquí.
# El orden es el de las tablas: AUC-ROC primero por ser la métrica principal.
METRICAS = {
    "auc": "AUC-ROC",
    "recall_fake": "Recall",
    "especificidad": "Especificidad",
    "accuracy": "Accuracy",
}

# Generadores evaluados, en el orden en que aparecen en el estudio (distancia
# creciente respecto al dominio de entrenamiento).
GENERADORES = {
    "StyleGAN2_CelebA": "StyleGAN2",
    "StyleGAN3_CelebA": "StyleGAN3",
    "SDXL_CelebA": "SDXL",
    "Flux_CelebA": "FLUX.1",
}

EXPERIMENTO_META = {
    "A": {
        "titulo": "A — Línea base",
        "generador": "StyleGAN2",
        "descripcion": "Entrenamiento y evaluación en el mismo dominio GAN (StyleGAN2).",
    },
    "B": {
        "titulo": "B — Generalización GAN",
        "generador": "StyleGAN3",
        "descripcion": "Evaluación en StyleGAN3 sin reentrenar: mismo tipo de generador, huella distinta.",
    },
    "C": {
        "titulo": "C — Generalización a difusión",
        "generador": "SDXL",
        "descripcion": "Evaluación en SDXL sin reentrenar: cambio de familia generativa (GAN → difusión).",
    },
    "D": {
        "titulo": "D — Generalización a difusión",
        "generador": "FLUX.1",
        "descripcion": "Evaluación en FLUX.1 sin reentrenar, el segundo modelo de difusión del estudio.",
    },
    "E": {
        "titulo": "E — Ajuste fino progresivo",
        "generador": "Los cuatro",
        "descripcion": "Adaptación incremental FT-1/FT-2/FT-3 hasta cubrir los cuatro generadores.",
    },
}

FASES_E = ["FT-1", "FT-2", "FT-3"]

RUTA_METRICAS = Path(__file__).resolve().parent.parent / "Resultados" / "Resultados Experimentos"


def _cargar_json(ruta: Path) -> dict:
    with open(ruta, encoding="utf-8") as h:
        return json.load(h)


def _parse_semilla(ruta: Path) -> int | None:
    nombre = ruta.stem
    if "_semilla" in nombre:
        try:
            return int(nombre.rsplit("_semilla", 1)[1])
        except ValueError:
            return None
    return None


def _especificidad(matriz: list[list[int]] | None) -> float | None:
    """VN / (VN + FP): proporción de rostros REALES bien clasificados.

    La matriz viene como [[VP, FN], [FP, VN]]: la fila 0 es la clase sintética
    y la fila 1 la real, con la diagonal como aciertos. La especificidad no se
    guardó en los .json de los experimentos, así que se deriva aquí a partir de
    la matriz de cada semilla, igual que en las tablas de la tesis.
    """
    if not matriz or len(matriz) < 2 or len(matriz[1]) < 2:
        return None
    falsos_positivos, verdaderos_negativos = matriz[1][0], matriz[1][1]
    total_reales = falsos_positivos + verdaderos_negativos
    return verdaderos_negativos / total_reales if total_reales > 0 else None


def _normalizar(metricas: dict) -> dict:
    """Deja sólo las métricas reportadas y añade la especificidad derivada."""
    fila = {clave: metricas[clave] for clave in METRICAS if clave in metricas}
    especificidad = _especificidad(metricas.get("matriz_confusion"))
    if especificidad is not None:
        fila["especificidad"] = especificidad
    return fila


def _sumar_matrices(matrices: list[list[list[int]]]) -> list[list[int]] | None:
    """Suma element-wise varias matrices de confusión 2x2 (una por semilla).

    En la tesis las matrices de confusión no se promedian: se suman las tres
    semillas, de modo que los conteos mostrados son el total de imágenes
    evaluadas en las tres corridas."""
    matrices_validas = [m for m in matrices if m]
    if not matrices_validas:
        return None
    filas, columnas = len(matrices_validas[0]), len(matrices_validas[0][0])
    return [
        [sum(m[i][j] for m in matrices_validas) for j in range(columnas)]
        for i in range(filas)
    ]


def _stat(valores: list[float]) -> dict | None:
    """Media ± desviación estándar muestral (n-1), como se reporta en la tesis."""
    valores = [v for v in valores if isinstance(v, (int, float))]
    if not valores:
        return None
    return {
        "media": statistics.mean(valores),
        "desv_est": statistics.stdev(valores) if len(valores) > 1 else 0.0,
    }


def _media_desviacion_metricas(lista: list[dict]) -> dict:
    """Agrega una lista de métricas por semilla a media ± desviación estándar."""
    if not lista:
        return {}
    resultado = {}
    for clave in METRICAS:
        stat = _stat([item[clave] for item in lista if clave in item])
        if stat is not None:
            resultado[clave] = stat
    return resultado


def _agregar(metricas_por_semilla: list[dict]) -> dict:
    """Métricas agregadas + matriz de confusión sumada sobre las semillas."""
    normalizadas = [_normalizar(m) for m in metricas_por_semilla]
    agregado = _media_desviacion_metricas(normalizadas)
    matriz = _sumar_matrices([m.get("matriz_confusion", []) for m in metricas_por_semilla])
    if matriz is not None:
        agregado["matriz_confusion"] = {"matriz": matriz}
    agregado["n_semillas"] = len(metricas_por_semilla)
    return agregado


def _cargar_experimento_a(arquitectura: str) -> dict | None:
    patrones = list(RUTA_METRICAS.glob(f"experimentoA_{arquitectura}_StyleGAN2_CelebA*.json"))
    if not patrones:
        return None

    resultados_por_semilla: dict[int, dict] = {}
    for ruta in patrones:
        datos = _cargar_json(ruta)
        for fila in datos.get("resultados_por_semilla", []):
            semilla = fila.get("semilla")
            if semilla is None:
                continue
            resultados_por_semilla[semilla] = fila.get("metricas_test", {})

    if not resultados_por_semilla:
        return None

    return _agregar([resultados_por_semilla[s] for s in sorted(resultados_por_semilla)])


def _cargar_experimento_bcd(arquitectura: str, base: dict | None) -> dict[str, dict] | None:
    ruta = RUTA_METRICAS / f"experimentosBCD_{arquitectura}.json"
    if not ruta.exists():
        return None

    datos = _cargar_json(ruta)
    por_experimento: dict[str, dict[int, dict]] = defaultdict(dict)
    for item in datos.get("evaluaciones", []):
        if item.get("arquitectura") != arquitectura:
            continue
        experimento = item.get("experimento")
        semilla = item.get("semilla")
        if experimento and semilla is not None:
            por_experimento[experimento][semilla] = item

    auc_base = (base or {}).get("auc", {}).get("media")

    resultado: dict[str, dict] = {}
    for experimento in ["B", "C", "D"]:
        semillas = por_experimento.get(experimento, {})
        if not semillas:
            continue
        items = [semillas[s] for s in sorted(semillas)]
        metricas = _agregar([item["metricas"] for item in items])
        # La tesis cuantifica la degradación en puntos porcentuales de AUC-ROC
        # respecto al Experimento A, que es la métrica principal del estudio.
        auc_exp = metricas.get("auc", {}).get("media")
        degradacion_auc = (auc_base - auc_exp) if (auc_base is not None and auc_exp is not None) else None
        resultado[experimento] = {
            "metricas": metricas,
            "degradacion_auc": degradacion_auc,
        }
    return resultado


def _cargar_experimento_e(arquitectura: str) -> dict | None:
    rutas = sorted(RUTA_METRICAS.glob(f"experimentoE_{arquitectura}_semilla*.json"))
    if not rutas:
        return None

    datos_por_semilla: dict[int, dict] = {}
    for ruta in rutas:
        semilla = _parse_semilla(ruta)
        if semilla is None:
            continue
        datos_por_semilla[semilla] = _cargar_json(ruta)

    if not datos_por_semilla:
        return None

    corridas = [datos_por_semilla[s] for s in sorted(datos_por_semilla)]

    def _etapa(evaluaciones: list[dict], vistos: list[str]) -> dict:
        """Agrega una etapa (evaluación inicial o una fase de fine-tuning).

        `evaluaciones` trae un dict {generador: métricas} por semilla. Se agrega
        cada generador por separado y, además, el promedio sobre los cuatro, que
        es el valor que resume la tesis en la tabla comparativa final."""
        por_generador = {}
        for generador in GENERADORES:
            metricas_semillas = [ev[generador] for ev in evaluaciones if generador in ev]
            if not metricas_semillas:
                continue
            por_generador[generador] = {
                **_agregar(metricas_semillas),
                "visto": generador in vistos,
            }

        # Promedio sobre generadores dentro de cada semilla y después entre
        # semillas: así la desviación estándar sigue midiendo variación por
        # inicialización y no mezcla la dispersión entre generadores.
        promedios_por_semilla = []
        for ev in evaluaciones:
            fila = [_normalizar(ev[g]) for g in GENERADORES if g in ev]
            if fila:
                promedios_por_semilla.append({
                    clave: statistics.mean([f[clave] for f in fila if clave in f])
                    for clave in METRICAS
                    if any(clave in f for f in fila)
                })
        promedio = _media_desviacion_metricas(promedios_por_semilla)

        return {"por_generador": por_generador, "promedio": promedio}

    inicial = _etapa(
        [datos.get("evaluacion_inicial", {}) for datos in corridas],
        vistos=["StyleGAN2_CelebA"],
    )

    fases_agg: list[dict] = []
    for fase in FASES_E:
        registros = []
        for datos in corridas:
            info = next((f for f in datos.get("fases", []) if f.get("fase") == fase), None)
            if info is not None:
                registros.append(info)
        if not registros:
            continue

        vistos = registros[0].get("generadores_entrenamiento", [])
        etapa = _etapa([r.get("evaluacion", {}) for r in registros], vistos=vistos)
        fases_agg.append({
            "fase": fase,
            "generadores_entrenamiento": vistos,
            "mejor_acc_val": _stat([r.get("mejor_acc_val") for r in registros]),
            **etapa,
        })

    return {
        "inicial": inicial,
        "fases": fases_agg,
        "final": fases_agg[-1] if fases_agg else None,
    }


def cargar_experimentos() -> dict:
    datos: dict[str, dict] = {}
    for arquitectura in ARCHITECTURA_LABELS:
        a = _cargar_experimento_a(arquitectura)
        bcd = _cargar_experimento_bcd(arquitectura, a)
        e = _cargar_experimento_e(arquitectura)
        if a is None and bcd is None and e is None:
            continue

        datos[arquitectura] = {
            "A": a,
            "B": bcd.get("B") if bcd else None,
            "C": bcd.get("C") if bcd else None,
            "D": bcd.get("D") if bcd else None,
            "E": e,
        }

    return {
        "arquitecturas": ARCHITECTURA_LABELS,
        "metricas": METRICAS,
        "generadores": GENERADORES,
        "experimentos": EXPERIMENTO_META,
        "datos": datos,
    }
