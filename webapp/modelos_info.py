"""
modelos_info.py
Metadatos reales (arquitectura, parámetros, tamaño en disco, fecha) de los
tres modelos CNN entrenados en la tesis, para la página "Modelos y
descargas". Todo se calcula a partir de archivos .pth reales en disco y de
la propia arquitectura de timm, no son cifras inventadas.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

RUTA_TESIS = Path(__file__).resolve().parent.parent
RUTA_DATASETS = RUTA_TESIS / "Datasets"
sys.path.insert(0, str(RUTA_DATASETS))

import timm  # noqa: E402

RUTA_MULTICLASE = RUTA_TESIS / "Resultados" / "modelos" / "multiclase" / "multiclase_efficientnet_b0_semilla42.pth"
RUTA_EXPERIMENTOS = RUTA_TESIS / "Resultados" / "Resultados Experimentos"

MODELOS = [
    {
        "id": "efficientnet_b0",
        "etiqueta": "EfficientNet-B0",
        "timm_id": "efficientnet_b0",
        "num_clases": 5,
        "descripcion": "Modelo multiclase (Flux / SDXL / StyleGAN2 / StyleGAN3 / real) en producción en la página de Análisis.",
        "ruta": RUTA_MULTICLASE,
    },
    {
        "id": "resnet50",
        "etiqueta": "ResNet50",
        "timm_id": "resnet50",
        "num_clases": 2,
        "descripcion": "Modelo binario (StyleGAN2 vs. real) del Experimento A, semilla 42.",
        "ruta": RUTA_EXPERIMENTOS / "resnet50_StyleGAN2_CelebA_semilla42.pth",
    },
    {
        "id": "legacy_xception",
        "etiqueta": "Xception",
        "timm_id": "legacy_xception",
        "num_clases": 2,
        "descripcion": "Modelo binario (StyleGAN2 vs. real) del Experimento A, semilla 42.",
        "ruta": RUTA_EXPERIMENTOS / "legacy_xception_StyleGAN2_CelebA_semilla42.pth",
    },
]

_CACHE_PARAMETROS: dict[str, int] = {}


def _contar_parametros(timm_id: str, num_clases: int) -> int:
    clave = f"{timm_id}:{num_clases}"
    if clave not in _CACHE_PARAMETROS:
        modelo = timm.create_model(timm_id, pretrained=False, num_classes=num_clases)
        _CACHE_PARAMETROS[clave] = sum(p.numel() for p in modelo.parameters())
    return _CACHE_PARAMETROS[clave]


def obtener_modelos() -> list[dict]:
    resultado = []
    for m in MODELOS:
        ruta: Path = m["ruta"]
        existe = ruta.exists()
        item = {
            "id": m["id"],
            "etiqueta": m["etiqueta"],
            "descripcion": m["descripcion"],
            "parametros": _contar_parametros(m["timm_id"], m["num_clases"]),
            "disponible": existe,
        }
        if existe:
            stat = ruta.stat()
            item["tamano_mb"] = round(stat.st_size / (1024 * 1024), 1)
            item["fecha"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
            item["archivo"] = ruta.name
        resultado.append(item)
    return resultado


def ruta_modelo(modelo_id: str) -> Path | None:
    for m in MODELOS:
        if m["id"] == modelo_id and m["ruta"].exists():
            return m["ruta"]
    return None
