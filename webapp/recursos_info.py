"""
recursos_info.py
Metadatos reales de los recursos descargables de la tesis (documentación y
código fuente) más los conteos reales de imágenes por partición, para la
sección de recursos de la página "Modelos y descargas".
"""

from __future__ import annotations

from pathlib import Path

RUTA_TESIS = Path(__file__).resolve().parent.parent
RUTA_DATASETS = RUTA_TESIS / "Datasets"
RUTA_MULTICLASE = RUTA_DATASETS / "Particiones" / "Experimentos" / "Multiclase"

DOCUMENTOS = [
    {"id": "tesis-pdf", "etiqueta": "Tesis (versión final para sínodo)", "ruta": RUTA_TESIS / "Tesis final para sinodo.pdf"},
    {"id": "tesis-docx", "etiqueta": "Tesis CNN (borrador actualizado)", "ruta": RUTA_TESIS / "Tesis_CNN_2_clrp_ACTUALIZADA.docx"},
    {"id": "estructura", "etiqueta": "Estructura de la tesis", "ruta": RUTA_TESIS / "Estructura_Tesis_ACTUALIZADA.docx"},
]

CODIGO = [
    {"id": "config", "etiqueta": "config.py", "ruta": RUTA_DATASETS / "config.py"},
    {"id": "nucleo", "etiqueta": "nucleo.py", "ruta": RUTA_DATASETS / "nucleo.py"},
    {"id": "experimento-a", "etiqueta": "experimento_A.py", "ruta": RUTA_TESIS / "experimento_A.py"},
    {"id": "experimentos-bcd", "etiqueta": "experimentos_bcd.py", "ruta": RUTA_TESIS / "experimentos_bcd.py"},
    {"id": "experimento-multiclase", "etiqueta": "experimento_multiclase.py", "ruta": RUTA_TESIS / "experimento_multiclase.py"},
    {"id": "app-flask", "etiqueta": "app.py (backend web)", "ruta": RUTA_TESIS / "webapp" / "app.py"},
]

_CACHE_PARTICIONES: dict[str, int] | None = None


def _contar_particiones() -> dict[str, int]:
    global _CACHE_PARTICIONES
    if _CACHE_PARTICIONES is None:
        conteo = {}
        for split in ["train", "val", "test"]:
            carpeta = RUTA_MULTICLASE / split
            conteo[split] = sum(1 for _ in carpeta.rglob("*.*")) if carpeta.exists() else 0
        _CACHE_PARTICIONES = conteo
    return _CACHE_PARTICIONES


def _meta_archivo(item: dict) -> dict | None:
    ruta: Path = item["ruta"]
    if not ruta.exists():
        return None
    stat = ruta.stat()
    return {
        "id": item["id"],
        "etiqueta": item["etiqueta"],
        "archivo": ruta.name,
        "tamano_kb": round(stat.st_size / 1024, 1),
    }


def obtener_recursos() -> dict:
    return {
        "particiones": _contar_particiones(),
        "documentacion": [m for d in DOCUMENTOS if (m := _meta_archivo(d))],
        "codigo": [m for c in CODIGO if (m := _meta_archivo(c))],
    }


def ruta_documento(doc_id: str) -> Path | None:
    for d in DOCUMENTOS:
        if d["id"] == doc_id and d["ruta"].exists():
            return d["ruta"]
    return None


def ruta_codigo(codigo_id: str) -> Path | None:
    for c in CODIGO:
        if c["id"] == codigo_id and c["ruta"].exists():
            return c["ruta"]
    return None
