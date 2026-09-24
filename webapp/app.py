"""
app.py
Backend API (JSON) para la plataforma de detección y atribución de rostros
sintéticos (Flux / SDXL / StyleGAN2 / StyleGAN3) o reales (FFHQ / CelebA-HQ),
usando el modelo del experimento multiclase.

Uso:
    cd webapp
    python app.py
    -> queda escuchando en http://127.0.0.1:5001
       (el 5000 lo ocupa el receptor AirPlay de macOS)

El frontend (Next.js, webapp/frontend, puerto 3000 en desarrollo) consume
estos endpoints /api/* como JSON. Esta app ya no sirve páginas HTML propias
(las páginas Jinja del escáner/telemetría/sonda térmica se retiraron al
migrar todo a Next.js).

El modelo se carga UNA sola vez al arrancar (no en cada petición). Si
todavía no corriste experimento_multiclase.py, el arranque falla con un
mensaje explicando qué falta (ver inferencia_multiclase.encontrar_modelo).

ULTIMO_ANALISIS guarda en memoria (un solo usuario local) el último rostro
subido vía /api/analizar, para que /api/gradcam pueda generar su Grad-CAM
sin tener que volver a subir la imagen.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

from experimentos import ARCHITECTURA_LABELS, cargar_experimentos
from inferencia_multiclase import ClasificadorMulticlase, ModeloNoEntrenadoError
import modelos_info
import recursos_info

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB por imagen
# Sin esto Flask ordena las claves alfabéticamente y se pierde el orden en que
# se declararon los generadores y las métricas, que el frontend usa para
# mostrarlos en el mismo orden que las tablas de la tesis.
app.json.sort_keys = False

CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

EXTENSIONES_VALIDAS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Descripción amigable de cada clase para mostrar en la página.
DESCRIPCION_CLASE = {
    "Flux": "Generada con Flux (modelo de difusión)",
    "SDXL": "Generada con Stable Diffusion XL (modelo de difusión)",
    "StyleGAN2": "Generada con StyleGAN2 (GAN)",
    "StyleGAN3": "Generada con StyleGAN3 (GAN)",
    "real": "Rostro real",
}

clasificador = None  # se carga en cargar_clasificador()
EXPERIMENTOS = cargar_experimentos()
DEFAULT_ARQUITECTURA = next(iter(EXPERIMENTOS["arquitecturas"].keys()))

# Último rostro subido vía /api/analizar: {"bytes", "filename", "prediccion"}.
# Permite que /api/gradcam genere su Grad-CAM sin pedir la imagen de nuevo.
ULTIMO_ANALISIS: dict | None = None


def _bytes_a_data_url(contenido: bytes, nombre_archivo: str) -> str:
    extension = Path(nombre_archivo).suffix.lower()
    if extension in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif extension == ".webp":
        mime = "image/webp"
    elif extension == ".bmp":
        mime = "image/bmp"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(contenido).decode('utf-8')}"


def _buscar_ejemplos_galeria() -> dict[str, str]:
    ruta_test = Path(__file__).resolve().parent.parent / "Datasets" / "Particiones" / "Experimentos" / "Multiclase" / "test"
    ejemplos: dict[str, str] = {}
    for clase in ["StyleGAN2", "StyleGAN3", "SDXL", "Flux", "real"]:
        ruta_carpeta = ruta_test / clase
        if ruta_carpeta.exists() and ruta_carpeta.is_dir():
            primer_archivo = next(ruta_carpeta.glob("*.*"), None)
            if primer_archivo is not None:
                try:
                    contenido = primer_archivo.read_bytes()
                    ejemplos[clase] = _bytes_a_data_url(contenido, primer_archivo.name)
                except Exception:
                    pass
    return ejemplos


GALLERY_EJEMPLOS = _buscar_ejemplos_galeria()


def cargar_clasificador() -> ClasificadorMulticlase:
    global clasificador
    if clasificador is None:
        clasificador = ClasificadorMulticlase()
    return clasificador


def _enriquecer_prediccion(prediccion: dict) -> dict:
    """Agrega descripción legible y porcentaje redondeado a cada clase."""
    for item in prediccion["probabilidades"]:
        item["descripcion"] = DESCRIPCION_CLASE.get(item["clase"], item["clase"])
        item["porcentaje"] = round(item["probabilidad"] * 100, 1)
    prediccion["descripcion_predicha"] = DESCRIPCION_CLASE.get(
        prediccion["clase_predicha"], prediccion["clase_predicha"])
    prediccion["es_sintetico"] = prediccion["clase_predicha"] != "real"
    return prediccion


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Cifras fijas del proyecto de tesis, para la página de inicio."""
    return jsonify({
        "generadores": len(DESCRIPCION_CLASE) - 1,  # sin contar "real"
        "arquitecturas_cnn": len(EXPERIMENTOS["arquitecturas"]),
        "imagenes_dataset": 91000,
        "experimentos": len(EXPERIMENTOS["experimentos"]),
    })


@app.route("/api/muestras", methods=["GET"])
def api_muestras():
    """Un ejemplo real y uno sintético del propio dataset, para componer el
    rostro dividido humano/sintético del hero de la landing."""
    clase_sintetico = next(
        (c for c in ["StyleGAN3", "SDXL", "StyleGAN2", "Flux"] if GALLERY_EJEMPLOS.get(c)),
        None,
    )
    return jsonify({
        "real": GALLERY_EJEMPLOS.get("real"),
        "sintetico": GALLERY_EJEMPLOS.get(clase_sintetico) if clase_sintetico else None,
        "clase_sintetico": clase_sintetico,
    })


@app.route("/api/analizar", methods=["POST"])
def api_analizar():
    """Analiza una imagen subida y devuelve la predicción real del modelo
    entrenado, para el frontend Next.js."""
    archivo = request.files.get("imagen")
    if archivo is None or archivo.filename == "":
        return jsonify({"error": "Elegí una imagen antes de enviar."}), 400

    extension = Path(archivo.filename).suffix.lower()
    if extension not in EXTENSIONES_VALIDAS:
        return jsonify({
            "error": f"Formato no soportado ({extension or 'sin extensión'}). "
                     f"Usá PNG, JPG, JPEG, WEBP o BMP.",
        }), 400

    try:
        contenido = archivo.read()
        imagen = Image.open(io.BytesIO(contenido))
        imagen.load()
    except UnidentifiedImageError:
        return jsonify({"error": "No pude leer esa imagen. ¿Está corrupta?"}), 400

    try:
        modelo = cargar_clasificador()
    except ModeloNoEntrenadoError as e:
        return jsonify({"error": str(e)}), 503

    inicio = time.perf_counter()
    prediccion = _enriquecer_prediccion(modelo.predecir(imagen))
    prediccion["tiempo_inferencia_ms"] = round((time.perf_counter() - inicio) * 1000, 1)
    prediccion["imagen_data_url"] = _bytes_a_data_url(contenido, archivo.filename)
    prediccion["arquitectura"] = ARCHITECTURA_LABELS.get(DEFAULT_ARQUITECTURA, DEFAULT_ARQUITECTURA)

    global ULTIMO_ANALISIS
    ULTIMO_ANALISIS = {
        "bytes": contenido,
        "filename": archivo.filename,
        "prediccion": prediccion,
    }
    return jsonify(prediccion)


@app.route("/api/experimentos", methods=["GET"])
def api_experimentos():
    """Resultados reales de los experimentos A-E (todas las arquitecturas
    disponibles), para el dashboard científico del frontend."""
    return jsonify({
        **EXPERIMENTOS,
        "arquitectura_predeterminada": DEFAULT_ARQUITECTURA,
    })


@app.route("/api/modelos", methods=["GET"])
def api_modelos():
    return jsonify(modelos_info.obtener_modelos())


@app.route("/api/modelos/<modelo_id>/descargar", methods=["GET"])
def api_descargar_modelo(modelo_id):
    ruta = modelos_info.ruta_modelo(modelo_id)
    if ruta is None:
        return jsonify({"error": "Ese modelo no está disponible."}), 404
    return send_file(ruta, as_attachment=True, download_name=ruta.name)


@app.route("/api/recursos", methods=["GET"])
def api_recursos():
    return jsonify(recursos_info.obtener_recursos())


@app.route("/api/recursos/documentacion/<doc_id>/descargar", methods=["GET"])
def api_descargar_documento(doc_id):
    ruta = recursos_info.ruta_documento(doc_id)
    if ruta is None:
        return jsonify({"error": "Ese documento no está disponible."}), 404
    return send_file(ruta, as_attachment=True, download_name=ruta.name)


@app.route("/api/recursos/codigo/<codigo_id>/descargar", methods=["GET"])
def api_descargar_codigo(codigo_id):
    ruta = recursos_info.ruta_codigo(codigo_id)
    if ruta is None:
        return jsonify({"error": "Ese archivo no está disponible."}), 404
    return send_file(ruta, as_attachment=True, download_name=ruta.name)


@app.route("/api/gradcam", methods=["GET", "DELETE"])
def api_gradcam():
    """Grad-CAM del último rostro analizado (vía /api/analizar). Stateless
    del lado del cliente: reutiliza ULTIMO_ANALISIS guardado en el servidor.

    DELETE limpia ese estado para que, al volver a /gradcam sin analizar de
    nuevo, la página arranque desde 0 (estado vacío)."""
    global ULTIMO_ANALISIS
    if request.method == "DELETE":
        ULTIMO_ANALISIS = None
        return ("", 204)

    if ULTIMO_ANALISIS is None:
        return jsonify({"error": "Todavía no analizaste ninguna imagen."}), 404

    try:
        modelo = cargar_clasificador()
    except ModeloNoEntrenadoError as e:
        return jsonify({"error": str(e)}), 503

    imagen = Image.open(io.BytesIO(ULTIMO_ANALISIS["bytes"]))
    resultado = modelo.generar_gradcam(imagen)
    resultado["descripcion_predicha"] = DESCRIPCION_CLASE.get(
        resultado["clase_predicha"], resultado["clase_predicha"])
    respuesta = jsonify(resultado)
    respuesta.headers["Cache-Control"] = "no-store, max-age=0"
    return respuesta


if __name__ == "__main__":
    # Cargar el modelo al arrancar (y no en el primer pedido) para que el
    # error de "no hay modelo entrenado todavía", si aparece, sea inmediato.
    try:
        cargar_clasificador()
        print("Modelo multiclase cargado correctamente.")
    except ModeloNoEntrenadoError as e:
        print(f"[AVISO] {e}")
        print("El servidor va a levantar igual, pero /api/analizar va a fallar "
              "hasta que haya un modelo disponible.")

    app.run(debug=True, host="127.0.0.1", port=5001)
