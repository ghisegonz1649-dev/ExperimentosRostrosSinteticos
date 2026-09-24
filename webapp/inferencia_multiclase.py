"""
inferencia_multiclase.py
Carga el modelo MULTICLASE ya entrenado (Flux / SDXL / StyleGAN2 / StyleGAN3
/ real) y predice, para UNA imagen de rostro, a qué clase pertenece.

Reutiliza config.py y nucleo.py de Datasets/ (mismo motor que el
entrenamiento), así que la arquitectura y las clases siempre coinciden con
las que se usaron para entrenar. El preprocesamiento (recorte central
cuadrado, resize a 256x256, igualación de historial de compresión JPEG) es
el MISMO que en Crear_Particiones_Multiclase.py: si la imagen subida no pasa
por los mismos pasos, el modelo la vería con una distribución distinta a la
de entrenamiento y las probabilidades no serían confiables.

No depende de Flask: se puede usar también desde una consola o notebook.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RUTA_TESIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUTA_TESIS / "Codigos Experimentos"))

import config  # noqa: E402
import nucleo  # noqa: E402

# Carpeta donde se espera encontrar el/los .pth del experimento multiclase
# (bajados manualmente de Drive tras correr experimento_multiclase.py).
RUTA_MODELOS_MULTICLASE = config.RUTA_MODELOS / "multiclase"

# Mismos parámetros de preprocesamiento que Crear_Particiones_Multiclase.py.
TAMANO_OBJETIVO = 256
CALIDAD_JPEG_COMUN = 90


class ModeloNoEntrenadoError(RuntimeError):
    """No se encontró ningún .pth del experimento multiclase."""


def _preprocesar(imagen: Image.Image) -> Image.Image:
    """Recorte central cuadrado + resize Lanczos a 256x256 + igualación de
    historial de compresión (recompresión JPEG común). Idéntico al
    preprocesamiento aplicado a las imágenes de entrenamiento."""
    imagen = imagen.convert("RGB")

    ancho, alto = imagen.size
    lado = min(ancho, alto)
    izq = (ancho - lado) // 2
    arriba = (alto - lado) // 2
    imagen = imagen.crop((izq, arriba, izq + lado, arriba + lado))

    imagen = imagen.resize((TAMANO_OBJETIVO, TAMANO_OBJETIVO), Image.LANCZOS)

    buffer = io.BytesIO()
    imagen.save(buffer, format="JPEG", quality=CALIDAD_JPEG_COMUN)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _a_tensor(imagen: Image.Image) -> torch.Tensor:
    """Convierte una imagen PIL ya preprocesada (256x256 RGB) al tensor
    normalizado que espera el modelo (misma normalización de ImageNet que
    en entrenamiento, sin aumentos aleatorios -- equivalente a val/test)."""
    from torchvision import transforms

    transformacion = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEDIA, std=config.IMAGENET_DESV),
    ])
    return transformacion(imagen).unsqueeze(0)  # añade dimensión de batch


# Cajas normalizadas (0-1) usadas para resumir en qué zona del rostro se
# concentró la atención del modelo. Son aproximaciones geométricas (el
# preprocesamiento ya centra y recorta el rostro en un cuadrado), no una
# detección de landmarks real.
_REGIONES_ROSTRO = {
    "Ojos": (0.15, 0.20, 0.85, 0.40),
    "Nariz": (0.35, 0.38, 0.65, 0.60),
    "Boca": (0.28, 0.62, 0.72, 0.80),
}


def _analizar_regiones(mapa_normalizado: np.ndarray) -> list[dict]:
    """Intensidad promedio del mapa de Grad-CAM (0-1, ya en resolución de
    imagen completa) dentro de zonas aproximadas del rostro."""
    alto, ancho = mapa_normalizado.shape
    y, x = np.mgrid[0:alto, 0:ancho]
    yn, xn = y / alto, x / ancho

    resultado = []
    for nombre, (x0, y0, x1, y1) in _REGIONES_ROSTRO.items():
        mascara = (xn >= x0) & (xn <= x1) & (yn >= y0) & (yn <= y1)
        valores = mapa_normalizado[mascara]
        resultado.append({
            "region": nombre,
            "intensidad": float(valores.mean()) if valores.size else 0.0,
        })

    mascara_contorno = (yn <= 0.12) | (yn >= 0.88) | (xn <= 0.15) | (xn >= 0.85)
    valores_contorno = mapa_normalizado[mascara_contorno]
    resultado.append({
        "region": "Contorno facial",
        "intensidad": float(valores_contorno.mean()) if valores_contorno.size else 0.0,
    })

    resultado.sort(key=lambda r: r["intensidad"], reverse=True)
    return resultado


def _imagen_a_data_url(imagen: Image.Image) -> str:
    """Convierte un PIL Image a un data URI PNG para mostrar en la web."""
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("utf-8")


class GradCAM:
    """Grad-CAM simple que funciona sobre la última capa convolucional."""

    def __init__(self, modelo: torch.nn.Module, capa_objetivo: torch.nn.Module):
        self.activaciones: torch.Tensor | None = None
        self.gradientes: torch.Tensor | None = None
        self.capa_objetivo = capa_objetivo
        self.capa_objetivo.register_forward_hook(self._guardar_activaciones)
        if hasattr(self.capa_objetivo, "register_full_backward_hook"):
            self.capa_objetivo.register_full_backward_hook(self._guardar_gradientes)
        else:
            self.capa_objetivo.register_backward_hook(self._guardar_gradientes)

    def _guardar_activaciones(self, module, input, output):
        self.activaciones = output.detach()

    def _guardar_gradientes(self, module, grad_input, grad_output):
        self.gradientes = grad_output[0].detach()

    def calcular(self, modelo: torch.nn.Module, entrada: torch.Tensor,
                 indice_clase: int | None = None) -> np.ndarray:
        salida = modelo(entrada)
        if indice_clase is None:
            indice_clase = int(torch.argmax(salida, dim=1).item())

        objetivo = salida[0, indice_clase]
        modelo.zero_grad()
        objetivo.backward(retain_graph=True)

        if self.activaciones is None or self.gradientes is None:
            raise RuntimeError("No se pudo capturar activaciones o gradientes para Grad-CAM.")

        gradientes = self.gradientes[0]
        activaciones = self.activaciones[0]
        pesos = gradientes.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((pesos * activaciones).sum(dim=0, keepdim=False))

        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.cpu().numpy()
        return cam_np


def encontrar_modelo(ruta: Path = None) -> Path:
    """Devuelve la ruta a un .pth del experimento multiclase. Si no se pasa
    una ruta explícita, busca en RUTA_MODELOS_MULTICLASE y toma el más
    reciente (útil mientras solo hay una arquitectura/semilla descargada)."""
    if ruta is not None:
        ruta = Path(ruta)
        if not ruta.exists():
            raise ModeloNoEntrenadoError(f"No existe el modelo indicado: {ruta}")
        return ruta

    if not RUTA_MODELOS_MULTICLASE.exists():
        raise ModeloNoEntrenadoError(
            f"No existe la carpeta {RUTA_MODELOS_MULTICLASE}. "
            f"Corré experimento_multiclase.py y bajá el/los .pth resultantes "
            f"a esa carpeta antes de usar la página web."
        )

    candidatos = sorted(RUTA_MODELOS_MULTICLASE.glob("*.pth"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidatos:
        raise ModeloNoEntrenadoError(
            f"{RUTA_MODELOS_MULTICLASE} existe pero no tiene ningún .pth todavía. "
            f"Corré experimento_multiclase.py y bajá el modelo entrenado ahí."
        )
    return candidatos[0]


class ClasificadorMulticlase:
    """Envuelve el modelo cargado para reutilizarlo entre pedidos (evita
    recargar los pesos en cada petición de la página web)."""

    def __init__(self, ruta_modelo: Path = None):
        self.ruta_modelo = encontrar_modelo(ruta_modelo)
        self.clases = config.CLASES_MULTICLASE
        self.modelo = nucleo.crear_modelo(num_clases=config.NUM_CLASES_MULTICLASE)
        estado = torch.load(self.ruta_modelo, map_location=config.DISPOSITIVO)
        self.modelo.load_state_dict(estado)
        self.modelo.eval()
        self.gradcam = GradCAM(self.modelo, self.modelo.conv_head)

    @torch.no_grad()
    def predecir(self, imagen: Image.Image) -> dict:
        """Predice la clase de una imagen PIL. Devuelve un diccionario con
        la clase más probable y las probabilidades de las 5 clases,
        ordenadas de mayor a menor."""
        imagen_procesada = _preprocesar(imagen)
        tensor = _a_tensor(imagen_procesada).to(config.DISPOSITIVO)

        salidas = self.modelo(tensor)
        probabilidades = torch.softmax(salidas.float(), dim=1)[0].cpu().numpy()

        pares = sorted(zip(self.clases, probabilidades),
                       key=lambda par: par[1], reverse=True)

        return {
            "clase_predicha": pares[0][0],
            "confianza": float(pares[0][1]),
            "probabilidades": [
                {"clase": clase, "probabilidad": float(p)} for clase, p in pares
            ],
            "modelo_usado": self.ruta_modelo.name,
        }

    def generar_gradcam(self, imagen: Image.Image, indice_clase: int | None = None) -> dict:
        imagen_procesada = _preprocesar(imagen)
        tensor = _a_tensor(imagen_procesada).to(config.DISPOSITIVO)

        if indice_clase is None:
            salidas = self.modelo(tensor)
            indice_clase = int(torch.argmax(salidas, dim=1).item())

        mapa = self.gradcam.calcular(self.modelo, tensor, indice_clase)
        mapa = np.uint8(np.clip(mapa * 255.0, 0, 255))
        imagen_calor = Image.fromarray(mapa, mode="L").resize(
            (TAMANO_OBJETIVO, TAMANO_OBJETIVO), Image.BILINEAR)

        color = np.zeros((TAMANO_OBJETIVO, TAMANO_OBJETIVO, 4), dtype=np.uint8)
        calor_np = np.asarray(imagen_calor)
        color[..., 0] = calor_np
        color[..., 3] = np.clip(calor_np.astype(float) * 0.8, 0, 255).astype(np.uint8)
        overlay = Image.alpha_composite(imagen_procesada.convert("RGBA"),
                                       Image.fromarray(color, mode="RGBA"))

        heatmap_color = np.zeros((TAMANO_OBJETIVO, TAMANO_OBJETIVO, 3), dtype=np.uint8)
        heatmap_color[..., 0] = calor_np
        heatmap_color[..., 1] = np.minimum(calor_np // 2 + 60, 255)
        heatmap_color[..., 2] = 255
        heatmap = Image.fromarray(heatmap_color, mode="RGB")

        mapa_normalizado = calor_np.astype(np.float64) / 255.0
        regiones = _analizar_regiones(mapa_normalizado)
        intensidad_promedio = float(mapa_normalizado.mean())
        pico = float(mapa_normalizado.max())
        concentracion = (pico - intensidad_promedio) / pico if pico > 0 else 0.0
        confianza_explicacion = float(np.clip(concentracion, 0.0, 1.0))

        return {
            "original": _imagen_a_data_url(imagen_procesada),
            "overlay": _imagen_a_data_url(overlay),
            "heatmap": _imagen_a_data_url(heatmap),
            "clase_predicha": self.clases[indice_clase],
            "regiones": regiones,
            "intensidad_promedio": intensidad_promedio,
            "confianza_explicacion": confianza_explicacion,
        }
