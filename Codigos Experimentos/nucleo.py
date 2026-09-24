"""
nucleo.py
Motor de entrenamiento y evaluación.
  - Construcción de DataLoaders (lee las carpetas de Particiones/)
  - Construcción del modelo 
  - Entrenamiento de una época
  - Evaluación con todas las métricas (accuracy, precision, recall, F1, AUC)
  - Bucle de entrenamiento completo con early stopping

Convención de clases (ImageFolder ordena alfabéticamente):
    fake -> 0  (clase POSITIVA: es lo que queremos detectar)
    real -> 1
"""
import io
import random
from pathlib import Path
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config

class RecompresionJPEGAleatoria:
    """Aumento de robustez: con probabilidad `p`, recomprime la imagen a una
    calidad JPEG aleatoria dentro de `calidad_rango`. Evita que "no tener
    historial de compresión" sea un atajo fijo para distinguir clases."""
    def __init__(self, p: float, calidad_rango: tuple[int, int]):
        self.p = p
        self.calidad_rango = calidad_rango

    def __call__(self, img):
        if random.random() > self.p:
            return img
        calidad = random.randint(*self.calidad_rango)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=calidad)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

# ============================================================================
# REPRODUCIBILIDAD
# ============================================================================
def fijar_semilla(semilla: int) -> None:
    """Fija todas las fuentes de aleatoriedad para que el entrenamiento sea reproducible con una semilla dada."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# DATOS
# ============================================================================
def _transforms(entrenamiento: bool) -> transforms.Compose:
    """Transformaciones que se aplican a cada imagen al cargarla.
    - Sin aumento geométrico (para no alterar las huellas del generador).
    - En entrenamiento (si config.AUMENTO_ROBUSTEZ), blur gaussiano y
      recompresión JPEG de intensidad ALEATORIA: fuerzan al modelo a no
      depender de un nivel fijo de nitidez/compresión como atajo (ver
      config.py, sección AUMENTO DE ROBUSTEZ). En val/test no se altera nada.
    - Conversión a tensor y normalización de ImageNet.
    Las imágenes ya vienen a 256x256 PNG, así que no se redimensiona.
    """
    lista = []
    if entrenamiento and config.AUMENTO_ROBUSTEZ:
        lista.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=15,
                                     sigma=config.BLUR_SIGMA_RANGO)],
            p=config.PROB_BLUR))
        lista.append(RecompresionJPEGAleatoria(
            p=config.PROB_JPEG, calidad_rango=config.JPEG_CALIDAD_RANGO))

    lista += [
        transforms.ToTensor(),   # de PIL [0,255] a tensor [0,1]
        transforms.Normalize(mean=config.IMAGENET_MEDIA,
                             std=config.IMAGENET_DESV),
    ]
    return transforms.Compose(lista)

def crear_dataloader(generador: str, split: str, semilla: int,
                     barajar: bool) -> DataLoader:
    ruta = config.RUTA_PARTICIONES / generador / split
    dataset = datasets.ImageFolder(
        root=str(ruta),
        transform=_transforms(entrenamiento=(split == "train")),
    )

    # Generador de aleatoriedad propio para que el barajado sea reproducible
    g = torch.Generator()
    g.manual_seed(semilla)

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=barajar,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DISPOSITIVO.type == "cuda"),
        generator=g,
    )

def crear_dataloader_multiclase(split: str, semilla: int,
                                barajar: bool) -> DataLoader:
    """Crea un DataLoader para el experimento MULTICLASE (identificar a qué
    generador pertenece un rostro, o si es real).
    Lee Particiones/Experimentos/Multiclase/<split>/{Flux,SDXL,StyleGAN2,
    StyleGAN3,real}/. ImageFolder asigna los índices por orden alfabético
    (ver config.CLASES_MULTICLASE para el mapeo índice -> nombre).
    """
    ruta = config.RUTA_PARTICIONES_MULTICLASE / split
    dataset = datasets.ImageFolder(
        root=str(ruta),
        transform=_transforms(entrenamiento=(split == "train")),
    )

    g = torch.Generator()
    g.manual_seed(semilla)

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=barajar,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DISPOSITIVO.type == "cuda"),
        generator=g,
    )

# ============================================================================
# MODELO
# ============================================================================
def crear_modelo(num_clases: int = None) -> nn.Module:
    """Crea el modelo definido en config (EfficientNet-B0 por defecto), con
    pesos preentrenados de ImageNet. Por defecto arma la cabeza binaria
    (config.NUM_CLASES); pasa num_clases=config.NUM_CLASES_MULTICLASE para
    el experimento multiclase."""
    if num_clases is None:
        num_clases = config.NUM_CLASES
    modelo = timm.create_model(
        config.MODELO,
        pretrained=config.PREENTRENADO,
        num_classes=num_clases,
    )
    if config.CONGELAR_CAPAS:
        for nombre, parametro in modelo.named_parameters():
            if "classifier" not in nombre and "fc" not in nombre:
                parametro.requires_grad = False

    return modelo.to(config.DISPOSITIVO)

# ============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
#============================================================================
def entrenar_una_epoca(modelo, loader, optimizador, criterio, escalador):
    """Entrena el modelo una época completa.
    Devuelve (perdida_promedio, accuracy_train) para las curvas de aprendizaje.
    """
    modelo.train()
    perdida_total = 0.0
    aciertos = 0
    total = 0

    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(config.DISPOSITIVO, non_blocking=True)
        etiquetas = etiquetas.to(config.DISPOSITIVO, non_blocking=True)
        optimizador.zero_grad()
        # Precisión mixta (AMP): reduce memoria y acelera en GPU
        with torch.autocast(device_type=config.DISPOSITIVO.type,
                            enabled=config.USAR_AMP):
            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)
        # El escalador maneja los gradientes en precisión mixta
        escalador.scale(perdida).backward()
        escalador.step(optimizador)
        escalador.update()

        perdida_total += perdida.item() * imagenes.size(0)
        # Accuracy de entrenamiento de esta época
        predicciones = salidas.float().argmax(dim=1)
        aciertos += (predicciones == etiquetas).sum().item()
        total += etiquetas.size(0)

    perdida_promedio = perdida_total / len(loader.dataset)
    accuracy_train = aciertos / total
    return perdida_promedio, accuracy_train

@torch.no_grad()
def evaluar(modelo, loader, criterio=None) -> dict:
    """Evalúa el modelo y calcula todas las métricas.

    Devuelve un diccionario con accuracy, precision, recall, f1, auc, la
    matriz de confusión y (si se pasa 'criterio') la pérdida promedio.
    La clase POSITIVA es 'fake' (etiqueta 0).

    El parámetro 'criterio' (la función de pérdida) es opcional: se pasa
    durante el entrenamiento para registrar la pérdida de validación por
    época (necesaria para las curvas de aprendizaje). En la evaluación
    final de test no es imprescindible, pero se puede pasar igual.
    """
    modelo.eval()
    todas_etiquetas = []
    todas_predicciones = []
    todas_probas_fake = []   # probabilidad de la clase fake (para el AUC)
    perdida_total = 0.0

    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(config.DISPOSITIVO, non_blocking=True)
        etiquetas_gpu = etiquetas.to(config.DISPOSITIVO, non_blocking=True)

        with torch.autocast(device_type=config.DISPOSITIVO.type,
                            enabled=config.USAR_AMP):
            salidas = modelo(imagenes)
            if criterio is not None:
                perdida = criterio(salidas, etiquetas_gpu)
                perdida_total += perdida.item() * imagenes.size(0)

        probabilidades = torch.softmax(salidas.float(), dim=1)
        predicciones = probabilidades.argmax(dim=1)

        todas_etiquetas.extend(etiquetas.numpy())
        todas_predicciones.extend(predicciones.cpu().numpy())
        # Probabilidad de clase 0 (fake) para el AUC con fake como positiva
        todas_probas_fake.extend(probabilidades[:, 0].cpu().numpy())

    y_true = np.array(todas_etiquetas)
    y_pred = np.array(todas_predicciones)
    p_fake = np.array(todas_probas_fake)

    # Para tratar 'fake' (0) como la clase positiva, invertimos las etiquetas:
    # positiva = 1 cuando la imagen es fake (etiqueta original 0).
    y_true_fake = (y_true == 0).astype(int)
    y_pred_fake = (y_pred == 0).astype(int)

    metricas = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true_fake, y_pred_fake,
                                          zero_division=0),
        "recall_fake": recall_score(y_true_fake, y_pred_fake,
                                    zero_division=0),
        "f1_fake": f1_score(y_true_fake, y_pred_fake, zero_division=0),
        "auc": roc_auc_score(y_true_fake, p_fake),
        # Matriz de confusión en orden [fake, real] -> filas: real, columnas: pred
        "matriz_confusion": confusion_matrix(y_true, y_pred).tolist(),
    }
    # Pérdida promedio (solo si se pasó el criterio) para las curvas
    if criterio is not None:
        metricas["perdida"] = perdida_total / len(loader.dataset)
    return metricas


def entrenar_modelo(generador: str, semilla: int, ruta_guardado: Path) -> dict:
    """Bucle de entrenamiento completo para UN modelo con UNA semilla.
    Entrena sobre <generador>/train, valida sobre <generador>/val con early
    stopping por accuracy de validación, guarda el mejor modelo, y al final
    evalúa sobre <generador>/test.
    Devuelve un diccionario con el historial y las métricas finales de test.
    """
    fijar_semilla(semilla)

    # DataLoaders
    loader_train = crear_dataloader(generador, "train", semilla, barajar=True)
    loader_val = crear_dataloader(generador, "val", semilla, barajar=False)
    loader_test = crear_dataloader(generador, "test", semilla, barajar=False)

    # Modelo, pérdida, optimizador, escalador AMP
    modelo = crear_modelo()
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    escalador = torch.amp.GradScaler(enabled=config.USAR_AMP)

    mejor_acc_val = 0.0
    epocas_sin_mejora = 0
    historial = []

    print(f"\n  Entrenando en {generador} (semilla {semilla})")
    for epoca in range(1, config.EPOCAS + 1):
        perdida_train, acc_train = entrenar_una_epoca(
            modelo, loader_train, optimizador, criterio, escalador)
        # Pasamos el criterio para que también calcule la pérdida de validación
        metricas_val = evaluar(modelo, loader_val, criterio=criterio)
        acc_val = metricas_val["accuracy"]
        perdida_val = metricas_val["perdida"]

        # Registramos las CUATRO cantidades por época -> curvas de aprendizaje
        # completas (loss y accuracy, en train y validación).
        historial.append({
            "epoca": epoca,
            "perdida_train": perdida_train,
            "perdida_val": perdida_val,
            "acc_train": acc_train,
            "acc_val": acc_val,
        })
        print(f"    Época {epoca:2d} | "
              f"loss train: {perdida_train:.4f} | loss val: {perdida_val:.4f} "
              f"| acc train: {acc_train:.4f} | acc val: {acc_val:.4f}")

        # ¿Mejoró la validación? -> guardar mejor modelo y reiniciar paciencia
        if acc_val > mejor_acc_val:
            mejor_acc_val = acc_val
            epocas_sin_mejora = 0
            torch.save(modelo.state_dict(), ruta_guardado)
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= config.PACIENCIA_EARLY_STOPPING:
                print(f"    Early stopping en época {epoca} "
                      f"(mejor acc val: {mejor_acc_val:.4f})")
                break

    # Evaluación final: cargar el MEJOR modelo y medir en test
    modelo.load_state_dict(torch.load(ruta_guardado))
    metricas_test = evaluar(modelo, loader_test)

    print(f"    Test -> acc: {metricas_test['accuracy']:.4f} "
          f"| auc: {metricas_test['auc']:.4f} "
          f"| f1(fake): {metricas_test['f1_fake']:.4f}")

    return {
        "generador": generador,
        "semilla": semilla,
        "mejor_acc_val": mejor_acc_val,
        "historial": historial,
        "metricas_test": metricas_test,
    }

# ============================================================================
# MULTICLASE (identificar a qué generador pertenece un rostro, o si es real)
# ============================================================================
@torch.no_grad()
def evaluar_multiclase(modelo, loader, clases: list[str], criterio=None) -> dict:
    """Evalúa el modelo multiclase y calcula accuracy global, precision/
    recall/f1 macro (promedio simple entre clases, sin pesar por tamaño),
    el reporte por clase individual, y la matriz de confusión NxN (filas =
    clase real, columnas = clase predicha, en el orden de 'clases')."""
    modelo.eval()
    todas_etiquetas = []
    todas_predicciones = []
    perdida_total = 0.0

    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(config.DISPOSITIVO, non_blocking=True)
        etiquetas_gpu = etiquetas.to(config.DISPOSITIVO, non_blocking=True)

        with torch.autocast(device_type=config.DISPOSITIVO.type,
                            enabled=config.USAR_AMP):
            salidas = modelo(imagenes)
            if criterio is not None:
                perdida = criterio(salidas, etiquetas_gpu)
                perdida_total += perdida.item() * imagenes.size(0)

        predicciones = salidas.float().argmax(dim=1)
        todas_etiquetas.extend(etiquetas.numpy())
        todas_predicciones.extend(predicciones.cpu().numpy())

    y_true = np.array(todas_etiquetas)
    y_pred = np.array(todas_predicciones)

    metricas = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "reporte_por_clase": classification_report(
            y_true, y_pred, labels=list(range(len(clases))),
            target_names=clases, output_dict=True, zero_division=0),
        "matriz_confusion": confusion_matrix(
            y_true, y_pred, labels=list(range(len(clases)))).tolist(),
        "clases": clases,
    }
    if criterio is not None:
        metricas["perdida"] = perdida_total / len(loader.dataset)
    return metricas


def entrenar_modelo_multiclase(semilla: int, ruta_guardado: Path) -> dict:
    """Bucle de entrenamiento completo para el clasificador MULTICLASE
    (Flux / SDXL / StyleGAN2 / StyleGAN3 / real), con UNA semilla.
    Mismo esquema que entrenar_modelo (early stopping por accuracy de
    validación, mejor modelo guardado en disco), pero sobre las 5 clases
    de Particiones/Experimentos/Multiclase en vez de fake/real de un solo
    generador.
    """
    fijar_semilla(semilla)
    clases = config.CLASES_MULTICLASE

    loader_train = crear_dataloader_multiclase("train", semilla, barajar=True)
    loader_val = crear_dataloader_multiclase("val", semilla, barajar=False)
    loader_test = crear_dataloader_multiclase("test", semilla, barajar=False)

    modelo = crear_modelo(num_clases=config.NUM_CLASES_MULTICLASE)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    escalador = torch.amp.GradScaler(enabled=config.USAR_AMP)

    mejor_acc_val = 0.0
    epocas_sin_mejora = 0
    historial = []

    print(f"\n  Entrenando multiclase (semilla {semilla})")
    for epoca in range(1, config.EPOCAS + 1):
        perdida_train, acc_train = entrenar_una_epoca(
            modelo, loader_train, optimizador, criterio, escalador)
        metricas_val = evaluar_multiclase(modelo, loader_val, clases,
                                          criterio=criterio)
        acc_val = metricas_val["accuracy"]
        perdida_val = metricas_val["perdida"]

        historial.append({
            "epoca": epoca,
            "perdida_train": perdida_train,
            "perdida_val": perdida_val,
            "acc_train": acc_train,
            "acc_val": acc_val,
        })
        print(f"    Época {epoca:2d} | "
              f"loss train: {perdida_train:.4f} | loss val: {perdida_val:.4f} "
              f"| acc train: {acc_train:.4f} | acc val: {acc_val:.4f}")

        if acc_val > mejor_acc_val:
            mejor_acc_val = acc_val
            epocas_sin_mejora = 0
            torch.save(modelo.state_dict(), ruta_guardado)
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= config.PACIENCIA_EARLY_STOPPING:
                print(f"    Early stopping en época {epoca} "
                      f"(mejor acc val: {mejor_acc_val:.4f})")
                break

    modelo.load_state_dict(torch.load(ruta_guardado))
    metricas_test = evaluar_multiclase(modelo, loader_test, clases)

    print(f"    Test -> acc: {metricas_test['accuracy']:.4f} "
          f"| f1 macro: {metricas_test['f1_macro']:.4f}")

    return {
        "semilla": semilla,
        "clases": clases,
        "mejor_acc_val": mejor_acc_val,
        "historial": historial,
        "metricas_test": metricas_test,
    }
