# Detección de rostros sintéticos — código de la tesis

Este repositorio contiene el código para entrenar y evaluar
clasificadores CNN (EfficientNet-B0, ResNet-50, Xception) que distinguen
rostros reales de rostros generados por StyleGAN2, StyleGAN3, SDXL y Flux, así
como el experimento multiclase (identificar a qué generador pertenece un
rostro) y la webapp de demostración.

## Estructura

```
Codigos Experimentos/        Config, motor de entrenamiento/evaluación,
                              creación de particiones y notebooks de cada
                              experimento (A, B/C/D, E, multiclase).
Codigos Figuras y Tablas/     Scripts que, a partir de los .json de métricas
                              ya generados, producen las figuras y tablas
                              finales (no reentrenan ni reevalúan nada).
JSON Experimentos/            Métricas crudas (.json) de cada experimento,
                              organizadas por experimento — el archivo de
                              donde salen las figuras y tablas.
Figuras y Tablas/             Figuras (.png) y tablas (.csv/.xlsx) finales,
                              las que se usan en el documento de la tesis.
webapp/                       Backend Flask + frontend Next.js de la página
                              de demostración (clasificación e inferencia
                              multiclase con Grad-CAM).
```



