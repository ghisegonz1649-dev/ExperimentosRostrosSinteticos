export type ClaseGenerador = "real" | "StyleGAN2" | "StyleGAN3" | "SDXL" | "Flux";

export const GENERADOR_INFO: Record<
  ClaseGenerador,
  { etiqueta: string; descripcion: string; color: string; familia: string }
> = {
  real: {
    etiqueta: "Real",
    descripcion: "Rostro real",
    color: "#287a5a",
    familia: "Fotografía",
  },
  StyleGAN2: {
    etiqueta: "StyleGAN2",
    descripcion: "Generada con StyleGAN2",
    color: "#155e75",
    familia: "GAN",
  },
  StyleGAN3: {
    etiqueta: "StyleGAN3",
    descripcion: "Generada con StyleGAN3",
    color: "#0f766e",
    familia: "GAN",
  },
  SDXL: {
    etiqueta: "SDXL",
    descripcion: "Generada con Stable Diffusion XL (modelo de difusión)",
    color: "#9a5b3e",
    familia: "Difusión",
  },
  Flux: {
    etiqueta: "FLUX.1",
    descripcion: "Generada con FLUX.1",
    color: "#557a65",
    familia: "Difusión",
  },
};

export const ORDEN_CLASES: ClaseGenerador[] = [
  "real",
  "StyleGAN2",
  "StyleGAN3",
  "SDXL",
  "Flux",
];

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:5000";

export type ExperimentoId = "A" | "B" | "C" | "D" | "E";

export const EXPERIMENTO_COLOR: Record<ExperimentoId, string> = {
  A: "#5a9bb5",
  B: "#F05043",
  C: "#c78b6a",
  D: "#8aa485",
  E: "#B079FC",
};

/**
 * Las cuatro métricas del análisis de resultados de la tesis, en el orden de
 * las tablas: AUC-ROC es la métrica principal por no depender de un umbral fijo.
 */
export const METRICAS_ORDEN = ["auc", "recall_fake", "especificidad", "accuracy"] as const;

export type MetricaId = (typeof METRICAS_ORDEN)[number];

export const METRICA_LABEL: Record<MetricaId, string> = {
  auc: "AUC-ROC",
  recall_fake: "Recall",
  especificidad: "Especificidad",
  accuracy: "Accuracy",
};

export const METRICA_DESCRIPCION: Record<MetricaId, string> = {
  auc: "Capacidad de separar ambas clases sin depender del umbral de decisión.",
  recall_fake: "Proporción de rostros sintéticos detectados correctamente.",
  especificidad: "Proporción de rostros reales (CelebA-HQ) reconocidos como reales.",
  accuracy: "Aciertos totales sobre el conjunto de prueba completo.",
};
