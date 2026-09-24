import { API_BASE, type ClaseGenerador } from "@/lib/constants";

export interface Probabilidad {
  clase: ClaseGenerador;
  probabilidad: number;
  porcentaje: number;
  descripcion: string;
}

export interface Prediccion {
  clase_predicha: ClaseGenerador;
  descripcion_predicha: string;
  confianza: number;
  es_sintetico: boolean;
  probabilidades: Probabilidad[];
  modelo_usado: string;
  arquitectura: string;
  tiempo_inferencia_ms: number;
  imagen_data_url: string;
}

export interface Stats {
  generadores: number;
  arquitecturas_cnn: number;
  imagenes_dataset: number;
  experimentos: number;
}

export class ApiError extends Error {}

export async function obtenerStats(): Promise<Stats> {
  const res = await fetch(`${API_BASE}/api/stats`);
  if (!res.ok) throw new ApiError("No se pudieron cargar las estadísticas.");
  return res.json();
}

export async function analizarImagen(archivo: File): Promise<Prediccion> {
  const formData = new FormData();
  formData.append("imagen", archivo);

  const res = await fetch(`${API_BASE}/api/analizar`, {
    method: "POST",
    body: formData,
  });

  const datos = await res.json();
  if (!res.ok) {
    throw new ApiError(datos.error ?? "No se pudo analizar la imagen.");
  }
  return datos as Prediccion;
}

export interface Region {
  region: string;
  intensidad: number;
}

export interface GradCamResultado {
  original: string;
  overlay: string;
  heatmap: string;
  clase_predicha: ClaseGenerador;
  descripcion_predicha: string;
  regiones: Region[];
  intensidad_promedio: number;
  confianza_explicacion: number;
}

export async function obtenerGradCam(): Promise<GradCamResultado> {
  const res = await fetch(`${API_BASE}/api/gradcam`, { cache: "no-store" });
  const datos = await res.json();
  if (!res.ok) {
    throw new ApiError(datos.error ?? "No se pudo generar la explicación.");
  }
  return datos as GradCamResultado;
}

// Limpia el Grad-CAM guardado en el servidor para que, al volver a la página,
// arranque desde 0. Silencioso: si falla, no es crítico.
export async function reiniciarGradCam(): Promise<void> {
  try {
    await fetch(`${API_BASE}/api/gradcam`, { method: "DELETE", keepalive: true });
  } catch {
    // sin acción: el reset es best-effort
  }
}

export interface Stat {
  media: number;
  desv_est: number;
}

/**
 * Las cuatro métricas reportadas en el capítulo de resultados de la tesis.
 * Precisión y F1 quedaron fuera del análisis, así que el backend ya no las envía.
 */
export interface MetricasExperimento {
  auc?: Stat;
  recall_fake?: Stat;
  especificidad?: Stat;
  accuracy?: Stat;
  matriz_confusion?: { matriz: [[number, number], [number, number]] };
  n_semillas?: number;
}

export interface ExperimentoBCD {
  metricas: MetricasExperimento;
  /** Caída en AUC-ROC respecto al Experimento A, en proporción (0-1). */
  degradacion_auc: number | null;
}

export interface MetricasGenerador extends MetricasExperimento {
  /** Si ese generador ya formaba parte del entrenamiento en esa etapa. */
  visto: boolean;
}

export interface EtapaE {
  por_generador: Record<string, MetricasGenerador>;
  /** Promedio sobre los cuatro generadores evaluados. */
  promedio: MetricasExperimento;
}

export interface FaseE extends EtapaE {
  fase: string;
  generadores_entrenamiento: string[];
  mejor_acc_val: Stat | null;
}

export interface ExperimentoE {
  inicial: EtapaE;
  fases: FaseE[];
  final: FaseE | null;
}

export interface DatosArquitectura {
  A: MetricasExperimento | null;
  B: ExperimentoBCD | null;
  C: ExperimentoBCD | null;
  D: ExperimentoBCD | null;
  E: ExperimentoE | null;
}

export interface ExperimentoMeta {
  titulo: string;
  generador: string;
  descripcion: string;
}

export interface ExperimentosResponse {
  arquitecturas: Record<string, string>;
  metricas: Record<string, string>;
  generadores: Record<string, string>;
  experimentos: Record<string, ExperimentoMeta>;
  datos: Record<string, DatosArquitectura>;
  arquitectura_predeterminada: string;
}

export async function obtenerExperimentos(): Promise<ExperimentosResponse> {
  const res = await fetch(`${API_BASE}/api/experimentos`, { cache: "no-store" });
  if (!res.ok) throw new ApiError("No se pudieron cargar los experimentos.");
  return res.json();
}
