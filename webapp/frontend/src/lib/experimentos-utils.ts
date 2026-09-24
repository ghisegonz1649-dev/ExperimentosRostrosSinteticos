import type { DatosArquitectura, MetricasExperimento, Stat } from "@/lib/api";
import type { ExperimentoId, MetricaId } from "@/lib/constants";

/**
 * Métricas agregadas de un experimento. Para E se toma el promedio sobre los
 * cuatro generadores tras la última fase (FT-3), que es el valor con el que la
 * tesis resume el experimento en la tabla comparativa final.
 */
export function metricasDe(datos: DatosArquitectura, id: ExperimentoId): MetricasExperimento | null {
  if (id === "A") return datos.A;
  if (id === "E") return datos.E?.final?.promedio ?? null;
  return datos[id]?.metricas ?? null;
}

export function statDe(
  datos: DatosArquitectura,
  id: ExperimentoId,
  metrica: MetricaId,
): Stat | null {
  return metricasDe(datos, id)?.[metrica] ?? null;
}

export function aucDe(datos: DatosArquitectura, id: ExperimentoId): Stat | null {
  return statDe(datos, id, "auc");
}

export function accuracyDe(datos: DatosArquitectura, id: ExperimentoId): Stat | null {
  return statDe(datos, id, "accuracy");
}

export function especificidadDe(datos: DatosArquitectura, id: ExperimentoId): Stat | null {
  return statDe(datos, id, "especificidad");
}

/**
 * Matriz de confusión sumada sobre las tres semillas. En E corresponde al
 * modelo final (FT-3); como allí hay una matriz por generador, se indica cuál.
 */
export function matrizDe(
  datos: DatosArquitectura,
  id: ExperimentoId,
  generador?: string,
): [[number, number], [number, number]] | null {
  if (id === "E") {
    const porGenerador = datos.E?.final?.por_generador;
    if (!porGenerador) return null;
    const clave = generador ?? Object.keys(porGenerador)[0];
    return porGenerador[clave]?.matriz_confusion?.matriz ?? null;
  }
  return metricasDe(datos, id)?.matriz_confusion?.matriz ?? null;
}

/** Caída en AUC-ROC respecto al Experimento A (sólo aplica a B, C y D). */
export function degradacionDe(datos: DatosArquitectura, id: ExperimentoId): number | null {
  if (id === "A" || id === "E") return null;
  return datos[id]?.degradacion_auc ?? null;
}

/** Dos decimales en todos los porcentajes, como en las tablas de la tesis. */
export function formatoPct(valor: number | null | undefined, decimales = 2): string {
  if (valor === null || valor === undefined) return "N/D";
  return `${(valor * 100).toFixed(decimales)}%`;
}

/** Media ± desviación estándar, el formato en el que la tesis reporta todo. */
export function formatoStat(stat: Stat | null | undefined, decimales = 2): string {
  if (!stat) return "N/D";
  const media = (stat.media * 100).toFixed(decimales);
  const desv = (stat.desv_est * 100).toFixed(decimales);
  return `${media}% ± ${desv}`;
}

export function formatoMedia(stat: Stat | null | undefined, decimales = 2): string {
  if (!stat) return "N/D";
  return `${(stat.media * 100).toFixed(decimales)}%`;
}

export function formatoDesv(stat: Stat | null | undefined, decimales = 2): string {
  if (!stat) return "";
  return `± ${(stat.desv_est * 100).toFixed(decimales)}`;
}

/** Diferencia en puntos porcentuales, con signo explícito. */
export function formatoPuntos(valor: number | null | undefined, decimales = 2): string {
  if (valor === null || valor === undefined) return "N/D";
  const puntos = valor * 100;
  return `${puntos >= 0 ? "+" : "−"}${Math.abs(puntos).toFixed(decimales)} pts`;
}

export function mediaOf(stat: Stat | number | null | undefined): number | null {
  if (stat === null || stat === undefined) return null;
  if (typeof stat === "number") return stat;
  return (stat as Stat).media;
}
