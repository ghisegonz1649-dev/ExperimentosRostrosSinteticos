"use client";

import type { DatosArquitectura, ExperimentoMeta } from "@/lib/api";
import {
  EXPERIMENTO_COLOR,
  METRICAS_ORDEN,
  METRICA_DESCRIPCION,
  METRICA_LABEL,
  type ExperimentoId,
} from "@/lib/constants";
import { formatoDesv, formatoMedia, metricasDe } from "@/lib/experimentos-utils";

const ORDEN: ExperimentoId[] = ["A", "B", "C", "D", "E"];

/**
 * Las cuatro métricas del análisis para los cinco experimentos, tal como se
 * reportan en la tesis: media ± desviación estándar de las tres semillas de
 * inicialización (42, 123 y 2024), con la clase sintética como clase positiva.
 */
export function MetricsTable({
  datos,
  meta,
}: {
  datos: DatosArquitectura;
  meta: Record<string, ExperimentoMeta>;
}) {
  // El mejor valor de cada métrica se resalta para leer la tabla de un vistazo.
  const mejorPorMetrica = Object.fromEntries(
    METRICAS_ORDEN.map((metrica) => {
      const valores = ORDEN.map((id) => metricasDe(datos, id)?.[metrica]?.media).filter(
        (v): v is number => v !== undefined,
      );
      return [metrica, valores.length ? Math.max(...valores) : null];
    }),
  ) as Record<(typeof METRICAS_ORDEN)[number], number | null>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[46rem] border-collapse text-base">
        <thead>
          <tr className="border-b border-border text-left text-sm text-muted-foreground">
            <th className="py-2.5 pr-4 font-medium">Experimento</th>
            <th className="py-2.5 pr-4 font-medium">Generador evaluado</th>
            {METRICAS_ORDEN.map((metrica) => (
              <th key={metrica} className="py-2.5 pr-4 text-right font-medium" title={METRICA_DESCRIPCION[metrica]}>
                {METRICA_LABEL[metrica]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ORDEN.map((id) => {
            const metricas = metricasDe(datos, id);
            const color = EXPERIMENTO_COLOR[id];
            return (
              <tr key={id} className="border-b border-border/50 last:border-0">
                <td className="py-3 pr-4">
                  <span className="flex items-center gap-2">
                    <span
                      className="inline-flex size-6 shrink-0 items-center justify-center rounded-md text-sm font-bold"
                      style={{ backgroundColor: `${color}22`, color }}
                    >
                      {id}
                    </span>
                    <span className="text-foreground">
                      {meta[id]?.titulo?.split("—")[1]?.trim() ?? id}
                    </span>
                  </span>
                </td>
                <td className="py-3 pr-4 text-muted-foreground">{meta[id]?.generador}</td>
                {METRICAS_ORDEN.map((metrica) => {
                  const stat = metricas?.[metrica];
                  const esMejor = stat != null && stat.media === mejorPorMetrica[metrica];
                  return (
                    <td key={metrica} className="py-3 pr-4 text-right">
                      <span
                        className={`block tabular-nums ${esMejor ? "font-semibold text-accent" : "text-foreground"}`}
                      >
                        {formatoMedia(stat)}
                      </span>
                      <span className="block text-xs tabular-nums text-muted-foreground">
                        {formatoDesv(stat)}
                      </span>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
        Media ± desviación estándar de las tres semillas de inicialización (42, 123 y 2024), con la clase sintética
        como clase positiva. En los experimentos B, C y D no hubo reentrenamiento, por lo que la especificidad es la
        misma del experimento A: el conjunto de rostros reales (CelebA-HQ) no cambia. El experimento E se resume como
        el promedio sobre los cuatro generadores tras la fase FT-3.
      </p>
    </div>
  );
}
