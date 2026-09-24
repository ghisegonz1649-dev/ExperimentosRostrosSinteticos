"use client";

import { useState } from "react";
import { Check, Minus } from "lucide-react";
import type { ExperimentoE } from "@/lib/api";
import { METRICAS_ORDEN, METRICA_LABEL } from "@/lib/constants";
import { formatoDesv, formatoMedia } from "@/lib/experimentos-utils";

/**
 * Detalle del ajuste fino progresivo: cada etapa se evalúa contra los cuatro
 * generadores, marcando cuáles ya formaban parte del entrenamiento en ese
 * momento. La evaluación inicial corresponde al modelo del Experimento A.
 */
export function FineTuningTable({
  experimentoE,
  generadores,
}: {
  experimentoE: ExperimentoE;
  generadores: Record<string, string>;
}) {
  const etapas = [
    {
      id: "Inicial",
      titulo: "Evaluación inicial",
      entrenado: "StyleGAN2",
      datos: experimentoE.inicial,
    },
    ...experimentoE.fases.map((f) => ({
      id: f.fase,
      titulo: f.fase,
      entrenado: f.generadores_entrenamiento.map((g) => generadores[g] ?? g).join(" + "),
      datos: f,
    })),
  ];

  const [activa, setActiva] = useState(etapas[etapas.length - 1]?.id ?? "Inicial");
  const etapa = etapas.find((e) => e.id === activa) ?? etapas[0];

  if (!etapa) return null;

  const claves = Object.keys(etapa.datos.por_generador);

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-2">
        {etapas.map((e) => (
          <button
            key={e.id}
            onClick={() => setActiva(e.id)}
            className={`rounded-full border px-3.5 py-1.5 text-sm font-medium transition-colors ${
              activa === e.id
                ? "brand-active text-accent"
                : "border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            {e.titulo}
          </button>
        ))}
      </div>

      <p className="mb-4 text-sm text-muted-foreground">
        Generadores en el entrenamiento en esta etapa:{" "}
        <span className="font-medium text-foreground">{etapa.entrenado}</span>
      </p>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[42rem] border-collapse text-base">
          <thead>
            <tr className="border-b border-border text-left text-sm text-muted-foreground">
              <th className="py-2.5 pr-4 font-medium">Generador evaluado</th>
              <th className="py-2.5 pr-4 font-medium">Visto</th>
              {METRICAS_ORDEN.map((metrica) => (
                <th key={metrica} className="py-2.5 pr-4 text-right font-medium">
                  {METRICA_LABEL[metrica]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {claves.map((clave) => {
              const fila = etapa.datos.por_generador[clave];
              return (
                <tr key={clave} className="border-b border-border/50 last:border-0">
                  <td className="py-3 pr-4 text-foreground">{generadores[clave] ?? clave}</td>
                  <td className="py-3 pr-4">
                    {fila.visto ? (
                      <span className="inline-flex items-center gap-1 text-sm text-emerald-500">
                        <Check className="size-3.5" strokeWidth={2.25} /> Sí
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-sm text-muted-foreground">
                        <Minus className="size-3.5" strokeWidth={2.25} /> No
                      </span>
                    )}
                  </td>
                  {METRICAS_ORDEN.map((metrica) => (
                    <td key={metrica} className="py-3 pr-4 text-right">
                      <span className="block tabular-nums text-foreground">{formatoMedia(fila[metrica])}</span>
                      <span className="block text-xs tabular-nums text-muted-foreground">
                        {formatoDesv(fila[metrica])}
                      </span>
                    </td>
                  ))}
                </tr>
              );
            })}
            <tr className="border-t border-border">
              <td className="py-3 pr-4 font-medium text-foreground">Promedio</td>
              <td className="py-3 pr-4" />
              {METRICAS_ORDEN.map((metrica) => (
                <td key={metrica} className="py-3 pr-4 text-right">
                  <span className="block font-semibold tabular-nums text-accent">
                    {formatoMedia(etapa.datos.promedio[metrica])}
                  </span>
                  <span className="block text-xs tabular-nums text-muted-foreground">
                    {formatoDesv(etapa.datos.promedio[metrica])}
                  </span>
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
        Un generador &ldquo;visto&rdquo; ya estaba en el entrenamiento de esa etapa; los demás se evalúan sin haber
        sido incorporados. La especificidad es la misma para los cuatro generadores dentro de cada etapa, porque el
        conjunto de rostros reales no cambia: sólo varía el conjunto sintético evaluado.
      </p>
    </div>
  );
}
