"use client";

import { Line, LineChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import type { ExperimentoE } from "@/lib/api";
import { mediaOf } from "@/lib/experimentos-utils";

const SERIES = [
  { clave: "auc", nombre: "AUC-ROC", color: "var(--chart-series-a)" },
  { clave: "recall_fake", nombre: "Recall (sintéticos)", color: "var(--chart-sintetico)" },
  { clave: "especificidad", nombre: "Especificidad (reales)", color: "var(--chart-real)" },
  { clave: "accuracy", nombre: "Accuracy", color: "var(--chart-series-e)" },
] as const;

/**
 * Evolución de las cuatro métricas a lo largo del ajuste fino progresivo,
 * promediadas sobre los cuatro generadores evaluados en cada etapa.
 */
export function PerformanceLineChart({ experimentoE }: { experimentoE: ExperimentoE }) {
  const etapas = [
    { etapa: "Inicial", metricas: experimentoE.inicial.promedio },
    ...experimentoE.fases.map((f) => ({ etapa: f.fase, metricas: f.promedio })),
  ];

  const datos = etapas.map(({ etapa, metricas }) => ({
    etapa,
    ...Object.fromEntries(SERIES.map(({ clave }) => [clave, mediaOf(metricas?.[clave])])),
  }));

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={datos} margin={{ top: 16, right: 16, bottom: 0, left: -12 }}>
          <CartesianGrid stroke="var(--grid-line)" vertical={false} />
          <XAxis dataKey="etapa" stroke="var(--muted-foreground)" tick={{ fill: "var(--muted-foreground)", fontSize: 14 }} />
          <YAxis
            stroke="var(--muted-foreground)"
            tick={{ fill: "var(--muted-foreground)", fontSize: 14 }}
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            domain={[0, 1]}
          />
          <Tooltip
            contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 12 }}
            labelStyle={{ color: "var(--popover-foreground)" }}
            itemStyle={{ color: "var(--popover-foreground)" }}
            formatter={(value) => (value == null ? "N/D" : `${(Number(value) * 100).toFixed(2)}%`)}
          />
          <Legend wrapperStyle={{ fontSize: 14, color: "var(--muted-foreground)" }} />
          {SERIES.map(({ clave, nombre, color }) => (
            <Line
              key={clave}
              type="monotone"
              dataKey={clave}
              name={nombre}
              stroke={color}
              strokeWidth={2.5}
              dot={{ r: 4, fill: color, stroke: "var(--background)", strokeWidth: 2 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
