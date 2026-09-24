"use client";

import { motion, useReducedMotion } from "motion/react";
import type { DatosArquitectura, Stat } from "@/lib/api";
import { GENERADOR_INFO } from "@/lib/constants";
import { aucDe, especificidadDe, formatoPct, metricasDe } from "@/lib/experimentos-utils";

const MAPEO = [
  { experimento: "A" as const, clase: "StyleGAN2" as const },
  { experimento: "B" as const, clase: "StyleGAN3" as const },
  { experimento: "C" as const, clase: "SDXL" as const },
  { experimento: "D" as const, clase: "Flux" as const },
];

const COLOR_REAL = "var(--chart-real)";
const COLOR_SINTETICO = "var(--chart-sintetico)";
const TICKS = [1, 0.75, 0.5, 0.25, 0];

interface FilaRobustez {
  generador: string;
  familia: string;
  especificidad: number;
  recall: Stat;
  auc: Stat | null;
}

/** Centro de la columna i dentro de n columnas, en porcentaje del ancho útil. */
const centroX = (i: number, n: number) => ((i + 0.5) / n) * 100;
/** Un valor 0-1 a coordenada vertical (0% = arriba). */
const posY = (valor: number) => (1 - valor) * 100;

function Leyenda() {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
      {[
        { color: COLOR_REAL, texto: "Rostros reales — especificidad" },
        { color: COLOR_SINTETICO, texto: "Rostros sintéticos — recall" },
      ].map(({ color, texto }) => (
        <div key={texto} className="flex items-center gap-2">
          <span className="size-2.5 rounded-full" style={{ backgroundColor: color }} />
          <span className="text-sm text-muted-foreground">{texto}</span>
        </div>
      ))}
    </div>
  );
}

function Punto({
  valor,
  color,
  etiquetado,
  ladoEtiqueta,
  delay,
  reduce,
}: {
  valor: number;
  color: string;
  etiquetado: boolean;
  /** En la última columna la etiqueta va a la izquierda para no salirse del área. */
  ladoEtiqueta: "izquierda" | "derecha";
  delay: number;
  reduce: boolean;
}) {
  return (
    <motion.div
      className="absolute left-1/2 -translate-x-1/2 -translate-y-1/2"
      style={{ top: `${posY(valor)}%` }}
      initial={reduce ? false : { opacity: 0, scale: 0.4 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true, amount: 0.4 }}
      transition={{ duration: 0.4, delay }}
    >
      {/* Anillo del color de la superficie: mantiene el punto legible al solaparse. */}
      <span
        className="block size-2.5 rounded-full ring-2 ring-[var(--card)]"
        style={{ backgroundColor: color }}
      />
      {etiquetado && (
        <span
          className={`absolute top-1/2 -translate-y-1/2 whitespace-nowrap text-xs font-semibold tabular-nums text-foreground ${
            ladoEtiqueta === "derecha" ? "left-1/2 ml-3" : "right-1/2 mr-3"
          }`}
        >
          {formatoPct(valor)}
        </span>
      )}
    </motion.div>
  );
}

export function RealFaceRobustness({ datos }: { datos: DatosArquitectura }) {
  const reduce = useReducedMotion();

  const puntos: FilaRobustez[] = MAPEO.flatMap(({ experimento, clase }) => {
    const especificidad = especificidadDe(datos, experimento)?.media ?? null;
    const recall = metricasDe(datos, experimento)?.recall_fake;
    if (especificidad === null || !recall) return [];
    return [
      {
        generador: GENERADOR_INFO[clase].etiqueta,
        familia: GENERADOR_INFO[clase].familia,
        especificidad,
        recall,
        auc: aucDe(datos, experimento),
      },
    ];
  });

  if (puntos.length < 2) {
    return <p className="text-base text-muted-foreground">No hay matrices de confusión para esta arquitectura.</p>;
  }

  const n = puntos.length;
  const especificidades = puntos.map((p) => p.especificidad);
  const espMin = Math.min(...especificidades);
  const espMax = Math.max(...especificidades);
  // El conjunto de rostros reales es el mismo en A-D, así que la especificidad
  // se mantiene fija; sólo mostramos un rango si por alguna razón no lo fuera.
  const espEstable = espMax - espMin < 0.005;

  const recallPrimero = puntos[0].recall.media;
  const recallUltimo = puntos[n - 1].recall.media;

  const linea = (clave: "especificidad" | "recall") =>
    puntos.map((p, i) => {
      const valor = clave === "especificidad" ? p.especificidad : p.recall.media;
      return `${centroX(i, n)},${posY(valor)}`;
    }).join(" ");

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-xl border border-border px-4 py-3">
          <p className="text-sm text-muted-foreground">Especificidad en rostros reales</p>
          <p className="mt-0.5 font-heading text-3xl font-semibold" style={{ color: COLOR_REAL }}>
            {espEstable ? formatoPct(espMin) : `${formatoPct(espMin)} – ${formatoPct(espMax)}`}
          </p>
          <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
            {espEstable
              ? "Idéntica en los cuatro experimentos: cambiar el generador no afecta cómo el modelo trata a los rostros reales."
              : "Varía poco al cambiar el generador evaluado."}
          </p>
        </div>
        <div className="rounded-xl border border-border px-4 py-3">
          <p className="text-sm text-muted-foreground">Recall en rostros sintéticos</p>
          <p className="mt-0.5 font-heading text-3xl font-semibold" style={{ color: COLOR_SINTETICO }}>
            {formatoPct(recallPrimero)} → {formatoPct(recallUltimo)}
          </p>
          <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
            De {puntos[0].generador} a {puntos[n - 1].generador}: aquí es donde se concentra toda la pérdida de
            robustez.
          </p>
        </div>
      </div>

      <Leyenda />

      <div className="flex gap-3">
        <div className="relative h-64 w-9 shrink-0">
          {TICKS.map((t) => (
            <span
              key={t}
              className="absolute right-0 -translate-y-1/2 text-xs tabular-nums text-muted-foreground"
              style={{ top: `${posY(t)}%` }}
            >
              {Math.round(t * 100)}%
            </span>
          ))}
        </div>

        <div className="flex-1">
          <div className="relative h-64">
            {TICKS.map((t) => (
              <span
                key={t}
                className="absolute inset-x-0 border-t border-border/60"
                style={{ top: `${posY(t)}%` }}
              />
            ))}

            {/* Tendencia de cada serie. `preserveAspectRatio=none` estira la caja,
                por eso el trazo se fija con vector-effect. */}
            <svg
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              className="pointer-events-none absolute inset-0 size-full overflow-visible"
            >
              {(
                [
                  ["especificidad", COLOR_REAL],
                  ["recall", COLOR_SINTETICO],
                ] as const
              ).map(([clave, color]) => (
                <polyline
                  key={clave}
                  points={linea(clave)}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  vectorEffect="non-scaling-stroke"
                  opacity={0.5}
                />
              ))}
            </svg>

            {puntos.map((p, i) => {
              const recallMedia = p.recall.media;
              const arriba = Math.max(p.especificidad, recallMedia);
              const abajo = Math.min(p.especificidad, recallMedia);
              const brecha = arriba - abajo;

              return (
                <div
                  key={p.generador}
                  tabIndex={0}
                  className="group focus-ring-accessible absolute inset-y-0"
                  style={{ left: `${(i / n) * 100}%`, width: `${100 / n}%` }}
                >
                  {/* Conector: su largo ES la brecha entre ambas tasas. */}
                  <motion.span
                    className="absolute left-1/2 w-px -translate-x-1/2 bg-border"
                    style={{ top: `${posY(arriba)}%`, height: `${brecha * 100}%` }}
                    initial={reduce ? false : { opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true, amount: 0.4 }}
                    transition={{ duration: 0.4, delay: i * 0.08 }}
                  />

                  <Punto
                    valor={p.especificidad}
                    color={COLOR_REAL}
                    etiquetado={i === 0 || i === n - 1}
                    ladoEtiqueta={i === n - 1 ? "izquierda" : "derecha"}
                    delay={i * 0.08}
                    reduce={!!reduce}
                  />
                  <Punto
                    valor={p.recall.media}
                    color={COLOR_SINTETICO}
                    etiquetado={i === 0 || i === n - 1}
                    ladoEtiqueta={i === n - 1 ? "izquierda" : "derecha"}
                    delay={i * 0.08 + 0.04}
                    reduce={!!reduce}
                  />

                  {/* Anclado al borde en las columnas extremas: centrado se saldría del panel. */}
                  <span
                    className={`pointer-events-none absolute top-2 z-20 w-max max-w-[12rem] rounded-lg border border-border bg-popover px-3 py-2 text-sm text-popover-foreground opacity-0 shadow-lg transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 ${
                      i === 0 ? "left-0" : i === n - 1 ? "right-0" : "left-1/2 -translate-x-1/2"
                    }`}
                  >
                    <span className="block font-medium">{p.generador}</span>
                    <span className="mt-1 block text-muted-foreground">
                      Reales <span className="tabular-nums text-popover-foreground">{formatoPct(p.especificidad)}</span>
                      {" · "}
                      Sintéticos{" "}
                      <span className="tabular-nums text-popover-foreground">{formatoPct(p.recall.media)}</span>
                    </span>
                    <span className="mt-0.5 block text-muted-foreground">
                      Brecha <span className="tabular-nums text-popover-foreground">{formatoPct(brecha)}</span>
                    </span>
                  </span>
                </div>
              );
            })}
          </div>

          <div className="mt-2 flex">
            {puntos.map((p, i) => (
              <div key={p.generador} className="text-center" style={{ width: `${100 / n}%` }}>
                <p className="text-sm font-medium text-foreground">{p.generador}</p>
                <p className="text-xs text-muted-foreground">{MAPEO[i].experimento} · {p.familia}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Vista en tabla: ningún valor queda sólo detrás del cursor. */}
      <div className="overflow-x-auto">
        <table className="w-full min-w-[26rem] text-base">
          <thead>
            <tr className="border-b border-border text-left text-sm text-muted-foreground">
              <th className="py-2 pr-3 font-medium">Generador evaluado</th>
              <th className="py-2 pr-3 text-right font-medium">Reales</th>
              <th className="py-2 pr-3 text-right font-medium">Sintéticos</th>
              <th className="py-2 text-right font-medium">AUC-ROC</th>
            </tr>
          </thead>
          <tbody>
            {puntos.map((p) => (
              <tr key={p.generador} className="border-b border-border/50 last:border-0">
                <td className="py-2 pr-3 text-foreground">{p.generador}</td>
                <td className="py-2 pr-3 text-right tabular-nums text-foreground">{formatoPct(p.especificidad)}</td>
                <td className="py-2 pr-3 text-right tabular-nums text-foreground">{formatoPct(p.recall.media)}</td>
                <td className="py-2 text-right tabular-nums text-muted-foreground">{formatoPct(p.auc?.media ?? null)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
