"use client";

import { motion, useReducedMotion } from "motion/react";
import type { DatosArquitectura, ExperimentoMeta } from "@/lib/api";
import { EXPERIMENTO_COLOR, type ExperimentoId } from "@/lib/constants";
import { aucDe, degradacionDe, formatoMedia, formatoDesv, formatoPuntos } from "@/lib/experimentos-utils";

const ORDEN: ExperimentoId[] = ["A", "B", "C", "D", "E"];

export function ExperimentCards({
  meta,
  datos,
}: {
  meta: Record<string, ExperimentoMeta>;
  datos: DatosArquitectura;
}) {
  const reduce = useReducedMotion();

  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-5">
      {ORDEN.map((id, i) => {
        const color = EXPERIMENTO_COLOR[id];
        const auc = aucDe(datos, id);
        const degradacion = degradacionDe(datos, id);
        const info = meta[id];

        return (
          <motion.div
            key={id}
            initial={reduce ? false : { opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: i * 0.06 }}
            className="glass-panel relative overflow-hidden rounded-2xl p-5"
            style={{ boxShadow: `inset 0 1px 0 rgba(255,255,255,0.06), 0 0 32px -20px ${color}` }}
          >
            <div
              className="absolute inset-x-0 top-0 h-1 rounded-t-2xl"
              style={{ background: color }}
            />
            <div className="flex items-center gap-2">
              <span
                className="inline-flex size-8 items-center justify-center rounded-lg text-base font-bold"
                style={{ backgroundColor: `${color}22`, color }}
              >
                {id}
              </span>
              <span className="rounded-full border border-border px-2 py-0.5 text-xs font-medium text-muted-foreground">
                {info?.generador}
              </span>
            </div>
            <h3 className="mt-3 font-heading text-base font-semibold text-foreground">
              {info?.titulo?.split("—")[1]?.trim() ?? info?.titulo}
            </h3>
            <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">{info?.descripcion}</p>
            <div className="mt-4 flex flex-col gap-0.5">
              <span className="font-heading text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                AUC-ROC
              </span>
              <span className="font-heading text-2xl font-semibold tabular-nums text-foreground">
                {formatoMedia(auc)}
              </span>
              <span className="text-xs tabular-nums text-muted-foreground">{formatoDesv(auc)}</span>
            </div>
            {degradacion !== null && degradacion > 0 && (
              <p className="mt-2 text-xs text-rose-400">
                {formatoPuntos(-degradacion)} vs. línea base
              </p>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}
