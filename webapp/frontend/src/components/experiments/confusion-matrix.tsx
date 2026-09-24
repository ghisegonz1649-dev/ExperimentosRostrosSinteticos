"use client";

import { useState } from "react";
import { motion } from "motion/react";
import { CircleCheck, CircleX } from "lucide-react";
import type { DatosArquitectura } from "@/lib/api";
import { EXPERIMENTO_COLOR, type ExperimentoId } from "@/lib/constants";
import { formatoPct, matrizDe } from "@/lib/experimentos-utils";

const OPCIONES: ExperimentoId[] = ["A", "B", "C", "D", "E"];

export function ConfusionMatrix({
  datos,
  generadores,
}: {
  datos: DatosArquitectura;
  generadores: Record<string, string>;
}) {
  const [seleccion, setSeleccion] = useState<ExperimentoId>("A");
  // Tras FT-3 el modelo del experimento E ve los cuatro generadores, así que
  // tiene una matriz por cada uno en vez de una sola.
  const clavesE = Object.keys(datos.E?.final?.por_generador ?? {});
  const [generador, setGenerador] = useState<string>(clavesE[0] ?? "");
  const esE = seleccion === "E";
  const matriz = matrizDe(datos, seleccion, esE ? generador : undefined);

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-2">
        {OPCIONES.map((id) => (
          <button
            key={id}
            onClick={() => setSeleccion(id)}
            className="rounded-full border px-3.5 py-1.5 text-sm font-medium transition-colors"
            style={
              seleccion === id
                ? {
                    borderColor: `${EXPERIMENTO_COLOR[id]}80`,
                    backgroundColor: `${EXPERIMENTO_COLOR[id]}18`,
                    color: EXPERIMENTO_COLOR[id],
                  }
                : { borderColor: "var(--border)", color: "var(--muted-foreground)" }
            }
          >
            Experimento {id}
          </button>
        ))}
      </div>

      {esE && clavesE.length > 0 && (
        <div className="mb-5 flex flex-wrap items-center gap-2">
          <span className="text-sm text-muted-foreground">Modelo final (FT-3) evaluado en:</span>
          {clavesE.map((clave) => (
            <button
              key={clave}
              onClick={() => setGenerador(clave)}
              className={`rounded-full border px-3 py-1 text-sm font-medium transition-colors ${
                generador === clave
                  ? "brand-active text-accent"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {generadores[clave] ?? clave}
            </button>
          ))}
        </div>
      )}

      {!matriz ? (
        <p className="text-base text-muted-foreground">
          No hay datos de matriz de confusión para este experimento.
        </p>
      ) : (
        <>
          <MatrizGrid matriz={matriz} />
          <Resumen matriz={matriz} />
        </>
      )}
    </div>
  );
}

function MatrizGrid({ matriz }: { matriz: [[number, number], [number, number]] }) {
  const [[tp, fn], [fp, tn]] = matriz;
  const filaSintetico = tp + fn;
  const filaReal = fp + tn;

  const celdas = [
    { valor: tp, total: filaSintetico, correcto: true, etiqueta: "VP" },
    { valor: fn, total: filaSintetico, correcto: false, etiqueta: "FN" },
    { valor: fp, total: filaReal, correcto: false, etiqueta: "FP" },
    { valor: tn, total: filaReal, correcto: true, etiqueta: "VN" },
  ];

  return (
    <div className="grid grid-cols-[auto_1fr_1fr] gap-3 text-center">
      <div />
      <p className="self-end pb-1 text-sm font-medium uppercase tracking-wide text-muted-foreground">Predicho: Sintético</p>
      <p className="self-end pb-1 text-sm font-medium uppercase tracking-wide text-muted-foreground">Predicho: Real</p>

      <p className="flex items-center justify-end pr-2 text-sm font-medium uppercase tracking-wide text-muted-foreground">Real: Sintético</p>
      {celdas.slice(0, 2).map((c) => (
        <Celda key={c.etiqueta} {...c} />
      ))}

      <p className="flex items-center justify-end pr-2 text-sm font-medium uppercase tracking-wide text-muted-foreground">Real: Auténtico</p>
      {celdas.slice(2, 4).map((c) => (
        <Celda key={c.etiqueta} {...c} />
      ))}
    </div>
  );
}

/** Recall y especificidad se leen directamente de las filas de la matriz. */
function Resumen({ matriz }: { matriz: [[number, number], [number, number]] }) {
  const [[tp, fn], [fp, tn]] = matriz;
  const total = tp + fn + fp + tn;
  const recall = tp + fn > 0 ? tp / (tp + fn) : null;
  const especificidad = fp + tn > 0 ? tn / (fp + tn) : null;

  return (
    <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
      {[
        { etiqueta: "Recall (fila sintética)", valor: formatoPct(recall) },
        { etiqueta: "Especificidad (fila real)", valor: formatoPct(especificidad) },
        { etiqueta: "Imágenes evaluadas", valor: total.toLocaleString("es-ES") },
      ].map(({ etiqueta, valor }) => (
        <div key={etiqueta} className="rounded-xl border border-border px-4 py-3">
          <p className="text-sm text-muted-foreground">{etiqueta}</p>
          <p className="mt-0.5 font-heading text-xl font-semibold tabular-nums text-foreground">{valor}</p>
        </div>
      ))}
    </div>
  );
}

function Celda({
  valor,
  total,
  correcto,
  etiqueta,
}: {
  valor: number;
  total: number;
  correcto: boolean;
  etiqueta: string;
}) {
  const pct = total > 0 ? valor / total : 0;
  const color = correcto ? "#22c55e" : "#f43f5e";
  const Icono = correcto ? CircleCheck : CircleX;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col items-center gap-2 rounded-xl border p-4"
      style={{
        borderColor: `${color}30`,
        backgroundColor: `${color}${Math.round(pct * 35 + 8).toString(16).padStart(2, "0")}`,
      }}
    >
      <Icono className="size-4" style={{ color }} strokeWidth={1.75} />
      <p className="font-heading text-3xl font-semibold tabular-nums text-foreground">{valor.toLocaleString("es-ES")}</p>
      <p className="text-sm font-medium" style={{ color }}>
        {etiqueta} · {(pct * 100).toFixed(2)}%
      </p>
    </motion.div>
  );
}
