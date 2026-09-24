"use client";

import { useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { TrendingUp, Grid3x3, ShieldCheck, Table2, Layers, type LucideIcon } from "lucide-react";
import { ExperimentCards } from "@/components/experiments/experiment-cards";
import { MetricsTable } from "@/components/experiments/metrics-table";
import { FineTuningTable } from "@/components/experiments/fine-tuning-table";
import { PerformanceLineChart } from "@/components/experiments/performance-line-chart";
import { ConfusionMatrix } from "@/components/experiments/confusion-matrix";
import { RealFaceRobustness } from "@/components/experiments/real-face-robustness";
import { InterpretationCard } from "@/components/experiments/interpretation-card";
import { ArchitectureComparison } from "@/components/experiments/architecture-comparison";
import { Breadcrumb } from "@/components/layout/breadcrumb";
import type { ExperimentosResponse } from "@/lib/api";

function Seccion({
  titulo,
  descripcion,
  icono: Icono,
  children,
  delay = 0,
}: {
  titulo: string;
  descripcion?: string;
  icono: LucideIcon;
  children: React.ReactNode;
  delay?: number;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.section
      initial={reduce ? false : { opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ duration: 0.5, delay }}
      className="glass-panel rounded-2xl p-6 md:p-8"
    >
      <div className="flex items-center gap-3">
        <span className="brand-soft flex size-11 shrink-0 items-center justify-center rounded-xl">
          <Icono className="size-5.5 text-accent" strokeWidth={1.75} />
        </span>
        <div>
          <h2 className="font-heading text-xl font-semibold text-foreground md:text-2xl">{titulo}</h2>
          {descripcion && <p className="mt-0.5 text-base text-muted-foreground">{descripcion}</p>}
        </div>
      </div>
      <div className="mt-6">{children}</div>
    </motion.section>
  );
}

export function ExperimentsDashboard({ data }: { data: ExperimentosResponse }) {
  const arquitecturas = Object.keys(data.arquitecturas);
  const [arquitectura, setArquitectura] = useState(data.arquitectura_predeterminada);
  const datos = data.datos[arquitectura];

  if (!datos) {
    return <p className="p-8 text-base text-muted-foreground">No hay datos de experimentos disponibles.</p>;
  }

  return (
    <div>
      <Breadcrumb items={[{ label: "Inicio", href: "/" }, { label: "Experimentos" }]} />
      <div className="mx-auto max-w-7xl space-y-8 px-6 pb-24 pt-6 md:px-12 lg:px-16">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="font-heading text-3xl font-semibold text-foreground md:text-4xl">Panel de experimentos</h1>
          <p className="mt-1 text-base text-muted-foreground">
            Resultados de los experimentos A–E de la tesis: media ± desviación estándar de tres semillas.
          </p>
        </div>
        <div className="flex gap-2">
          {arquitecturas.map((arch) => (
            <button
              key={arch}
              onClick={() => setArquitectura(arch)}
              className={`rounded-full border px-3.5 py-1.5 text-sm font-medium transition-colors ${
                arquitectura === arch
                  ? "brand-active text-accent"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {data.arquitecturas[arch]}
            </button>
          ))}
        </div>
      </div>

      <ExperimentCards meta={data.experimentos} datos={datos} />

      <Seccion
        titulo="Métricas por experimento"
        descripcion="AUC-ROC, Recall, Especificidad y Accuracy en los cinco experimentos, con la clase sintética como clase positiva."
        icono={Table2}
        delay={0.05}
      >
        <MetricsTable datos={datos} meta={data.experimentos} />
      </Seccion>

      <Seccion
        titulo="Matrices de confusión"
        descripcion="Conteos reales por experimento, sumados sobre las tres semillas de inicialización."
        icono={Grid3x3}
        delay={0.1}
      >
        <ConfusionMatrix datos={datos} generadores={data.generadores} />
      </Seccion>

      <Seccion
        titulo="Robustez en rostros reales"
        descripcion="Especificidad (aciertos sobre rostros reales) frente a recall (detección de sintéticos), al cambiar el generador evaluado."
        icono={ShieldCheck}
        delay={0.2}
      >
        <RealFaceRobustness datos={datos} />
      </Seccion>

      {datos.E && (
        <>
          <Seccion
            titulo="Evolución del ajuste fino progresivo (Experimento E)"
            descripcion="Promedio sobre los cuatro generadores en cada etapa, desde el modelo base hasta FT-3."
            icono={TrendingUp}
            delay={0.4}
          >
            <PerformanceLineChart experimentoE={datos.E} />
          </Seccion>

          <Seccion
            titulo="Detalle por generador y fase (Experimento E)"
            descripcion="Cada etapa evaluada contra los cuatro generadores, indicando cuáles ya estaban en el entrenamiento."
            icono={Layers}
            delay={0.5}
          >
            <FineTuningTable experimentoE={datos.E} generadores={data.generadores} />
          </Seccion>
        </>
      )}

      <InterpretationCard
        datos={datos}
        etiquetas={data.arquitecturas}
        arquitecturaActiva={arquitectura}
      />

      <ArchitectureComparison todas={data.datos} etiquetas={data.arquitecturas} />
      </div>
    </div>
  );
}
