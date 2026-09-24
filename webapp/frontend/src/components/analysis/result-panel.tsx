"use client";

import Link from "next/link";
import { motion } from "motion/react";
import { Pie, PieChart, Cell, ResponsiveContainer } from "recharts";
import {
  ShieldCheck,
  ShieldAlert,
  Flame,
  Timer,
  Cpu,
  TriangleAlert,
  ArrowRight,
  ChartPie,
} from "lucide-react";
import { ORDEN_CLASES, GENERADOR_INFO } from "@/lib/constants";
import { useRipple, RippleLayer } from "@/components/ui/ripple";
import type { Prediccion } from "@/lib/api";
import type { EstadoAnalisis } from "@/components/analysis/types";

const UMBRAL_ALTO = 0.85;
const UMBRAL_MEDIO = 0.7;

function colorPorConfianza(confianza: number) {
  if (confianza >= UMBRAL_ALTO) return "var(--success)";
  if (confianza >= UMBRAL_MEDIO) return "var(--accent)";
  return "var(--warning)";
}

function EsqueletoBarra({ delay }: { delay: number }) {
  return (
    <div className="space-y-1.5">
      <div className="h-3 w-20 animate-pulse rounded bg-muted" style={{ animationDelay: `${delay}ms` }} />
      <div className="h-2.5 w-full animate-pulse rounded-full bg-muted" style={{ animationDelay: `${delay}ms` }} />
    </div>
  );
}

function DonutDistribucion({
  resultado,
  colorConfianza,
}: {
  resultado: Prediccion;
  colorConfianza: string;
}) {
  const data = ORDEN_CLASES.map((clase) => ({
    clase,
    valor: resultado.probabilidades.find((x) => x.clase === clase)?.probabilidad ?? 0,
    color: GENERADOR_INFO[clase].color,
  })).filter((d) => d.valor > 0.001);

  const info = GENERADOR_INFO[resultado.clase_predicha];

  return (
    <div className="relative size-[150px] shrink-0">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="valor"
            nameKey="clase"
            innerRadius="68%"
            outerRadius="100%"
            paddingAngle={data.length > 1 ? 3 : 0}
            cornerRadius={6}
            stroke="none"
            startAngle={90}
            endAngle={450}
            isAnimationActive
            animationDuration={700}
          >
            {data.map((d) => (
              <Cell key={d.clase} fill={d.color} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-0.5 text-center">
        <span className="font-heading text-3xl font-semibold tabular-nums text-foreground" style={{ color: colorConfianza }}>
          {Math.round(resultado.confianza * 100)}%
        </span>
        <span className="max-w-[80px] truncate text-xs font-medium" style={{ color: info.color }}>
          {info.etiqueta}
        </span>
      </div>
    </div>
  );
}

export function ResultPanel({
  estado,
  resultado,
  errorMsg,
}: {
  estado: EstadoAnalisis;
  resultado: Prediccion | null;
  errorMsg: string | null;
}) {
  const ripple = useRipple();

  if (estado === "error") {
    return (
      <div className="flex flex-col items-center justify-center gap-4 p-8 text-center md:h-full">
        <span className="flex size-12 items-center justify-center rounded-2xl bg-destructive/10">
          <TriangleAlert className="size-6 text-destructive" strokeWidth={1.5} />
        </span>
        <p className="font-heading text-lg font-semibold text-foreground">No se pudo analizar</p>
        <p className="max-w-xs text-base text-muted-foreground">{errorMsg}</p>
      </div>
    );
  }

  if (estado === "inactivo") {
    return (
      <div className="flex flex-col gap-8 p-6 md:h-full md:p-8">
        <div>
          <h2 className="font-heading text-xl font-semibold text-foreground">Resultado</h2>
          <p className="mt-1 text-base text-muted-foreground">Aquí verás el veredicto del modelo.</p>
        </div>
        <div className="flex flex-1 flex-col items-center justify-start gap-5 pt-2 text-center">
          <div className="relative flex size-24 items-center justify-center">
            <span className="brand-soft absolute inset-3 rounded-full" />
            <ChartPie className="relative size-9 text-accent" strokeWidth={1.25} />
          </div>
          <div>
            <p className="font-heading text-xl font-semibold text-foreground">Esperando análisis</p>
            <p className="mt-2 max-w-[300px] text-lg text-muted-foreground">
              Sube la imagen de un rostro y presiona analizar para ver con qué generador fue creada la imagen.
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (estado === "analizando") {
    return (
      <div className="flex flex-col gap-8 p-6 md:h-full md:p-8">
        <div>
          <h2 className="font-heading text-xl font-semibold text-foreground">Resultado</h2>
          <p className="mt-1 text-base text-muted-foreground">Procesando la imagen…</p>
        </div>
        <div className="glass-panel h-28 animate-pulse rounded-2xl" />
        <div className="space-y-4">
          {ORDEN_CLASES.map((c, i) => (
            <EsqueletoBarra key={c} delay={i * 80} />
          ))}
        </div>
      </div>
    );
  }

  if (!resultado) return null;

  const info = GENERADOR_INFO[resultado.clase_predicha];
  const colorConfianza = colorPorConfianza(resultado.confianza);
  const confianzaBaja = resultado.confianza < UMBRAL_MEDIO;

  return (
    <div className="flex flex-col gap-7 p-6 md:h-full md:overflow-y-auto md:p-8">
      <div>
        <h2 className="font-heading text-xl font-semibold text-foreground">Resultado</h2>
        <p className="mt-1 text-base text-muted-foreground">Veredicto del modelo entrenado.</p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="glass-panel flex flex-col items-center gap-4 rounded-2xl p-6 text-center"
        style={{ boxShadow: `inset 0 1px 0 rgba(255,255,255,0.06), 0 0 40px -18px ${colorConfianza}` }}
      >
        <DonutDistribucion resultado={resultado} colorConfianza={colorConfianza} />
        <p className="text-sm font-medium uppercase tracking-wide text-muted-foreground">Confianza</p>

        <div className="mt-2 flex w-full items-center gap-3 border-t pt-4" style={{ borderColor: "var(--border)" }}>
          <span
            className="flex size-11 shrink-0 items-center justify-center rounded-xl"
            style={{ backgroundColor: `${info.color}22` }}
          >
            {resultado.es_sintetico ? (
              <ShieldAlert className="size-5" style={{ color: info.color }} strokeWidth={1.75} />
            ) : (
              <ShieldCheck className="size-5" style={{ color: info.color }} strokeWidth={1.75} />
            )}
          </span>
          <div className="text-left">
            <p className="font-heading text-2xl font-semibold text-foreground">{info.etiqueta}</p>
            <span
              className="mt-1 inline-block rounded-full border px-2 py-0.5 text-xs font-medium"
              style={{ borderColor: `${info.color}40`, color: info.color, backgroundColor: `${info.color}15` }}
            >
              {info.familia}
            </span>
          </div>
        </div>
        <p className="text-base text-muted-foreground">{resultado.descripcion_predicha}</p>

        <div className="grid w-full grid-cols-2 gap-4 border-t pt-4 text-base" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center gap-2 text-muted-foreground">
            <Timer className="size-4" strokeWidth={1.5} />
            {resultado.tiempo_inferencia_ms.toFixed(0)} ms
          </div>
          <div className="flex items-center gap-2 text-muted-foreground">
            <Cpu className="size-4" strokeWidth={1.5} />
            {resultado.arquitectura}
          </div>
        </div>
      </motion.div>

      {confianzaBaja && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="rounded-2xl border p-4"
          style={{
            borderColor: "color-mix(in oklch, var(--warning), transparent 60%)",
            backgroundColor: "color-mix(in oklch, var(--warning), transparent 90%)",
          }}
        >
          <div className="flex items-start gap-2.5">
            <TriangleAlert className="mt-0.5 size-4 shrink-0" style={{ color: "var(--warning)" }} strokeWidth={2} />
            <div className="text-base">
              <p className="font-semibold text-foreground">Confianza baja</p>
              <p className="mt-1 text-muted-foreground">El resultado es incierto. Puede ser:</p>
              <ul className="mt-1.5 space-y-1 text-muted-foreground">
                <li>• Rostro de distribución desconocida</li>
                <li>• Generador no entrenado</li>
                <li>• Imagen muy comprimida o con filtros</li>
              </ul>
              <p className="mt-2 font-medium text-foreground">Sugerencia: intenta con otra imagen.</p>
            </div>
          </div>
        </motion.div>
      )}

      <div className="space-y-2.5">
        {ORDEN_CLASES.map((clase, i) => {
          const p = resultado.probabilidades.find((x) => x.clase === clase);
          const pct = p?.porcentaje ?? 0;
          const cInfo = GENERADOR_INFO[clase];
          const esGanadora = clase === resultado.clase_predicha;
          return (
            <div
              key={clase}
              className="rounded-xl p-2.5 transition-colors"
              style={esGanadora ? { backgroundColor: `${cInfo.color}12`, border: `1px solid ${cInfo.color}30` } : undefined}
            >
              <div className="mb-1.5 flex items-center justify-between text-base">
                <span className="flex items-center gap-2 font-medium text-foreground/90">
                  <span className="size-2.5 shrink-0 rounded-full" style={{ backgroundColor: cInfo.color }} />
                  {cInfo.etiqueta}
                </span>
                <span
                  className="tabular-nums"
                  style={{ color: esGanadora ? cInfo.color : "var(--muted-foreground)", fontWeight: esGanadora ? 600 : 400 }}
                >
                  {pct.toFixed(1)}%
                </span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: cInfo.color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.7, delay: i * 0.06, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <Link
        href="/gradcam"
        prefetch={false}
        onPointerDown={ripple.onPointerDown}
        className="brand-surface focus-ring-accessible glow-border relative mt-auto flex items-center justify-center gap-2 overflow-hidden rounded-full px-6 py-3.5 text-base font-semibold text-white transition-transform active:scale-[0.98]"
      >
        <Flame className="size-4" strokeWidth={2} />
        Ver explicación (Grad-CAM)
        <ArrowRight className="size-4" strokeWidth={2} />
        <RippleLayer ripples={ripple.ripples} />
      </Link>
    </div>
  );
}
