import { Lightbulb, ArrowDownRight, ArrowUpRight } from "lucide-react";
import type { DatosArquitectura } from "@/lib/api";
import { aucDe, formatoMedia, metricasDe } from "@/lib/experimentos-utils";

const ROJO = "#f43f5e";
const VERDE = "#22c55e";

/** Cifras clave de la arquitectura seleccionada en el panel. */
export function InterpretationCard({
  datos,
  etiquetas,
  arquitecturaActiva,
}: {
  datos: DatosArquitectura;
  etiquetas: Record<string, string>;
  arquitecturaActiva: string;
}) {
  const aucA = aucDe(datos, "A");
  const aucC = aucDe(datos, "C");
  const aucE = aucDe(datos, "E");
  const recallE = metricasDe(datos, "E")?.recall_fake ?? null;

  return (
    <div className="glass-panel rounded-2xl p-6 md:p-8">
      <div className="flex items-center gap-3">
        <span className="brand-soft flex size-11 shrink-0 items-center justify-center rounded-xl">
          <Lightbulb className="size-5.5 text-accent" strokeWidth={1.75} />
        </span>
        <div>
          <h3 className="font-heading text-xl font-semibold text-foreground md:text-2xl">
            Interpretación de resultados
          </h3>
          <p className="mt-0.5 text-base text-muted-foreground">
            {etiquetas[arquitecturaActiva] ?? arquitecturaActiva}
          </p>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-xl border border-border p-3 text-center">
          <p className="font-heading text-base font-semibold tabular-nums text-foreground">{formatoMedia(aucA)}</p>
          <p className="mt-0.5 text-xs text-muted-foreground">AUC-ROC línea base (StyleGAN2)</p>
        </div>
        <div className="rounded-xl border p-3 text-center" style={{ borderColor: `${ROJO}30`, backgroundColor: `${ROJO}0d` }}>
          <p className="flex items-center justify-center gap-1 font-heading text-base font-semibold tabular-nums" style={{ color: ROJO }}>
            <ArrowDownRight className="size-4" strokeWidth={2} />
            {formatoMedia(aucC)}
          </p>
          <p className="mt-0.5 text-xs text-muted-foreground">Peor caso: SDXL (difusión)</p>
        </div>
        <div className="rounded-xl border p-3 text-center" style={{ borderColor: `${VERDE}30`, backgroundColor: `${VERDE}0d` }}>
          <p className="flex items-center justify-center gap-1 font-heading text-base font-semibold tabular-nums" style={{ color: VERDE }}>
            <ArrowUpRight className="size-4" strokeWidth={2} />
            {formatoMedia(aucE)}
          </p>
          <p className="mt-0.5 text-xs text-muted-foreground">Tras el ajuste fino (FT-3)</p>
        </div>
        <div className="rounded-xl border border-border p-3 text-center">
          <p className="font-heading text-base font-semibold tabular-nums text-foreground">{formatoMedia(recallE)}</p>
          <p className="mt-0.5 text-xs text-muted-foreground">Recall final sobre sintéticos</p>
        </div>
      </div>
    </div>
  );
}
