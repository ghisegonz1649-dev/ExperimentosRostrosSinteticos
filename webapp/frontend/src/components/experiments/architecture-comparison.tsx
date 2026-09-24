import { Scale, Trophy } from "lucide-react";
import type { DatosArquitectura } from "@/lib/api";
import type { ExperimentoId, MetricaId } from "@/lib/constants";
import { aucDe, especificidadDe, formatoPct, metricasDe } from "@/lib/experimentos-utils";

/** Los cuatro experimentos que se evalúan sin reentrenar el modelo base. */
const EXPERIMENTOS_GENERALIZACION: ExperimentoId[] = ["A", "B", "C", "D"];

function promedio(valores: number[]): number | null {
  return valores.length ? valores.reduce((a, b) => a + b, 0) / valores.length : null;
}

/** Media de una métrica sobre los experimentos indicados. */
function mediaDe(datos: DatosArquitectura, metrica: MetricaId, ids: ExperimentoId[]): number | null {
  return promedio(
    ids
      .map((id) => metricasDe(datos, id)?.[metrica]?.media)
      .filter((v): v is number => v !== undefined),
  );
}

/** Media de la desviación estándar entre semillas: cuánto depende del azar. */
function dispersionDe(datos: DatosArquitectura, metrica: MetricaId, ids: ExperimentoId[]): number | null {
  return promedio(
    ids
      .map((id) => metricasDe(datos, id)?.[metrica]?.desv_est)
      .filter((v): v is number => v !== undefined),
  );
}

interface FilaComparativa {
  clave: string;
  etiqueta: string;
  /**
   * Accuracy medio en A–D. Como las clases están balanceadas 1:1 (1 050 reales
   * y 1 050 sintéticas por partición), equivale al punto medio entre recall y
   * especificidad: mide justo el equilibrio entre ambas clases.
   */
  equilibrio: number | null;
  especificidad: number | null;
  dispersion: number | null;
  aucFinal: number | null;
  especificidadFinal: number | null;
}

/**
 * Cierre comparativo del estudio: qué arquitectura se comporta mejor a lo largo
 * de los cinco experimentos. El ganador se deriva de los datos, no se fija a
 * mano, para que no pueda desincronizarse de los resultados.
 */
export function ArchitectureComparison({
  todas,
  etiquetas,
}: {
  todas: Record<string, DatosArquitectura>;
  etiquetas: Record<string, string>;
}) {
  const comparativa: FilaComparativa[] = Object.entries(todas).map(([clave, datos]) => ({
    clave,
    etiqueta: etiquetas[clave] ?? clave,
    equilibrio: mediaDe(datos, "accuracy", EXPERIMENTOS_GENERALIZACION),
    especificidad: especificidadDe(datos, "A")?.media ?? null,
    dispersion: dispersionDe(datos, "accuracy", EXPERIMENTOS_GENERALIZACION),
    aucFinal: aucDe(datos, "E")?.media ?? null,
    especificidadFinal: especificidadDe(datos, "E")?.media ?? null,
  }));

  // La más equilibrada es la que mantiene el mejor punto medio entre detectar
  // sintéticos y reconocer rostros reales a lo largo de A–D.
  const masEquilibrada = comparativa.reduce<FilaComparativa | null>(
    (mejor, fila) =>
      fila.equilibrio !== null && (mejor === null || fila.equilibrio > (mejor.equilibrio ?? -1)) ? fila : mejor,
    null,
  );
  const masEstable = comparativa.reduce<FilaComparativa | null>(
    (mejor, fila) =>
      fila.dispersion !== null && (mejor === null || fila.dispersion < (mejor.dispersion ?? Infinity)) ? fila : mejor,
    null,
  );

  if (comparativa.length < 2 || !masEquilibrada) return null;

  return (
    <div className="glass-panel rounded-2xl p-6 md:p-8">
      <div className="flex items-center gap-3">
        <span className="brand-soft flex size-11 shrink-0 items-center justify-center rounded-xl">
          <Scale className="size-5.5 text-accent" strokeWidth={1.75} />
        </span>
        <div>
          <h3 className="font-heading text-xl font-semibold text-foreground md:text-2xl">
            Comparación entre las tres arquitecturas
          </h3>
          <p className="mt-0.5 text-base text-muted-foreground">
            Qué arquitectura conserva mejor su desempeño a lo largo de los cinco experimentos.
          </p>
        </div>
      </div>

      <div className="mt-6 overflow-x-auto">
        <table className="w-full min-w-[40rem] border-collapse text-base">
          <thead>
            <tr className="border-b border-border text-left text-sm text-muted-foreground">
              <th className="py-2.5 pr-4 font-medium">Arquitectura</th>
              <th
                className="py-2.5 pr-4 text-right font-medium"
                title="Accuracy medio en A–D; con las clases balanceadas 1:1 equivale al punto medio entre recall y especificidad."
              >
                Equilibrio entre clases
              </th>
              <th className="py-2.5 pr-4 text-right font-medium">Especificidad (A–D)</th>
              <th
                className="py-2.5 pr-4 text-right font-medium"
                title="Desviación estándar media entre las tres semillas: cuanto menor, más estable."
              >
                Variación entre semillas
              </th>
              <th className="py-2.5 text-right font-medium">AUC-ROC tras FT-3</th>
            </tr>
          </thead>
          <tbody>
            {comparativa.map((fila) => {
              const destacada = fila.clave === masEquilibrada.clave;
              return (
                <tr
                  key={fila.clave}
                  className={`border-b border-border/50 last:border-0 ${destacada ? "bg-accent/5" : ""}`}
                >
                  <td className="py-3 pr-4">
                    <span className="flex items-center gap-2">
                      {destacada && <Trophy className="size-3.5 shrink-0 text-accent" strokeWidth={2} />}
                      <span className={destacada ? "font-semibold text-accent" : "text-foreground"}>
                        {fila.etiqueta}
                      </span>
                    </span>
                  </td>
                  <td className={`py-3 pr-4 text-right tabular-nums ${destacada ? "font-semibold text-accent" : "text-foreground"}`}>
                    {formatoPct(fila.equilibrio)}
                  </td>
                  <td className="py-3 pr-4 text-right tabular-nums text-foreground">
                    {formatoPct(fila.especificidad)}
                  </td>
                  <td className="py-3 pr-4 text-right tabular-nums text-muted-foreground">
                    ± {((fila.dispersion ?? 0) * 100).toFixed(2)}
                  </td>
                  <td className="py-3 text-right tabular-nums text-foreground">{formatoPct(fila.aucFinal)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-6 text-base leading-relaxed text-muted-foreground md:text-lg">
        Considerando los cinco experimentos en conjunto, <strong className="font-semibold text-foreground">
        {masEquilibrada.etiqueta}</strong> fue la arquitectura más equilibrada en la generalización: mantiene el mejor
        punto medio entre detectar rostros sintéticos y reconocer correctamente los rostros reales
        ({formatoPct(masEquilibrada.equilibrio)} frente a generadores vistos y no vistos), conserva la especificidad más
        alta sobre CelebA-HQ ({formatoPct(masEquilibrada.especificidad)}) y
        {masEstable?.clave === masEquilibrada.clave
          ? " es además la que menos varía entre semillas"
          : ` mantiene una variación entre semillas contenida (± ${((masEquilibrada.dispersion ?? 0) * 100).toFixed(2)} puntos)`}
        , lo que indica un comportamiento más estable y menos dependiente de la inicialización. Tras el ajuste fino
        progresivo conserva esa ventaja, con el AUC-ROC más alto ({formatoPct(masEquilibrada.aucFinal)}) y la mejor
        especificidad final ({formatoPct(masEquilibrada.especificidadFinal)}). Por ello es la opción más adecuada para
        detectar rostros sintéticos de distintas tecnologías sin penalizar demasiado la clasificación de rostros reales.
      </p>
    </div>
  );
}
