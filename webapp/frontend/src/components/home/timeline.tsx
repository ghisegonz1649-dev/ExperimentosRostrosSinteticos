"use client";

import { motion, useReducedMotion } from "motion/react";
import { Upload, Grid3x3, Fingerprint, Crosshair, Eye } from "lucide-react";

const PASOS = [
  { numero: 1, titulo: "Carga del rostro", descripcion: "Subir una imagen del rostro a analizar.", icono: Upload },
  { numero: 2, titulo: "Extracción de características", descripcion: "La CNN extrae patrones visuales de bajo y alto nivel.", icono: Grid3x3 },
  { numero: 3, titulo: "Detección de huella", descripcion: "Se busca la firma característica de cada generador.", icono: Fingerprint },
  { numero: 4, titulo: "Atribución del generador", descripcion: "El modelo clasifica entre rostro real y 4 familias de generadores.", icono: Crosshair },
  { numero: 5, titulo: "Explicabilidad (Grad-CAM)", descripcion: "Se visualizan las regiones que motivaron la decisión.", icono: Eye },
] as const;

export function Timeline() {
  const reduce = useReducedMotion();

  return (
    <section className="px-6 pb-8 pt-12 md:px-12 md:pb-10 md:pt-16 lg:px-16">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="font-heading text-3xl font-semibold text-foreground md:text-4xl">
            Cómo funciona el análisis
          </h2>
          <p className="mt-3 max-w-xl text-base text-muted-foreground md:text-lg">
            Cinco pasos que componen la explicación multiclase visual de la detección.
          </p>
        </motion.div>

        <div className="relative mt-14 flex snap-x gap-8 overflow-x-auto pb-4 md:grid md:snap-none md:grid-cols-5 md:gap-6 md:overflow-visible md:pb-0">
          <div
            aria-hidden
            className="absolute left-0 right-0 top-9 hidden h-px md:block"
            style={{ background: "linear-gradient(90deg, var(--border), var(--accent-2), var(--border))" }}
          />
          {PASOS.map((paso, i) => (
            <motion.div
              key={paso.numero}
              initial={reduce ? false : { opacity: 0, y: 24, scale: 0.96, filter: "blur(8px)" }}
              whileInView={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
              viewport={{ once: true, amount: 0.4 }}
              transition={{ duration: 0.6, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] }}
              className="group relative min-w-[220px] shrink-0 snap-start md:min-w-0"
            >
              <div className="relative z-10 flex size-14 items-center justify-center rounded-2xl border border-border bg-background transition-all duration-300 group-hover:border-accent group-hover:shadow-[0_0_24px_-8px_var(--accent)]">
                <motion.span
                  animate={{ opacity: [0.7, 1, 0.7] }}
                  transition={{ duration: 2.4, repeat: Infinity, delay: i * 0.15, ease: "easeInOut" }}
                >
                  <paso.icono className="size-6 text-accent" strokeWidth={1.5} />
                </motion.span>
              </div>
              <span className="mt-4 block font-heading text-4xl font-bold text-accent">
                {paso.numero}
              </span>
              <h3 className="mt-2 font-heading text-lg font-semibold text-foreground">{paso.titulo}</h3>
              <p className="mt-1.5 text-base leading-relaxed text-muted-foreground">{paso.descripcion}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
