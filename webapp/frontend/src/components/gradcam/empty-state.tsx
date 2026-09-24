"use client";

import Link from "next/link";
import { motion, useReducedMotion } from "motion/react";
import { ArrowRight, ImageIcon, Layers, ScanFace, Thermometer } from "lucide-react";

const PANELES = [
  { icon: ImageIcon, label: "Original" },
  { icon: Thermometer, label: "Mapa de calor" },
  { icon: Layers, label: "Superposición" },
] as const;

export function GradCamEmptyState() {
  const reduce = useReducedMotion();

  return (
    <div className="mx-auto flex min-h-[calc(100dvh-156px)] max-w-4xl flex-col items-center justify-center px-6 py-20 text-center">
      <motion.div
        initial={reduce ? false : { opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative flex size-32 items-center justify-center"
      >
        <motion.span
          className="absolute inset-0 rounded-full border"
          style={{ borderColor: "var(--accent)" }}
          animate={reduce ? undefined : { opacity: [0.5, 0.1, 0.5], scale: [1, 1.15, 1] }}
          transition={{ duration: 2.6, repeat: Infinity, ease: "easeInOut" }}
        />
        <span className="brand-soft absolute inset-3 rounded-full" />
        {/* Marcas de esquina, misma estética de visor que el panel de análisis. */}
        {(
          [
            "-top-1 -left-1 rounded-tl-lg border-t-2 border-l-2",
            "-top-1 -right-1 rounded-tr-lg border-t-2 border-r-2",
            "-bottom-1 -left-1 rounded-bl-lg border-b-2 border-l-2",
            "-bottom-1 -right-1 rounded-br-lg border-b-2 border-r-2",
          ] as const
        ).map((cls) => (
          <span key={cls} className={`absolute size-5 ${cls}`} style={{ borderColor: "var(--accent)" }} />
        ))}
        <ScanFace className="relative size-12 text-accent" strokeWidth={1.25} />
      </motion.div>

      <motion.h1
        initial={reduce ? false : { opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="mt-8 font-heading text-3xl font-semibold text-foreground md:text-4xl"
      >
        Todavía no hay nada que explicar
      </motion.h1>
      <motion.p
        initial={reduce ? false : { opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.15 }}
        className="mt-3 max-w-xl text-base leading-relaxed text-muted-foreground md:text-lg"
      >
        Analiza un rostro primero para ver el mapa de calor Grad-CAM y entender por qué el modelo tomó esa decisión.
      </motion.p>

      <motion.div
        initial={reduce ? false : { opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.25 }}
        className="mt-12 grid w-full grid-cols-3 gap-4 sm:gap-6"
      >
        {PANELES.map(({ icon: Icon, label }, i) => (
          <div
            key={label}
            className="glass-panel flex aspect-square flex-col items-center justify-center gap-3 overflow-hidden rounded-2xl border-dashed"
          >
            <motion.span
              animate={reduce ? undefined : { opacity: [0.3, 0.6, 0.3] }}
              transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut", delay: i * 0.3 }}
            >
              <Icon className="size-9 text-muted-foreground/70 md:size-11" strokeWidth={1.4} />
            </motion.span>
            <p className="px-2 text-center text-sm font-medium text-foreground md:text-base">{label}</p>
          </div>
        ))}
      </motion.div>

      <motion.div
        initial={reduce ? false : { opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.35 }}
      >
        <Link
          href="/analisis"
          className="brand-surface focus-ring-accessible glow-border mt-12 flex items-center gap-2 rounded-full px-8 py-4 text-base font-semibold text-white transition-transform active:scale-[0.98]"
        >
          Ir a análisis
          <ArrowRight className="size-5" strokeWidth={2} />
        </Link>
      </motion.div>
    </div>
  );
}
