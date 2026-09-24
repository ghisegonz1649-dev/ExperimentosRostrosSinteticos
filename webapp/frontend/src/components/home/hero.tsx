"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion, useReducedMotion } from "motion/react";
import { ArrowRight, FlaskConical, ChevronDown } from "lucide-react";
import { useRipple, RippleLayer } from "@/components/ui/ripple";

/**
 * Palabra del título con gradiente en movimiento. Combina dos animaciones en
 * el mismo elemento vía transiciones por-propiedad: la entrada escalonada
 * (opacity/y con `delay`) y el flujo continuo del gradiente (backgroundPosition).
 */
function PalabraGradiente({
  children,
  from,
  via,
  to,
  delay,
  reduce,
}: {
  children: string;
  from: string;
  via: string;
  to: string;
  delay: number;
  reduce: boolean;
}) {
  return (
    <motion.span
      className="inline-block bg-clip-text text-transparent"
      style={{
        backgroundImage: `linear-gradient(90deg, ${from}, ${via}, ${to}, ${from})`,
        backgroundSize: "300% 100%",
      }}
      initial={reduce ? false : { opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0, backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"] }}
      transition={{
        opacity: { duration: 0.45, delay },
        y: { duration: 0.45, delay, ease: [0.16, 1, 0.3, 1] },
        backgroundPosition: { duration: 4, repeat: Infinity, ease: "linear" },
      }}
    >
      {children}
    </motion.span>
  );
}

/** Palabra de color plano con la misma entrada escalonada. */
function PalabraSimple({ children, delay, reduce }: { children: string; delay: number; reduce: boolean }) {
  return (
    <motion.span
      className="inline-block"
      initial={reduce ? false : { opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.span>
  );
}

/** Blob animado decorativo. */
function Blob({
  tamaño,
  posicion,
  delay,
  duracion,
  reduce,
}: {
  tamaño: string;
  posicion: string;
  delay: number;
  duracion: number;
  reduce: boolean;
}) {
  return (
    <motion.div
      className={`pointer-events-none absolute rounded-full blur-3xl opacity-30 ${tamaño} ${posicion}`}
      style={{
        background: "linear-gradient(135deg, #0f766e, #155e75, #0fa094)",
      }}
      animate={
        reduce
          ? {}
          : {
              scale: [1, 1.1, 0.95, 1],
              x: [0, 20, -20, 0],
              y: [0, -30, 30, 0],
            }
      }
      transition={{
        duration: duracion,
        repeat: Infinity,
        ease: "easeInOut",
        delay,
      }}
    />
  );
}

export function Hero() {
  const reduce = useReducedMotion();
  const [scrolleado, setScrolleado] = useState(false);
  const ripple = useRipple();

  useEffect(() => {
    const onScroll = () => setScrolleado(window.scrollY > 24);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    // La altura mínima se queda bastante por debajo del viewport: con el
    // contenido centrado, reservar la pantalla completa dejaba un hueco muy
    // grande entre los botones y la franja de cifras.
    <section className="relative flex min-h-[56vh] items-center overflow-hidden px-6 pb-12 pt-40 md:min-h-[60vh] md:pb-16 md:px-12 lg:px-16 xl:pt-44">
      {/* Blobs animados de fondo — tres capas con diferentes tamaños y ritmos. */}
      <Blob tamaño="w-96 h-96" posicion="top-0 -left-1/3" delay={0} duracion={8} reduce={!!reduce} />
      <Blob tamaño="w-80 h-80" posicion="top-1/4 right-0" delay={0.5} duracion={10} reduce={!!reduce} />
      <Blob tamaño="w-72 h-72" posicion="bottom-0 left-1/4" delay={1} duracion={12} reduce={!!reduce} />

      {/* Línea decorativa superior que entra con el contenido. */}
      <motion.div
        className="pointer-events-none absolute top-12 left-1/2 h-px w-24 -translate-x-1/2 bg-gradient-to-r from-transparent via-accent to-transparent"
        initial={reduce ? false : { opacity: 0, scaleX: 0 }}
        animate={{ opacity: 1, scaleX: 1 }}
        transition={{ duration: 0.8, delay: 0.3 }}
      />

      <div className="relative z-10 mx-auto w-full max-w-4xl text-center">
        <h1 className="font-heading text-[1.75rem] font-bold leading-[1.35] tracking-normal text-foreground sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl">
          <PalabraSimple delay={0} reduce={!!reduce}>
            Detección y Atribución
          </PalabraSimple>{" "}
          <PalabraGradiente from="#0f766e" via="#155e75" to="#0f766e" delay={0.2} reduce={!!reduce}>
            de Rostros Generados
          </PalabraGradiente>{" "}
          <PalabraGradiente from="#155e75" via="#0f766e" to="#155e75" delay={0.4} reduce={!!reduce}>
            con IA
          </PalabraGradiente>
        </h1>

        {/* Glow sutil detrás del título. */}
        <motion.div
          className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 h-72 bg-gradient-to-b from-accent/20 to-transparent blur-3xl"
          initial={reduce ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.5 }}
        />

        <motion.p
          initial={reduce ? false : { opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="mx-auto mt-6 max-w-4xl text-pretty text-xl leading-relaxed text-foreground md:text-2xl"
        >
          Detecta rostros sintéticos, identifica su generador y explica por qué.
        </motion.p>

        <motion.div
          initial={reduce ? false : { opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="mt-9 flex flex-wrap items-center justify-center gap-4"
        >
          {/* Botón primario con más énfasis. */}
          <motion.div
            whileHover={reduce ? {} : { scale: 1.05 }}
            whileTap={reduce ? {} : { scale: 0.98 }}
          >
            <Link
              href="/analisis"
              onPointerDown={ripple.onPointerDown}
              className="brand-surface focus-ring-accessible glow-border group relative flex items-center gap-2 overflow-hidden rounded-full px-6 py-3 text-sm font-semibold text-white transition-all hover:-translate-y-1 active:scale-[0.98] shadow-lg hover:shadow-xl"
            >
              Iniciar análisis
              <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" strokeWidth={2} />
              <RippleLayer ripples={ripple.ripples} />
            </Link>
          </motion.div>

          {/* Botón secundario con animación sutil. */}
          <motion.div
            initial={reduce ? false : { opacity: 0.8 }}
            whileHover={reduce ? {} : { scale: 1.02, opacity: 1 }}
            whileTap={reduce ? {} : { scale: 0.98 }}
          >
            <Link
              href="/experimentos"
              className="focus-ring-accessible flex items-center gap-2 rounded-full border border-border bg-transparent px-6 py-3 text-sm font-medium text-foreground/90 transition-colors hover:bg-muted active:scale-[0.98]"
            >
              <FlaskConical className="size-4 text-accent" strokeWidth={1.75} />
              Explorar experimentos
            </Link>
          </motion.div>
        </motion.div>
      </div>

      {/* Chevron animado que desaparece al scrollear. */}
      <motion.div
        animate={{ opacity: scrolleado ? 0 : [0.5, 1, 0.5], y: [0, 4, 0] }}
        transition={{
          opacity: { duration: 2, repeat: scrolleado ? 0 : Infinity, ease: "easeInOut" },
          y: { duration: 1.5, repeat: scrolleado ? 0 : Infinity, ease: "easeInOut" },
        }}
        className="pointer-events-none absolute inset-x-0 bottom-6 z-10 flex flex-col items-center gap-1"
      >
        <ChevronDown className="size-5 text-muted-foreground" strokeWidth={1.5} />
      </motion.div>
    </section>
  );
}
