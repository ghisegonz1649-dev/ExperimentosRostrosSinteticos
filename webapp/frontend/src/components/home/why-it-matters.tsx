"use client";

import { motion, useReducedMotion } from "motion/react";
import { UserX, Newspaper, ShieldHalf, BadgeCheck, Microscope } from "lucide-react";

const TARJETAS = [
  {
    titulo: "Deepfakes",
    descripcion: "Rostros sintéticos cada vez más realistas circulan sin ninguna marca de origen visible.",
    icono: UserX,
  },
  {
    titulo: "Desinformación",
    descripcion: "Contenido generado por IA se usa para suplantar identidades y difundir noticias falsas.",
    icono: Newspaper,
  },
  {
    titulo: "Seguridad Digital",
    descripcion: "Sistemas de verificación biométrica quedan expuestos si no pueden distinguir rostros reales de rostros sinteticos.",
    icono: ShieldHalf,
  },
  {
    titulo: "Verificación de Contenido",
    descripcion: "Periodistas e investigadores necesitan herramientas para confirmar la autenticidad de una imagen.",
    icono: BadgeCheck,
  },
  {
    titulo: "Investigación Científica",
    descripcion: "Entender las huellas que dejan los generadores ayuda a construir defensas más robustas.",
    icono: Microscope,
  },
] as const;

export function WhyItMatters() {
  const reduce = useReducedMotion();

  return (
    <section className="px-6 pb-12 pt-8 md:px-12 md:pb-16 md:pt-10 lg:px-16">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="font-heading text-3xl font-semibold text-foreground md:text-4xl">
            ¿Por qué es importante?
          </h2>
          <p className="mt-3 max-w-xl text-base text-muted-foreground md:text-lg">
            La detección de rostros sintéticos ha dejado de ser un simple reto técnico de la visión por computadora para convertirse en una línea de defensa crítica para la seguridad, la privacidad y la confianza en la sociedad digital.
          </p>
        </motion.div>

        <div className="mt-10 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-5">
          {TARJETAS.map((t, i) => (
            <motion.div
              key={t.titulo}
              initial={reduce ? false : { opacity: 0, y: 24, scale: 0.96, filter: "blur(8px)" }}
              whileInView={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ duration: 0.6, delay: i * 0.07, ease: [0.16, 1, 0.3, 1] }}
              className="glass-panel group rounded-3xl border border-transparent p-6 transition-all duration-300 hover:-translate-y-1 hover:border-accent/40"
            >
              <span className="brand-soft flex size-16 items-center justify-center rounded-2xl">
                <t.icono className="size-7 text-accent" strokeWidth={1.5} />
              </span>
              <h3 className="mt-5 font-heading text-lg font-semibold text-foreground">{t.titulo}</h3>
              <p className="mt-2 text-base leading-relaxed text-muted-foreground">{t.descripcion}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
