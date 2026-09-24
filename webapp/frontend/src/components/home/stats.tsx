"use client";

import { motion, useReducedMotion } from "motion/react";
import { CountUp } from "@/components/home/count-up";
import type { Stats } from "@/lib/api";

const ITEMS = (stats: Stats) => [
  { value: stats.generadores, label: "Generadores utilizados" },
  { value: stats.arquitecturas_cnn, label: "Arquitecturas CNN" },
  { value: stats.imagenes_dataset, label: "Imágenes Utilizadas" },
  { value: stats.experimentos, label: "Experimentos" },
];

export function StatsStrip({ stats }: { stats: Stats }) {
  const reduce = useReducedMotion();

  return (
    <section className="border-y border-border bg-secondary-surface px-6 py-10 md:px-12 lg:px-16">
      <div className="mx-auto grid max-w-6xl grid-cols-2 gap-8 md:grid-cols-4">
        {ITEMS(stats).map((item, i) => (
          <motion.div
            key={item.label}
            initial={reduce ? false : { opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 0.5, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] }}
            className="text-center md:text-left"
          >
            <p className="font-heading text-4xl font-semibold tabular-nums text-foreground md:text-4xl">
              <CountUp value={item.value} />
            </p>
            <p className="mt-1.5 text-sm text-muted-foreground md:text-base">{item.label}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
