"use client";

import { useEffect } from "react";
import { motion, useReducedMotion } from "motion/react";
import { Target, Gauge, ShieldQuestion, Info } from "lucide-react";
import { CompareSlider } from "@/components/gradcam/compare-slider";
import { IntensityLegend } from "@/components/gradcam/intensity-legend";
import { generarExplicacion } from "@/components/gradcam/explicacion";
import { GENERADOR_INFO } from "@/lib/constants";
import { reiniciarGradCam, type GradCamResultado } from "@/lib/api";

export function GradCamView({ resultado }: { resultado: GradCamResultado }) {
  const reduce = useReducedMotion();

  // Al salir de la página, limpiar el estado del servidor para que la próxima
  // visita arranque desde 0. El flag `armado` evita disparar el reset en el
  // desmontaje-fantasma de React Strict Mode (solo en dev).
  useEffect(() => {
    let armado = false;
    const id = setTimeout(() => {
      armado = true;
    }, 0);
    return () => {
      clearTimeout(id);
      if (armado) reiniciarGradCam();
    };
  }, []);

  const info = GENERADOR_INFO[resultado.clase_predicha];
  const esSintetico = resultado.clase_predicha !== "real";
  const explicacion = generarExplicacion(resultado.regiones, info.etiqueta, esSintetico);

  return (
    <div className="mx-auto max-w-6xl px-6 pb-24 pt-24 md:px-12 md:pt-16 lg:px-16">
      <motion.div
        initial={reduce ? false : { opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="font-heading text-3xl font-semibold text-foreground md:text-4xl">
            Explicabilidad Grad-CAM
          </h1>
          <span
            className="rounded-full border px-3 py-1 text-sm font-medium"
            style={{ borderColor: `${info.color}40`, color: info.color, backgroundColor: `${info.color}15` }}
          >
            {info.etiqueta}
          </span>
        </div>
        <p className="mt-2 max-w-2xl text-base text-muted-foreground md:text-lg">
          {esSintetico
            ? `Por qué el modelo atribuyó este rostro a ${info.etiqueta}.`
            : "Por qué el modelo consideró este rostro auténtico."}
        </p>
      </motion.div>

      <motion.div
        initial={reduce ? false : { opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.05 }}
        className="glass-panel mt-8 flex gap-4 rounded-2xl p-5 md:p-6"
      >
        <span className="brand-soft flex size-11 shrink-0 items-center justify-center rounded-xl">
          <Info className="size-5.5 text-accent" strokeWidth={1.6} />
        </span>
        <div>
          <h2 className="font-heading text-xl font-semibold text-foreground">¿Qué es Grad-CAM?</h2>
          <p className="mt-2 text-lg leading-relaxed text-foreground">
            Grad-CAM (<em>Gradient-weighted Class Activation Mapping</em>) es una técnica de explicabilidad que revela
            en qué zonas de la imagen se enfocó la red neuronal para tomar su decisión. Analiza los gradientes que
            llegan a la última capa convolucional y construye un mapa de calor: las zonas en tonos cálidos (rosa o
            magenta) fueron las más influyentes, mientras que las zonas en tonos fríos (azul) apenas influyeron. Así
            es posible verificar si el modelo se basó en rasgos faciales relevantes o en artefactos irrelevantes de la
            imagen.
          </p>
        </div>
      </motion.div>

      <div className="mt-10 grid grid-cols-1 gap-6 md:grid-cols-3">
        {[
          { src: resultado.original, label: "Imagen original" },
          { src: resultado.heatmap, label: "Mapa de activación" },
          { src: resultado.overlay, label: "Superposición" },
        ].map((panel, i) => (
          <motion.div
            key={panel.label}
            initial={reduce ? false : { opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: i * 0.08 }}
            className="glass-panel overflow-hidden rounded-2xl"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={panel.src} alt={panel.label} className="aspect-square w-full object-cover" />
            <p className="p-3 text-center text-base font-medium text-foreground">{panel.label}</p>
          </motion.div>
        ))}
      </div>

      <div className="mt-12 grid grid-cols-1 gap-10 md:grid-cols-5">
        <div className="md:col-span-3">
          <h2 className="font-heading text-xl font-semibold text-foreground">Comparar original y superposición</h2>
          <p className="mt-1 text-base text-muted-foreground">Arrastrá el control para revelar el mapa de calor.</p>
          <div className="mt-5 glass-panel rounded-2xl p-3">
            <CompareSlider
              before={resultado.original}
              after={resultado.overlay}
              beforeLabel="Original"
              afterLabel="Superposición"
            />
          </div>
          <div className="mt-4">
            <IntensityLegend />
          </div>
        </div>

        <div className="flex flex-col gap-6 md:col-span-2">
          <div className="glass-panel rounded-2xl p-6">
            <h3 className="font-heading text-lg font-semibold text-foreground">Interpretación</h3>
            <p className="mt-2 text-base leading-relaxed text-muted-foreground">{explicacion}</p>
          </div>

          <div className="glass-panel flex-1 rounded-2xl p-6">
            <h3 className="mb-4 font-heading text-lg font-semibold text-foreground">Resumen de la explicación</h3>

            <div className="mb-4">
              <div className="mb-2 flex items-center gap-2 text-base text-muted-foreground">
                <Target className="size-4 text-accent" strokeWidth={1.6} />
                Regiones más relevantes
              </div>
              <div className="space-y-2">
                {resultado.regiones.slice(0, 3).map((r) => (
                  <div key={r.region} className="flex items-center justify-between text-base">
                    <span className="text-foreground/90">{r.region}</span>
                    <span className="tabular-nums text-muted-foreground">{(r.intensidad * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 border-t border-border pt-4">
              <div>
                <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
                  <Gauge className="size-3.5" strokeWidth={1.6} />
                  Intensidad promedio
                </div>
                <p className="mt-1 font-heading text-2xl font-semibold text-foreground">
                  {(resultado.intensidad_promedio * 100).toFixed(0)}%
                </p>
              </div>
              <div>
                <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
                  <ShieldQuestion className="size-3.5" strokeWidth={1.6} />
                  Confianza de la explicación
                </div>
                <p className="mt-1 font-heading text-2xl font-semibold text-foreground">
                  {(resultado.confianza_explicacion * 100).toFixed(0)}%
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
