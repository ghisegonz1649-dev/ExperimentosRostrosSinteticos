"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ScanFace } from "lucide-react";
import { FacialMesh } from "@/components/analysis/facial-mesh";
import { CircularProgress } from "@/components/analysis/circular-progress";
import type { EstadoAnalisis } from "@/components/analysis/types";

const MENSAJES = [
  "Detectando rostro",
  "Extrayendo características",
  "Analizando huella del generador",
  "Comparando firmas espectrales",
  "Clasificando imagen",
];

export function ProcessPanel({
  estado,
  previewUrl,
}: {
  estado: EstadoAnalisis;
  previewUrl: string | null;
}) {
  const [progreso, setProgreso] = useState(0);
  const [mensajeIdx, setMensajeIdx] = useState(0);

  useEffect(() => {
    if (estado !== "analizando") return;

    const reinicio = requestAnimationFrame(() => {
      setProgreso(0);
      setMensajeIdx(0);
    });
    const pasoProgreso = setInterval(() => {
      setProgreso((p) => (p >= 92 ? 92 : p + Math.random() * 6 + 2));
    }, 150);
    const pasoMensaje = setInterval(() => {
      setMensajeIdx((i) => (i + 1) % MENSAJES.length);
    }, 480);

    return () => {
      cancelAnimationFrame(reinicio);
      clearInterval(pasoProgreso);
      clearInterval(pasoMensaje);
    };
  }, [estado]);

  const activo = estado === "analizando";
  const progresoMostrado = estado === "completo" ? 100 : estado === "inactivo" ? 0 : progreso;

  return (
    <div className="flex flex-col items-center justify-center gap-8 border-border p-6 py-10 md:h-full md:overflow-y-auto md:border-r md:p-8">
      <div className="relative aspect-square w-full max-w-[340px] overflow-hidden rounded-2xl">
        <div
          className="pointer-events-none absolute inset-0 z-10 opacity-20"
          style={{
            backgroundImage:
              "linear-gradient(to right, var(--grid-line) 1px, transparent 1px), linear-gradient(to bottom, var(--grid-line) 1px, transparent 1px)",
            backgroundSize: "20px 20px",
          }}
        />

        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="Rostro en análisis"
            className={`size-full object-cover transition-[filter] duration-500 ${
              activo ? "saturate-[0.7] brightness-[0.85]" : ""
            }`}
          />
        ) : (
          <div className="flex size-full flex-col items-center justify-center gap-4 bg-muted/50 text-center">
            <div className="relative flex size-24 items-center justify-center">
              <span className="brand-soft absolute inset-3 rounded-full" />
              <ScanFace className="relative size-9 text-accent" strokeWidth={1.25} />
            </div>
            <p className="max-w-[200px] text-base text-muted-foreground">
              Sube una imagen a la izquierda para comenzar
            </p>
          </div>
        )}

        {previewUrl && <FacialMesh activo={activo} />}

        {activo && (
          <motion.div
            className="absolute inset-x-0 z-20 h-[2px]"
            style={{
              background:
                "linear-gradient(90deg, transparent, rgba(34,211,238,0.9), rgba(139,92,246,0.9), transparent)",
              boxShadow: "0 0 16px 2px rgba(34,211,238,0.6)",
            }}
            animate={{ top: ["4%", "96%", "4%"] }}
            transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
          />
        )}
      </div>

      <CircularProgress porcentaje={progresoMostrado} colorSolido="var(--accent)" />

      <div className="h-6 text-center">
        <AnimatePresence mode="wait">
          {activo ? (
            <motion.p
              key={mensajeIdx}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.25 }}
              className="text-base font-medium text-foreground/90"
            >
              {MENSAJES[mensajeIdx]}…
            </motion.p>
          ) : estado === "completo" ? (
            <p className="text-base font-medium" style={{ color: "var(--success)" }}>Análisis completo</p>
          ) : (
            <p className="text-base text-muted-foreground">En espera</p>
          )}
        </AnimatePresence>
      </div>

      <div className="h-1.5 w-full max-w-[340px] overflow-hidden rounded-full bg-muted">
        <motion.div
          className="brand-surface h-full rounded-full"
          animate={{ width: `${progresoMostrado}%` }}
          transition={{ duration: 0.2 }}
        />
      </div>
    </div>
  );
}
