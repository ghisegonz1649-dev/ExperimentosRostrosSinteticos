"use client";

import { useCallback, useState, type PointerEvent } from "react";
import { AnimatePresence, motion } from "motion/react";

interface RippleInstancia {
  id: number;
  x: number;
  y: number;
  size: number;
}

let contador = 0;

/**
 * Hook de microinteracción "ripple": el consumidor pone `position: relative;
 * overflow: hidden` en el elemento interactivo, agrega `onPointerDown={onPointerDown}`
 * y renderiza `<RippleLayer ripples={ripples} />` como hijo.
 */
export function useRipple() {
  const [ripples, setRipples] = useState<RippleInstancia[]>([]);

  const onPointerDown = useCallback((e: PointerEvent<HTMLElement>) => {
    const target = e.currentTarget;
    const rect = target.getBoundingClientRect();
    const size = Math.min(160, Math.max(rect.width, rect.height) * 1.4);
    const nueva: RippleInstancia = {
      id: contador++,
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      size,
    };
    setRipples((prev) => [...prev, nueva]);
    window.setTimeout(() => {
      setRipples((prev) => prev.filter((r) => r.id !== nueva.id));
    }, 400);
  }, []);

  return { ripples, onPointerDown };
}

export function RippleLayer({ ripples }: { ripples: { id: number; x: number; y: number; size: number }[] }) {
  return (
    <span className="pointer-events-none absolute inset-0 overflow-hidden rounded-[inherit]" aria-hidden>
      <AnimatePresence>
        {ripples.map((r) => (
          <motion.span
            key={r.id}
            initial={{ opacity: 0.3, scale: 0 }}
            animate={{ opacity: 0, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="absolute rounded-full bg-white"
            style={{
              left: r.x - r.size / 2,
              top: r.y - r.size / 2,
              width: r.size,
              height: r.size,
            }}
          />
        ))}
      </AnimatePresence>
    </span>
  );
}
