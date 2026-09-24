"use client";

import { useEffect, useRef } from "react";
import { useInView, useReducedMotion, animate } from "motion/react";

export function CountUp({ value, suffix = "" }: { value: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const reduce = useReducedMotion();

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    if (!inView) return;

    if (reduce) {
      el.textContent = `${value.toLocaleString("es-ES")}${suffix}`;
      return;
    }

    const controls = animate(0, value, {
      duration: 1.4,
      ease: [0.16, 1, 0.3, 1],
      onUpdate(latest) {
        el.textContent = `${Math.round(latest).toLocaleString("es-ES")}${suffix}`;
      },
    });

    return () => controls.stop();
  }, [inView, value, suffix, reduce]);

  return <span ref={ref}>0{suffix}</span>;
}
