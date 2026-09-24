"use client";

import { useId } from "react";

const RADIO = 42;
const CIRCUNFERENCIA = 2 * Math.PI * RADIO;

export function CircularProgress({
  porcentaje,
  tamanoPx = 112,
  colorSolido,
  numeroClassName = "text-2xl",
}: {
  porcentaje: number;
  tamanoPx?: number;
  /** Si se pasa, el anillo usa este color plano en vez del gradiente de marca. */
  colorSolido?: string;
  numeroClassName?: string;
}) {
  const offset = CIRCUNFERENCIA * (1 - porcentaje / 100);
  const gradientId = useId();

  return (
    <div className="relative flex items-center justify-center" style={{ width: tamanoPx, height: tamanoPx }}>
      <svg viewBox="0 0 100 100" className="size-full -rotate-90">
        <circle cx="50" cy="50" r={RADIO} fill="none" stroke="var(--border)" strokeWidth={6} />
        <circle
          cx="50"
          cy="50"
          r={RADIO}
          fill="none"
          stroke={colorSolido ?? `url(#${gradientId})`}
          strokeWidth={6}
          strokeLinecap="round"
          strokeDasharray={CIRCUNFERENCIA}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.3s ease" }}
        />
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="var(--primary)" />
            <stop offset="50%" stopColor="var(--accent-2)" />
            <stop offset="100%" stopColor="var(--accent-3)" />
          </linearGradient>
        </defs>
      </svg>
      <span
        className={`absolute font-heading font-semibold tabular-nums text-foreground ${numeroClassName}`}
        style={colorSolido ? { color: colorSolido } : undefined}
      >
        {Math.round(porcentaje)}%
      </span>
    </div>
  );
}
