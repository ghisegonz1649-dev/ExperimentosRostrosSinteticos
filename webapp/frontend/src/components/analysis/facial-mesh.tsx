"use client";

import { motion, useReducedMotion } from "motion/react";

const PUNTOS: [number, number][] = [
  [50, 16], [34, 28], [66, 28],
  [30, 38], [44, 36], [56, 36], [70, 38],
  [50, 40], [45, 56], [50, 58], [55, 56],
  [38, 70], [50, 73], [62, 70],
  [26, 56], [30, 76], [50, 90], [70, 76], [74, 56],
];

const EJES: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6], [3, 4], [4, 7], [5, 7], [5, 6],
  [7, 8], [7, 10], [8, 9], [9, 10], [8, 11], [10, 13], [11, 12], [12, 13],
  [3, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 6], [11, 15], [13, 17], [9, 12],
];

export function FacialMesh({ activo }: { activo: boolean }) {
  const reduce = useReducedMotion();
  const animado = activo && !reduce;

  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      className="pointer-events-none absolute inset-0 size-full"
    >
      {EJES.map(([a, b], i) => (
        <motion.line
          key={i}
          x1={PUNTOS[a][0]}
          y1={PUNTOS[a][1]}
          x2={PUNTOS[b][0]}
          y2={PUNTOS[b][1]}
          stroke="url(#mesh-gradient)"
          strokeWidth={0.25}
          initial={{ opacity: 0 }}
          animate={{ opacity: animado ? [0.15, 0.55, 0.15] : 0.25 }}
          transition={
            animado
              ? { duration: 2.4, repeat: Infinity, delay: i * 0.03, ease: "easeInOut" }
              : { duration: 0.6 }
          }
        />
      ))}
      {PUNTOS.map(([x, y], i) => (
        <motion.circle
          key={i}
          cx={x}
          cy={y}
          r={0.6}
          fill={i % 2 === 0 ? "#2d9a91" : "#3b7887"}
          initial={{ opacity: 0.3 }}
          animate={{ opacity: animado ? [0.4, 1, 0.4] : 0.5 }}
          transition={
            animado
              ? { duration: 1.8, repeat: Infinity, delay: i * 0.05, ease: "easeInOut" }
              : { duration: 0.6 }
          }
        />
      ))}
      <defs>
        <linearGradient id="mesh-gradient" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#0f766e" />
          <stop offset="50%" stopColor="#155e75" />
          <stop offset="100%" stopColor="#2d9a91" />
        </linearGradient>
      </defs>
    </svg>
  );
}
