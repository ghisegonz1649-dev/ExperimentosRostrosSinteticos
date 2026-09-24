"use client";

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { motion, AnimatePresence } from "motion/react";
import { Moon, Sun } from "lucide-react";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [montado, setMontado] = useState(false);

  useEffect(() => {
    const id = requestAnimationFrame(() => setMontado(true));
    return () => cancelAnimationFrame(id);
  }, []);

  if (!montado) {
    return <div className="size-9" aria-hidden />;
  }

  const oscuro = resolvedTheme === "dark";

  return (
    <button
      type="button"
      onClick={() => setTheme(oscuro ? "light" : "dark")}
      aria-label={oscuro ? "Cambiar a modo claro" : "Cambiar a modo oscuro"}
      className="focus-ring-accessible relative flex size-9 items-center justify-center overflow-hidden rounded-full border border-border text-foreground/80 transition-colors hover:bg-muted"
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={oscuro ? "moon" : "sun"}
          initial={{ rotate: -180, opacity: 0, scale: 0.6 }}
          animate={{ rotate: 0, opacity: 1, scale: 1 }}
          exit={{ rotate: 180, opacity: 0, scale: 0.6 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="flex items-center justify-center"
        >
          {oscuro ? (
            <Moon className="size-[18px]" strokeWidth={1.75} />
          ) : (
            <Sun className="size-[18px]" strokeWidth={1.75} />
          )}
        </motion.span>
      </AnimatePresence>
    </button>
  );
}
