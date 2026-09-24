"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { motion, useMotionValueEvent, useScroll, AnimatePresence } from "motion/react";
import { Menu, X, ArrowRight } from "lucide-react";
import { ThemeToggle } from "@/components/layout/theme-toggle";
import { useRipple, RippleLayer } from "@/components/ui/ripple";

const NAV_ITEMS = [
  { href: "/", label: "Inicio" },
  { href: "/analisis", label: "Análisis" },
  { href: "/gradcam", label: "Grad-CAM" },
  { href: "/experimentos", label: "Experimentos" },
] as const;

export function Navbar() {
  const pathname = usePathname();
  const [comprimido, setComprimido] = useState(false);
  const [abierto, setAbierto] = useState(false);
  const { scrollY } = useScroll();
  const ripple = useRipple();

  useMotionValueEvent(scrollY, "change", (y) => {
    setComprimido(y > 50);
  });

  const activo = (href: string) => (href === "/" ? pathname === "/" : pathname.startsWith(href));

  return (
    <>
      <motion.header
        animate={{ height: comprimido ? 88 : 104 }}
        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="fixed inset-x-0 top-0 z-[1000] border-b backdrop-blur-[10px]"
        style={{
          backgroundColor: "var(--navbar-bg)",
          borderColor: comprimido ? "var(--border)" : "transparent",
        }}
      >
        <div className="mx-auto flex h-full max-w-7xl items-center justify-between gap-4 px-4 md:px-8">
          <div className="flex min-w-0 items-center gap-3">
            {/* Escudo institucional (UAEM · ISI). El PNG lleva fondo
                transparente para que funcione en tema claro y oscuro. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/logo-isi.png"
              alt="Ingeniería en Sistemas Inteligentes — UAEM"
              className="h-14 w-auto shrink-0 select-none md:h-[72px]"
            />
            <span className="hidden h-11 w-px shrink-0 bg-border sm:block" />
            <Link
              href="/"
              className="flex min-w-0 items-center font-heading text-base font-semibold text-foreground"
            >
              <span className="hidden truncate sm:inline">Facial Synthesis Detection</span>
            </Link>
          </div>

          <nav className="hidden items-center gap-1 md:flex">
            {NAV_ITEMS.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className={`focus-ring-accessible relative rounded-full px-3.5 py-2 text-sm font-medium transition-colors ${
                  activo(href) ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {activo(href) && (
                  <motion.span
                    layoutId="navbar-active-pill"
                    className="brand-active absolute inset-0 rounded-full border"
                    transition={{ type: "spring", stiffness: 400, damping: 32 }}
                  />
                )}
                <span className="relative z-10">{label}</span>
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-2 md:gap-3">
            <ThemeToggle />
            <Link
              href="/analisis"
              onPointerDown={ripple.onPointerDown}
              className="brand-surface focus-ring-accessible glow-border relative hidden items-center gap-1.5 overflow-hidden rounded-full px-4 py-2 text-sm font-semibold text-white transition-transform hover:shadow-lg active:scale-[0.98] md:flex"
            >
              Iniciar análisis
              <ArrowRight className="size-3.5" strokeWidth={2} />
              <RippleLayer ripples={ripple.ripples} />
            </Link>
            <button
              type="button"
              aria-label={abierto ? "Cerrar menú" : "Abrir menú"}
              onClick={() => setAbierto((v) => !v)}
              className="focus-ring-accessible rounded-lg p-2 text-foreground md:hidden"
            >
              {abierto ? <X className="size-5" /> : <Menu className="size-5" />}
            </button>
          </div>
        </div>
      </motion.header>

      <AnimatePresence>
        {abierto && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            // Se ancla justo debajo del header, que cambia de alto al desplazarse.
            className={`fixed inset-x-0 z-[999] border-b border-border bg-background p-4 shadow-xl md:hidden ${
              comprimido ? "top-[88px]" : "top-[104px]"
            }`}
          >
            <nav className="flex flex-col gap-1">
              {NAV_ITEMS.map(({ href, label }) => (
                <Link
                  key={href}
                  href={href}
                  onClick={() => setAbierto(false)}
                  className={`focus-ring-accessible rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                    activo(href) ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {label}
                </Link>
              ))}
              <Link
                href="/analisis"
                onClick={() => setAbierto(false)}
                className="brand-surface mt-2 flex items-center justify-center gap-1.5 rounded-full px-4 py-2.5 text-sm font-semibold text-white"
              >
                Iniciar análisis
                <ArrowRight className="size-3.5" strokeWidth={2} />
              </Link>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
