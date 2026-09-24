"use client";

import { useCallback, useRef, useState } from "react";
import { motion } from "motion/react";
import { UploadCloud, ImageOff, ScanFace, RotateCcw } from "lucide-react";
import { useRipple, RippleLayer } from "@/components/ui/ripple";

const FORMATOS = ["PNG", "JPG", "JPEG", "WEBP", "BMP"];
const EXTENSIONES_VALIDAS = new Set([".png", ".jpg", ".jpeg", ".webp", ".bmp"]);
const TAMANO_MAX_MB = 10;

export function UploadPanel({
  previewUrl,
  deshabilitado,
  analizando,
  completo,
  onArchivoSeleccionado,
  onAnalizar,
  onReiniciar,
}: {
  previewUrl: string | null;
  deshabilitado: boolean;
  analizando: boolean;
  completo: boolean;
  onArchivoSeleccionado: (archivo: File) => void;
  onAnalizar: () => void;
  onReiniciar: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [arrastrando, setArrastrando] = useState(false);
  const [errorLocal, setErrorLocal] = useState<string | null>(null);
  const ripple = useRipple();

  const validarYSeleccionar = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const extension = "." + file.name.split(".").pop()?.toLowerCase();
      if (!EXTENSIONES_VALIDAS.has(extension)) {
        setErrorLocal(`Formato no soportado (${extension}). Usa PNG, JPG, JPEG, WEBP o BMP.`);
        return;
      }
      if (file.size > TAMANO_MAX_MB * 1024 * 1024) {
        setErrorLocal(`La imagen supera los ${TAMANO_MAX_MB} MB permitidos.`);
        return;
      }
      setErrorLocal(null);
      onArchivoSeleccionado(file);
    },
    [onArchivoSeleccionado],
  );

  return (
    <div className="flex flex-col gap-6 border-border p-6 md:h-full md:overflow-y-auto md:border-r md:p-8">
      <div>
        <h2 className="font-heading text-xl font-semibold text-foreground">Cargar rostro</h2>
        <p className="mt-1 text-base text-muted-foreground">
          Sube una imagen para analizar su origen.
        </p>
      </div>

      <div
        role="button"
        tabIndex={0}
        onClick={() => !deshabilitado && inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && !deshabilitado && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!deshabilitado) setArrastrando(true);
        }}
        onDragLeave={() => setArrastrando(false)}
        onDrop={(e) => {
          e.preventDefault();
          setArrastrando(false);
          if (!deshabilitado) validarYSeleccionar(e.dataTransfer.files[0]);
        }}
        className={`focus-ring-accessible glass-panel relative flex min-h-[260px] flex-1 cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed p-6 text-center transition-colors ${
          arrastrando ? "border-accent bg-accent/[0.06] glow-border" : "border-border hover:border-foreground/20"
        } ${deshabilitado ? "pointer-events-none opacity-60" : ""}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.webp,.bmp"
          className="hidden"
          onChange={(e) => validarYSeleccionar(e.target.files?.[0])}
        />

        {previewUrl ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative aspect-square w-full max-w-[220px] overflow-hidden rounded-xl"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={previewUrl} alt="Vista previa del rostro cargado" className="size-full object-cover" />
          </motion.div>
        ) : (
          <>
            <div className="relative flex size-16 items-center justify-center">
              <motion.span
                className="brand-soft absolute inset-0 rounded-2xl"
                animate={{ scale: [1, 1.08, 1], opacity: [0.6, 1, 0.6] }}
                transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
              />
              <UploadCloud className="relative size-6 text-accent" strokeWidth={1.5} />
            </div>
            <div>
              <p className="text-base font-medium text-foreground/90">
                Arrastra una imagen o haz clic para elegirla
              </p>
              <p className="mt-1 text-sm text-muted-foreground">Un rostro por imagen, fondo simple</p>
            </div>
          </>
        )}
      </div>

      {errorLocal && (
        <p className="flex items-center gap-2 text-base text-destructive">
          <ImageOff className="size-4 shrink-0" strokeWidth={1.75} />
          {errorLocal}
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        {FORMATOS.map((f) => (
          <span
            key={f}
            className="rounded-full border border-border bg-muted/50 px-2.5 py-1 text-xs font-medium text-muted-foreground"
          >
            {f}
          </span>
        ))}
        <span className="rounded-full border border-border bg-muted/50 px-2.5 py-1 text-xs font-medium text-muted-foreground">
          Máx. {TAMANO_MAX_MB} MB
        </span>
      </div>

      <div className="flex gap-3">
        <button
          type="button"
          onClick={onAnalizar}
          onPointerDown={ripple.onPointerDown}
          disabled={!previewUrl || analizando}
          className="brand-surface focus-ring-accessible glow-border relative flex flex-1 items-center justify-center gap-2 overflow-hidden rounded-full px-6 py-3 text-base font-semibold text-white transition-transform active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:active:scale-100"
        >
          <ScanFace className="size-4" strokeWidth={2} />
          {analizando ? "Analizando…" : "Analizar"}
          <RippleLayer ripples={ripple.ripples} />
        </button>
        {(previewUrl || completo) && (
          <button
            type="button"
            onClick={onReiniciar}
            disabled={analizando}
            aria-label="Analizar otra imagen"
            className="focus-ring-accessible flex items-center justify-center rounded-full border border-border bg-muted/50 px-4 py-3 text-foreground/80 transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-40"
          >
            <RotateCcw className="size-4" strokeWidth={1.75} />
          </button>
        )}
      </div>
    </div>
  );
}
