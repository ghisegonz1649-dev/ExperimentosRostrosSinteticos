"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { UploadPanel } from "@/components/analysis/upload-panel";
import { ProcessPanel } from "@/components/analysis/process-panel";
import { ResultPanel } from "@/components/analysis/result-panel";
import type { EstadoAnalisis } from "@/components/analysis/types";
import { Breadcrumb } from "@/components/layout/breadcrumb";
import { analizarImagen, ApiError, type Prediccion } from "@/lib/api";

const DURACION_MINIMA_MS = 2200;

function esperar(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function AnalysisWorkspace() {
  const [archivo, setArchivo] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [estado, setEstado] = useState<EstadoAnalisis>("inactivo");
  const [resultado, setResultado] = useState<Prediccion | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const previewRef = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
  }, []);

  const seleccionarArchivo = useCallback((file: File) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const url = URL.createObjectURL(file);
    previewRef.current = url;
    setArchivo(file);
    setPreviewUrl(url);
    setResultado(null);
    setErrorMsg(null);
    setEstado("inactivo");
  }, []);

  const analizar = useCallback(async () => {
    if (!archivo) return;
    setEstado("analizando");
    setErrorMsg(null);
    try {
      const [prediccion] = await Promise.all([analizarImagen(archivo), esperar(DURACION_MINIMA_MS)]);
      setResultado(prediccion);
      setEstado("completo");
    } catch (err) {
      setErrorMsg(err instanceof ApiError ? err.message : "Ocurrió un error inesperado. Reintenta en unos segundos.");
      setEstado("error");
    }
  }, [archivo]);

  const reiniciar = useCallback(() => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = null;
    setArchivo(null);
    setPreviewUrl(null);
    setResultado(null);
    setErrorMsg(null);
    setEstado("inactivo");
  }, []);

  return (
    <div>
      <Breadcrumb items={[{ label: "Inicio", href: "/" }, { label: "Análisis" }]} />
      <div className="pt-6 md:grid md:h-[calc(100dvh-104px-40px)] md:grid-cols-3">
        <UploadPanel
          previewUrl={previewUrl}
          deshabilitado={estado === "analizando"}
          analizando={estado === "analizando"}
          completo={estado === "completo"}
          onArchivoSeleccionado={seleccionarArchivo}
          onAnalizar={analizar}
          onReiniciar={reiniciar}
        />
        <ProcessPanel estado={estado} previewUrl={previewUrl} />
        <ResultPanel estado={estado} resultado={resultado} errorMsg={errorMsg} />
      </div>
    </div>
  );
}
