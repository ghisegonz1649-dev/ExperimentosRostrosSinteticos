import type { Metadata } from "next";
import { AnalysisWorkspace } from "@/components/analysis/analysis-workspace";

export const metadata: Metadata = {
  title: "Análisis del rostro — Facial Synthesis Detection",
};

export default function AnalisisPage() {
  return <AnalysisWorkspace />;
}
