import type { Region } from "@/lib/api";

export function generarExplicacion(regiones: Region[], etiquetaClase: string, esSintetico: boolean): string {
  const top = regiones.slice(0, 2).map((r) => r.region.toLowerCase());
  const zonas = top.length === 2 ? `las regiones de ${top[0]} y ${top[1]}` : `la región de ${top[0]}`;

  if (esSintetico) {
    return `El modelo concentró su atención principalmente en ${zonas}, donde detectó patrones compatibles con la huella del generador ${etiquetaClase}.`;
  }
  return `El modelo concentró su atención principalmente en ${zonas}, sin encontrar patrones compatibles con ningún generador conocido, consistente con un rostro real.`;
}
