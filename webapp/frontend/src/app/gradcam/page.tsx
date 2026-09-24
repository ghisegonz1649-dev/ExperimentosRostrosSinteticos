import { GradCamView } from "@/components/gradcam/gradcam-view";
import { GradCamEmptyState } from "@/components/gradcam/empty-state";
import { Breadcrumb } from "@/components/layout/breadcrumb";
import { obtenerGradCam, type GradCamResultado } from "@/lib/api";

export const dynamic = "force-dynamic";
export const revalidate = 0;

async function cargarResultado(): Promise<GradCamResultado | null> {
  try {
    return await obtenerGradCam();
  } catch {
    return null;
  }
}

export default async function GradCamPage() {
  const resultado = await cargarResultado();

  return (
    <div>
      <Breadcrumb items={[{ label: "Inicio", href: "/" }, { label: "Grad-CAM" }]} />
      {resultado ? <GradCamView resultado={resultado} /> : <GradCamEmptyState />}
    </div>
  );
}
