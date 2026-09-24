import { Hero } from "@/components/home/hero";
import { StatsStrip } from "@/components/home/stats";
import { Timeline } from "@/components/home/timeline";
import { WhyItMatters } from "@/components/home/why-it-matters";
import { obtenerStats, type Stats } from "@/lib/api";

const STATS_RESPALDO: Stats = {
  generadores: 4,
  arquitecturas_cnn: 3,
  imagenes_dataset: 91000,
  experimentos: 5,
};

export default async function InicioPage() {
  const stats = await obtenerStats().catch(() => STATS_RESPALDO);

  return (
    <div className="relative">
      <Hero />
      <StatsStrip stats={stats} />
      <Timeline />
      <WhyItMatters />
      <footer className="relative border-t border-border px-6 py-8 text-center text-xs text-muted-foreground md:px-12">
        Proyecto de tesis · Detección y atribución de rostros sintéticos con redes neuronales convolucionales
      </footer>
    </div>
  );
}
