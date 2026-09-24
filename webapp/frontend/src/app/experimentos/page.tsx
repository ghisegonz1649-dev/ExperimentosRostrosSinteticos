import { ExperimentsDashboard } from "@/components/experiments/experiments-dashboard";
import { obtenerExperimentos } from "@/lib/api";

export default async function ExperimentosPage() {
  const data = await obtenerExperimentos();
  return <ExperimentsDashboard data={data} />;
}
