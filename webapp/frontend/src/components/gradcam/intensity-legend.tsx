export function IntensityLegend() {
  return (
    <div className="flex items-center gap-3">
      <span className="text-sm text-muted-foreground">Baja</span>
      <div
        className="h-2.5 w-40 rounded-full"
        style={{ background: "linear-gradient(90deg, rgb(0,60,255), rgb(160,120,255), rgb(255,187,255))" }}
      />
      <span className="text-sm text-muted-foreground">Alta</span>
      <span className="ml-1 text-sm text-muted-foreground">Intensidad de activación</span>
    </div>
  );
}
