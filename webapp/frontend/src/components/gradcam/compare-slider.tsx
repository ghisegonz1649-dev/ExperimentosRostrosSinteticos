"use client";

import { useState } from "react";
import { GripVertical } from "lucide-react";

export function CompareSlider({
  before,
  after,
  beforeLabel,
  afterLabel,
}: {
  before: string;
  after: string;
  beforeLabel: string;
  afterLabel: string;
}) {
  const [pos, setPos] = useState(50);

  return (
    <div className="relative aspect-square w-full select-none overflow-hidden rounded-2xl">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={before} alt={beforeLabel} className="absolute inset-0 size-full object-cover" />
      <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={after} alt={afterLabel} className="absolute inset-0 size-full object-cover" />
      </div>

      <div className="pointer-events-none absolute inset-y-0 z-10 w-0.5 bg-accent/80" style={{ left: `${pos}%` }} />
      <div
        className="pointer-events-none absolute z-10 flex size-8 items-center justify-center rounded-full bg-white/95 shadow-lg"
        style={{ left: `calc(${pos}% - 16px)`, top: "calc(50% - 16px)" }}
      >
        <GripVertical className="size-4 text-slate-700" strokeWidth={2} />
      </div>

      <input
        type="range"
        min={0}
        max={100}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        aria-label={`Comparar ${beforeLabel} con ${afterLabel}`}
        className="absolute inset-0 size-full cursor-ew-resize opacity-0"
      />

      <span className="pointer-events-none absolute left-3 top-3 z-10 rounded-full bg-black/50 px-2.5 py-1 text-xs font-medium text-white backdrop-blur-sm">
        {beforeLabel}
      </span>
      <span className="pointer-events-none absolute right-3 top-3 z-10 rounded-full bg-black/50 px-2.5 py-1 text-xs font-medium text-white backdrop-blur-sm">
        {afterLabel}
      </span>
    </div>
  );
}
