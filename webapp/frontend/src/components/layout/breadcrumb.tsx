import Link from "next/link";
import { ChevronRight } from "lucide-react";

export interface CrumbItem {
  label: string;
  href?: string;
}

export function Breadcrumb({ items }: { items: CrumbItem[] }) {
  return (
    <nav aria-label="Breadcrumb" className="px-6 pt-[116px] text-sm text-muted-foreground md:px-12 md:pt-[132px] lg:px-16">
      <ol className="flex flex-wrap items-center gap-1.5">
        {items.map((item, i) => (
          <li key={item.label} className="flex items-center gap-1.5">
            {i > 0 && <ChevronRight className="size-3" strokeWidth={2} />}
            {item.href ? (
              <Link href={item.href} className="transition-colors hover:text-foreground">
                {item.label}
              </Link>
            ) : (
              <span className="text-foreground/70">{item.label}</span>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}
