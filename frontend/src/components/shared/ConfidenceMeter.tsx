import { cn } from "@/lib/utils";

export function ConfidenceMeter({ value, size = "md" }: { value: number; size?: "sm" | "md" | "lg" }) {
  const percentage = Math.round(value * 100);
  const color = percentage >= 85 ? "bg-success" : percentage >= 60 ? "bg-warning" : "bg-destructive";
  const heights = { sm: "h-1.5", md: "h-2", lg: "h-3" };

  return (
    <div className="flex items-center gap-2">
      <div className={cn("flex-1 bg-muted rounded-full overflow-hidden", heights[size])}>
        <div
          className={cn("h-full rounded-full transition-all duration-700 ease-out", color)}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-sm font-semibold text-foreground tabular-nums">{percentage}%</span>
    </div>
  );
}
