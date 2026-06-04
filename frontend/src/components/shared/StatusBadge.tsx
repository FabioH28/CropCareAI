import { cn } from "@/lib/utils";

type StatusType =
  | "healthy"
  | "diseased"
  | "warning"
  | "critical"
  | "pending"
  | "treated"
  | "monitoring"
  | "active"
  | "applied"
  | "completed";

const statusConfig: Record<StatusType, { label: string; className: string }> = {
  healthy: { label: "Healthy", className: "bg-success/10 text-success border-success/20" },
  diseased: { label: "Diseased", className: "bg-destructive/10 text-destructive border-destructive/20" },
  warning: { label: "Warning", className: "bg-warning/10 text-warning border-warning/20" },
  critical: { label: "Critical", className: "bg-destructive/15 text-destructive border-destructive/30" },
  pending: { label: "Pending", className: "bg-info/10 text-info border-info/20" },
  treated: { label: "Treated", className: "bg-primary/10 text-primary border-primary/20" },
  monitoring: { label: "Monitoring", className: "bg-accent/15 text-accent-foreground border-accent/30" },
  active: { label: "Active", className: "bg-primary/10 text-primary border-primary/20" },
  applied: { label: "Applied", className: "bg-primary/10 text-primary border-primary/20" },
  completed: { label: "Completed", className: "bg-success/10 text-success border-success/20" },
};

export function StatusBadge({ status, className }: { status: StatusType; className?: string }) {
  const config = statusConfig[status];
  return (
    <span className={cn("inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border", config.className, className)}>
      <span className={cn("w-1.5 h-1.5 rounded-full mr-1.5", {
        "bg-success": status === "healthy" || status === "treated" || status === "completed",
        "bg-destructive": status === "diseased" || status === "critical",
        "bg-warning": status === "warning",
        "bg-info": status === "pending",
        "bg-accent": status === "monitoring",
        "bg-primary": status === "active" || status === "applied",
      })} />
      {config.label}
    </span>
  );
}
