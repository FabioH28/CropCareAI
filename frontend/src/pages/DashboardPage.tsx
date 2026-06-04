import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { AlertTriangle, ArrowRight, Leaf, ScanLine, Upload, User } from "lucide-react";

import { ConfidenceMeter } from "@/components/shared/ConfidenceMeter";
import { EmptyState } from "@/components/shared/EmptyState";
import { PageHeader } from "@/components/shared/PageHeader";
import { StatCard } from "@/components/shared/StatCard";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/auth";
import {
  fetchDashboardSummary,
  listDiagnoses,
  type DashboardSummaryPayload,
  type DiagnosisRecord,
} from "@/lib/api";

function toTitleCase(value: string) {
  return value
    .replace(/__/g, " ")
    .replace(/_/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function displayCropName(value: string) {
  return value === "unknown_leaf_crop" ? "Unknown / Unsupported Crop" : toTitleCase(value);
}

function statusForBadge(status: DiagnosisRecord["health_status"]): "healthy" | "diseased" | "warning" {
  return status === "suspicious" ? "warning" : status;
}

function diseaseLabel(record: DiagnosisRecord) {
  if (record.predicted_crop === "unknown_leaf_crop") {
    return "Unsupported crop - disease label withheld";
  }
  if (record.health_status === "healthy") {
    return "No disease detected";
  }
  if (record.health_status === "suspicious" && !record.predicted_disease) {
    return "Suspicious damage detected";
  }
  if (!record.predicted_disease) {
    return "Disease not confidently identified";
  }
  return toTitleCase(record.predicted_disease);
}

function relativeTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  const diffMs = Date.now() - date.getTime();
  const hours = Math.round(diffMs / (1000 * 60 * 60));
  if (hours < 1) {
    return "just now";
  }
  if (hours < 24) {
    return `${hours}h ago`;
  }

  const days = Math.round(hours / 24);
  if (days < 7) {
    return `${days}d ago`;
  }

  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function countByStatus(summary: DashboardSummaryPayload, status: string): number {
  return summary.crop_health_distribution.find((item) => item.health_status === status)?.count ?? 0;
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardSummaryPayload | null>(null);
  const [diagnoses, setDiagnoses] = useState<DiagnosisRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const { token, user } = useAuth();

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      if (!token) {
        return;
      }

      setLoading(true);
      setError("");
      try {
        const [summaryResponse, diagnosesResponse] = await Promise.all([
          fetchDashboardSummary(token),
          listDiagnoses(token),
        ]);

        if (cancelled) {
          return;
        }

        setSummary(summaryResponse.data);
        setDiagnoses(diagnosesResponse.data.items);
      } catch (caughtError) {
        if (!cancelled) {
          setError(caughtError instanceof Error ? caughtError.message : "Could not load the dashboard.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadDashboard();
    return () => {
      cancelled = true;
    };
  }, [token]);

  const healthyCount = useMemo(() => (summary ? countByStatus(summary, "healthy") : 0), [summary]);
  const flaggedCount = useMemo(
    () => diagnoses.filter((record) => record.health_status !== "healthy").length,
    [diagnoses],
  );
  const latestRecord = diagnoses[0] ?? null;

  if (loading) {
    return (
      <div className="space-y-6 max-w-6xl">
        <PageHeader title="Dashboard" description="Loading your saved diagnosis overview..." />
        <div className="bg-card rounded-xl border border-border p-8 text-center text-muted-foreground">
          Loading dashboard data from the backend...
        </div>
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="space-y-6 max-w-6xl">
        <PageHeader title="Dashboard" description="Your saved diagnosis overview." />
        <div className="rounded-xl border border-destructive/20 bg-destructive/5 p-5 text-sm text-destructive">
          {error || "Could not load the dashboard."}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-6xl">
      <PageHeader
        title={`Welcome back, ${user?.fullName || "Farmer"}`}
        description="This dashboard is focused on the real app flow: run diagnoses, review history, and keep your account updated."
        actions={
          <div className="flex flex-wrap gap-2">
            <Link to="/diagnose">
              <Button className="rounded-xl gradient-hero border-0 text-primary-foreground">
                <Upload className="mr-2 h-4 w-4" /> New Diagnosis
              </Button>
            </Link>
            <Link to="/history">
              <Button variant="outline" className="rounded-xl">
                <ScanLine className="mr-2 h-4 w-4" /> Open History
              </Button>
            </Link>
          </div>
        }
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          title="Saved Scans"
          value={summary.total_diagnoses}
          icon={ScanLine}
          iconClassName="bg-primary/10 text-primary"
          subtitle="Results stored in SQL"
        />
        <StatCard
          title="Flagged Results"
          value={flaggedCount}
          icon={AlertTriangle}
          iconClassName="bg-warning/10 text-warning"
          subtitle="Diseased or suspicious scans"
        />
        <StatCard
          title="Healthy Scans"
          value={healthyCount}
          icon={Leaf}
          iconClassName="bg-success/10 text-success"
          subtitle="Saved results labeled healthy by the model"
        />
      </div>

      <div className="grid lg:grid-cols-[1.2fr_0.8fr] gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-card rounded-xl border border-border p-5"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-foreground">Recent Diagnoses</h3>
            <Link to="/history" className="text-sm text-primary hover:underline">
              View all
            </Link>
          </div>

          {diagnoses.length === 0 ? (
            <EmptyState
              icon={ScanLine}
              title="No saved diagnoses yet"
              description="Run your first plant diagnosis and save it to build your dashboard history."
              actionLabel="Open Diagnose"
              onAction={() => {
                window.location.href = "/diagnose";
              }}
            />
          ) : (
            <div className="space-y-3">
              {diagnoses.slice(0, 6).map((record, index) => (
                <motion.div
                  key={record.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.04 }}
                  className="rounded-xl border border-border bg-muted/20 p-4"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
                      <ScanLine className="h-5 w-5 text-primary" />
                    </div>
                    <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium text-foreground">{displayCropName(record.predicted_crop)}</span>
                        <StatusBadge status={statusForBadge(record.health_status)} />
                      </div>
                      <p className="text-sm text-muted-foreground">{diseaseLabel(record)}</p>
                    </div>
                    <div className="text-right shrink-0">
                      <ConfidenceMeter value={record.confidence_score} size="sm" />
                      <p className="text-xs text-muted-foreground mt-1">{relativeTime(record.created_at)}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08 }}
          className="space-y-4"
        >
          <div className="bg-card rounded-xl border border-border p-5">
            <h3 className="font-semibold text-foreground mb-3">Current Focus</h3>
            {latestRecord ? (
              <div className="space-y-3">
                <div>
                  <p className="text-sm text-muted-foreground">Latest saved result</p>
                  <p className="font-semibold text-foreground">
                    {displayCropName(latestRecord.predicted_crop)} - {diseaseLabel(latestRecord)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Recorded</p>
                  <p className="text-sm text-foreground">{relativeTime(latestRecord.created_at)}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Model</p>
                  <p className="text-sm text-foreground break-all">{latestRecord.model_version}</p>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No saved result exists yet. Run a diagnosis to populate the dashboard.
              </p>
            )}
          </div>

          <div className="bg-card rounded-xl border border-border p-5">
            <h3 className="font-semibold text-foreground mb-3">Quick Access</h3>
            <div className="space-y-3">
              <Link to="/diagnose" className="flex items-center justify-between rounded-xl border border-border px-4 py-3 hover:bg-muted/30 transition-colors">
                <span className="text-sm font-medium text-foreground">Run a new diagnosis</span>
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
              </Link>
              <Link to="/history" className="flex items-center justify-between rounded-xl border border-border px-4 py-3 hover:bg-muted/30 transition-colors">
                <span className="text-sm font-medium text-foreground">Review saved history</span>
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
              </Link>
              <Link to="/profile" className="flex items-center justify-between rounded-xl border border-border px-4 py-3 hover:bg-muted/30 transition-colors">
                <span className="text-sm font-medium text-foreground">Update your profile</span>
                <User className="h-4 w-4 text-muted-foreground" />
              </Link>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
