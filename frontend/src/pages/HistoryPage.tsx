import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Calendar, Filter, Leaf, Search, Trash2 } from "lucide-react";

import { ConfidenceMeter } from "@/components/shared/ConfidenceMeter";
import { EmptyState } from "@/components/shared/EmptyState";
import { PageHeader } from "@/components/shared/PageHeader";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/lib/auth";
import { deleteDiagnosis, listDiagnoses, type DiagnosisRecord, type HealthStatus } from "@/lib/api";

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

function statusForBadge(status: HealthStatus): "healthy" | "diseased" | "warning" {
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

function formatTimestamp(value: string) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return { date: value, time: "" };
  }
  return {
    date: parsed.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }),
    time: parsed.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }),
  };
}

export default function HistoryPage() {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<HealthStatus | "all">("all");
  const [records, setRecords] = useState<DiagnosisRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const { token } = useAuth();
  const { toast } = useToast();

  useEffect(() => {
    let cancelled = false;

    async function loadHistory() {
      if (!token) {
        return;
      }

      setLoading(true);
      setError("");
      try {
        const response = await listDiagnoses(token, {
          healthStatus: statusFilter === "all" ? null : statusFilter,
        });
        if (!cancelled) {
          setRecords(response.data.items);
        }
      } catch (caughtError) {
        if (!cancelled) {
          const message = caughtError instanceof Error ? caughtError.message : "Could not load diagnosis history.";
          setError(message);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadHistory();
    return () => {
      cancelled = true;
    };
  }, [statusFilter, token]);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) {
      return records;
    }
    return records.filter((record) => {
      const crop = record.predicted_crop.toLowerCase();
      const disease = (record.predicted_disease ?? "").toLowerCase();
      const model = record.model_version.toLowerCase();
      return crop.includes(query) || disease.includes(query) || model.includes(query);
    });
  }, [records, search]);

  const handleDelete = async (id: string) => {
    if (!token) {
      return;
    }

    try {
      await deleteDiagnosis(id, token);
      setRecords((current) => current.filter((record) => record.id !== id));
      toast({
        title: "Diagnosis deleted",
        description: "The saved result was removed from your project database.",
      });
    } catch (caughtError) {
      toast({
        title: "Delete failed",
        description: caughtError instanceof Error ? caughtError.message : "Could not delete diagnosis.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <PageHeader
        title="Detection History"
        description="View saved plant scans from your project backend database."
      />

      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search by crop, disease, or model..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            className="pl-9 rounded-xl"
          />
        </div>
        <div className="flex gap-2 overflow-x-auto pb-1">
          {(["all", "healthy", "diseased", "suspicious"] as const).map((status) => (
            <button
              key={status}
              onClick={() => setStatusFilter(status)}
              className={`px-3 py-1.5 rounded-full text-xs font-medium capitalize transition-colors ${
                statusFilter === status ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
              }`}
            >
              {status}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="bg-card rounded-xl border border-border p-8 text-center text-muted-foreground">
          Loading saved diagnoses...
        </div>
      ) : error ? (
        <div className="rounded-xl border border-destructive/20 bg-destructive/5 p-5 text-sm text-destructive">
          {error}
        </div>
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={Filter}
          title="No saved diagnoses yet"
          description="Run a plant diagnosis, save the result, and it will appear here."
        />
      ) : (
        <div className="space-y-3">
          {filtered.map((record, index) => {
            const timestamp = formatTimestamp(record.created_at);
            return (
              <motion.div
                key={record.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.03 }}
                className="bg-card rounded-xl border border-border p-4 hover:shadow-md transition-shadow"
              >
                <div className="flex items-center gap-4">
                  <img
                    src={record.image_url}
                    alt={`${record.predicted_crop} diagnosis`}
                    className="w-12 h-12 rounded-xl object-cover bg-muted shrink-0"
                    onError={(event) => {
                      event.currentTarget.style.display = "none";
                    }}
                  />
                  <div className="w-12 h-12 rounded-xl bg-muted flex items-center justify-center shrink-0">
                    <Leaf className="h-6 w-6 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-foreground">{displayCropName(record.predicted_crop)}</span>
                      <StatusBadge status={statusForBadge(record.health_status)} />
                    </div>
                    <p className="text-sm text-muted-foreground">{diseaseLabel(record)}</p>
                    <p className="text-xs text-muted-foreground mt-1">Model: {record.model_version}</p>
                  </div>
                  <div className="hidden sm:block w-32">
                    <ConfidenceMeter value={record.confidence_score} size="sm" />
                  </div>
                  <div className="text-right shrink-0">
                    <p className="text-sm text-foreground flex items-center gap-1">
                      <Calendar className="h-3 w-3" /> {timestamp.date}
                    </p>
                    <p className="text-xs text-muted-foreground">{timestamp.time}</p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="rounded-xl text-muted-foreground hover:text-destructive"
                    onClick={() => void handleDelete(record.id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}
