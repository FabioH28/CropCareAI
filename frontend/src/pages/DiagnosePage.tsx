import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle2,
  Image,
  Info,
  Leaf,
  RotateCcw,
  Save,
  ScanLine,
  Upload,
} from "lucide-react";

import { PageHeader } from "@/components/shared/PageHeader";
import { PlantAdvisorChat } from "@/components/advisor/PlantAdvisorChat";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { ConfidenceMeter } from "@/components/shared/ConfidenceMeter";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/lib/auth";
import {
  canSavePreview,
  getPreviewDiagnosisJob,
  saveDiagnosisPreview,
  startPreviewDiagnosisJob,
  type AdvisoryPayload,
  type DiagnosisPreviewJobPayload,
  type DiagnosisPreviewResponse,
} from "@/lib/api";

type DiagnosisState = "idle" | "analyzing" | "done" | "error";

type AnalysisImage = {
  title: string;
  url: string;
};

function formatDuration(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined) {
    return "--:--";
  }

  const totalSeconds = Math.max(Math.round(seconds), 0);
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`;
}

function toTitleCase(value: string) {
  return value
    .replace(/__/g, " ")
    .replace(/_/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function isUnknownCrop(value: string | null | undefined) {
  return !value || value === "unknown_leaf_crop" || value === "unknown" || value === "unsupported_crop";
}

function normalizeDiseaseName(value: string | null | undefined, healthStatus: string, crop: string | null | undefined) {
  if (isUnknownCrop(crop)) {
    return "No disease label for unsupported crop";
  }

  if (healthStatus === "healthy") {
    return "No disease detected";
  }

  if (!value && healthStatus === "suspicious") {
    return "Suspicious damage detected";
  }

  if (!value) {
    return "Disease not confidently identified";
  }

  const trimmed = value.includes("__") ? value.split("__").slice(1).join(" ") : value;
  return toTitleCase(trimmed);
}

function normalizeCropName(value: string | null | undefined) {
  return isUnknownCrop(value) ? "Unknown / Unsupported Crop" : toTitleCase(value);
}

function mapStatusBadge(status: string): "healthy" | "warning" | "diseased" {
  if (status === "healthy") {
    return "healthy";
  }
  if (status === "suspicious") {
    return "warning";
  }
  return "diseased";
}

function buildDetectionExplanation(
  healthStatus: string,
  predictedDisease: string | null | undefined,
  infectedArea: number | null,
  crop: string | null | undefined,
) {
  if (isUnknownCrop(crop)) {
    const areaText = infectedArea !== null ? ` Visible abnormal area is about ${infectedArea.toFixed(1)}%.` : "";
    return `The leaf does not confidently match one of the crops supported by this model, so no named disease was assigned.${areaText}`;
  }

  if (healthStatus === "healthy") {
    return "No strong disease evidence was found in this leaf.";
  }

  if (healthStatus === "suspicious" && !predictedDisease) {
    const areaText = infectedArea !== null ? `${infectedArea.toFixed(1)}% of the leaf looks abnormal` : "the CV stage found abnormal leaf regions";
    return `Abnormal regions were found, but they did not match one disease class clearly. ${areaText}.`;
  }

  return "This result combines the disease label with CV-based severity and affected-area estimates.";
}

function extractAnalysisImages(result: DiagnosisPreviewResponse | null): AnalysisImage[] {
  if (!result) {
    return [];
  }

  const analysis = result.data.raw_prediction_json as Record<string, unknown>;
  const roi = (analysis.roi as Record<string, unknown> | undefined) ?? {};
  const outputs = Array.isArray(analysis.outputs) ? (analysis.outputs as Record<string, unknown>[]) : [];

  const items: AnalysisImage[] = [];
  const leafUrl = typeof roi.classifier_primary_url === "string" ? roi.classifier_primary_url : null;
  const contextUrl = typeof roi.context_crop_url === "string" ? roi.context_crop_url : null;

  if (leafUrl) {
    items.push({ title: "Leaf-Focused ROI", url: leafUrl });
  } else if (contextUrl) {
    items.push({ title: "Leaf Context Crop", url: contextUrl });
  }

  const priorityLabels = [
    "marked_disease_spots",
    "consensus_overlay",
    "core_segmentation_comparison_grid",
  ];

  const prioritizedOutputs = [
    ...priorityLabels.flatMap((label) =>
      outputs.filter((output) => typeof output.label === "string" && output.label === label),
    ),
    ...outputs.filter(
      (output) => !priorityLabels.includes(typeof output.label === "string" ? output.label : ""),
    ),
  ];

  for (const output of prioritizedOutputs) {
    const imageUrl = typeof output.image_url === "string" ? output.image_url : null;
    if (!imageUrl || items.some((item) => item.url === imageUrl)) {
      continue;
    }

    const rawTitle =
      (typeof output.title === "string" && output.title) ||
      (typeof output.name === "string" && output.name) ||
      (typeof output.relative_path === "string" && output.relative_path.split("/").pop()) ||
      "CV Output";

    items.push({ title: toTitleCase(rawTitle.replace(/\.[^.]+$/, "")), url: imageUrl });
    if (items.length >= 3) {
      break;
    }
  }

  return items;
}

export default function DiagnosePage() {
  const [state, setState] = useState<DiagnosisState>("idle");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<DiagnosisPreviewResponse | null>(null);
  const [analysisJob, setAnalysisJob] = useState<DiagnosisPreviewJobPayload | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedDiagnosisId, setSavedDiagnosisId] = useState<string | null>(null);
  const activeRunRef = useRef(0);

  const { toast } = useToast();
  const { token } = useAuth();

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  useEffect(() => {
    return () => {
      activeRunRef.current += 1;
    };
  }, []);

  const reset = useCallback(() => {
    activeRunRef.current += 1;
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setResult(null);
    setAnalysisJob(null);
    setSaving(false);
    setSavedDiagnosisId(null);
    setError("");
    setState("idle");
  }, [previewUrl]);

  const handleFile = useCallback(
    async (file: File) => {
      const runId = activeRunRef.current + 1;
      activeRunRef.current = runId;

      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }

      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setAnalysisJob({
        job_id: "starting",
        status: "queued",
        stage_key: "uploading",
        stage_label: "Uploading image to the backend",
        progress_percent: 2,
        elapsed_seconds: 0,
        estimated_total_seconds: null,
        remaining_seconds: null,
        result: null,
        advisory: null,
        error_message: null,
        created_at: new Date().toISOString(),
        started_at: null,
        completed_at: null,
      });
      setSavedDiagnosisId(null);
      setError("");
      setState("analyzing");

      try {
        const queuedJob = await startPreviewDiagnosisJob(file);
        if (activeRunRef.current !== runId) {
          return;
        }

        setAnalysisJob(queuedJob.data);

        while (true) {
          const jobResponse = await getPreviewDiagnosisJob(queuedJob.data.job_id);
          if (activeRunRef.current !== runId) {
            return;
          }

          const nextJob = jobResponse.data;
          setAnalysisJob(nextJob);

          if (nextJob.status === "completed" && nextJob.result) {
            const nextResult: DiagnosisPreviewResponse = {
              success: true,
              message: "Live diagnosis complete",
              data: nextJob.result,
              meta: nextJob.advisory ? { advisory: nextJob.advisory } : undefined,
            };
            setResult(nextResult);
            setState("done");
            break;
          }

          if (nextJob.status === "failed") {
            throw new Error(nextJob.error_message || "Plant analysis failed.");
          }

          await new Promise((resolve) => {
            window.setTimeout(resolve, 700);
          });
        }

        toast({
          title: "Live diagnosis complete",
          description: "The backend analyzed your image with the current production model.",
        });
      } catch (caughtError) {
        const message = caughtError instanceof Error ? caughtError.message : "Plant analysis failed.";
        setError(message);
        setAnalysisJob((current) =>
          current
            ? {
                ...current,
                status: "failed",
                stage_key: "failed",
                stage_label: "Analysis failed",
                error_message: message,
              }
            : null,
        );
        setState("error");
        toast({
          title: "Diagnosis failed",
          description: message,
          variant: "destructive",
        });
      }
    },
    [previewUrl, toast],
  );

  const handleSave = useCallback(async () => {
    if (!result || !token) {
      toast({
        title: "Cannot save yet",
        description: "Please sign in and run a diagnosis first.",
        variant: "destructive",
      });
      return;
    }

    if (!canSavePreview(result)) {
      toast({
        title: "Preview cannot be saved",
        description: "The backend did not return a stored image path for this preview.",
        variant: "destructive",
      });
      return;
    }

    setSaving(true);
    try {
      const response = await saveDiagnosisPreview(result, token);
      setSavedDiagnosisId(response.data.id);
      toast({
        title: "Diagnosis saved",
        description: "You can now find this result in Detection History.",
      });
    } catch (caughtError) {
      const message = caughtError instanceof Error ? caughtError.message : "Could not save diagnosis.";
      toast({
        title: "Save failed",
        description: message,
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  }, [result, token, toast]);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragOver(false);
      const file = event.dataTransfer.files[0];
      if (file) {
        void handleFile(file);
      }
    },
    [handleFile],
  );

  const advisory = result?.meta?.advisory as AdvisoryPayload | undefined;
  const analysis = (result?.data.raw_prediction_json ?? {}) as Record<string, unknown>;
  const summary = (analysis.summary ?? {}) as Record<string, unknown>;
  const roi = (analysis.roi ?? {}) as Record<string, unknown>;
  const modelIntegration = (analysis.model_integration ?? {}) as Record<string, unknown>;
  const dlClassifier = (modelIntegration.dl_classifier ?? {}) as Record<string, unknown>;
  const postprocessing = (analysis.postprocessing ?? {}) as Record<string, unknown>;
  const infectedArea = typeof summary.infected_area_percentage === "number" ? summary.infected_area_percentage : null;
  const cvImages = useMemo(() => extractAnalysisImages(result), [result]);
  const progressPercent = Math.max(analysisJob?.progress_percent ?? 0, 2);
  const progressStageLabel = analysisJob?.stage_label ?? "Preparing plant analysis";
  const shortTreatmentSteps = advisory?.treatment_steps.slice(0, 2) ?? [];
  const shortActionReasons = advisory?.action_explanations.slice(0, 2) ?? [];
  const shortPreventionTips = advisory?.prevention_tips.slice(0, 2) ?? [];

  const displayCrop = normalizeCropName(result?.data.predicted_crop);
  const displayDisease = normalizeDiseaseName(
    result?.data.predicted_disease,
    result?.data.health_status ?? "diseased",
    result?.data.predicted_crop,
  );
  const statusBadge = mapStatusBadge(result?.data.health_status ?? "diseased");
  const leafDetected = Boolean(roi.leaf_detected);
  const branchUsed = typeof dlClassifier.branch_used === "string" ? dlClassifier.branch_used : null;
  const isSuspiciousWithoutNamedDisease =
    result?.data.health_status === "suspicious" && !result?.data.predicted_disease;
  const lesionRegionCount =
    typeof postprocessing.lesion_region_count === "number" ? postprocessing.lesion_region_count : null;
  const detectionExplanation = buildDetectionExplanation(
    result?.data.health_status ?? "diseased",
    result?.data.predicted_disease,
    infectedArea,
    result?.data.predicted_crop,
  );

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <PageHeader
        title="Plant Diagnosis"
        description="Upload a plant photo to run the live backend CV + CNN pipeline with the current production model."
      />

      <AnimatePresence mode="wait">
        {state === "idle" && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-4"
          >
            <div
              onDrop={handleDrop}
              onDragOver={(event) => {
                event.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 cursor-pointer ${
                dragOver ? "border-primary bg-primary/5 scale-[1.01]" : "border-border hover:border-primary/40 hover:bg-muted/30"
              }`}
            >
              <div className="p-4 rounded-2xl bg-primary/10 w-fit mx-auto mb-4">
                <Upload className="h-10 w-10 text-primary" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Drop your plant image here</h3>
              <p className="text-sm text-muted-foreground mb-6">or choose a photo from your device</p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                <label className="cursor-pointer">
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(event) => event.target.files?.[0] && void handleFile(event.target.files[0])}
                  />
                  <Button variant="outline" className="rounded-xl pointer-events-none">
                    <Image className="mr-2 h-4 w-4" /> Choose File
                  </Button>
                </label>
              </div>
              <p className="text-xs text-muted-foreground mt-4">
                Supports JPG, PNG, WebP. The app uses the current best production model automatically.
              </p>
            </div>
          </motion.div>
        )}

        {state === "analyzing" && (
          <motion.div
            key="loading"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="bg-card rounded-2xl border border-border p-8 text-center"
          >
            {previewUrl ? <img src={previewUrl} alt="Upload preview" className="w-48 h-48 object-cover rounded-xl mx-auto mb-6" /> : null}
            <div className="flex items-center justify-center gap-3 mb-4">
              <div className="w-8 h-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
              <span className="text-lg font-semibold text-foreground">AI is analyzing your plant...</span>
            </div>
            <div className="max-w-xl mx-auto space-y-4">
              <div className="space-y-2 text-left">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-foreground">{progressStageLabel}</span>
                  <span className="text-muted-foreground">{progressPercent.toFixed(0)}%</span>
                </div>
                <Progress value={progressPercent} className="h-3" />
              </div>
              <div className="rounded-xl border border-border bg-muted/30 px-4 py-3 text-left text-sm">
                <p className="text-muted-foreground">Current phase</p>
                <p className="font-medium text-foreground">{progressStageLabel}</p>
              </div>
              <p className="text-sm text-muted-foreground">
                This progress bar advances when real backend stages finish: upload validation, CV analysis, CNN classification, and advisory generation.
              </p>
            </div>
          </motion.div>
        )}

        {state === "error" && (
          <motion.div
            key="error"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="rounded-2xl border border-destructive/20 bg-destructive/5 p-6 space-y-4"
          >
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 text-destructive mt-0.5" />
              <div>
                <h3 className="text-lg font-semibold text-foreground">Analysis failed</h3>
                <p className="text-sm text-muted-foreground">{error}</p>
              </div>
            </div>
            <Button variant="outline" className="rounded-xl" onClick={reset}>
              <RotateCcw className="mr-2 h-4 w-4" /> Try Another Image
            </Button>
          </motion.div>
        )}

        {state === "done" && result && (
          <motion.div key="result" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
            <div className="bg-card rounded-2xl border border-border overflow-hidden">
              <div className="p-6 flex flex-col lg:flex-row gap-6">
                {previewUrl ? <img src={previewUrl} alt="Diagnosed plant" className="w-full lg:w-56 h-56 object-cover rounded-xl" /> : null}
                <div className="flex-1 space-y-4">
                  <div className="flex items-center gap-3 flex-wrap">
                    <h3 className="text-2xl font-bold text-foreground">{displayCrop}</h3>
                    <StatusBadge status={statusBadge} />
                  </div>

                  <div className="grid sm:grid-cols-3 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Result</p>
                      <p className="font-semibold text-foreground">{displayDisease}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Model</p>
                      <p className="font-semibold text-foreground break-all">{result.data.model_version}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Time</p>
                      <p className="font-semibold text-foreground">{formatDuration(analysisJob?.elapsed_seconds)}</p>
                    </div>
                  </div>

                  <div className="rounded-xl border border-primary/20 bg-primary/5 px-4 py-3 text-sm text-muted-foreground">
                    <p className="font-medium text-foreground mb-1">Quick note</p>
                    <p>{detectionExplanation}</p>
                  </div>

                  <div className="max-w-sm">
                    <p className="text-sm text-muted-foreground mb-1">Confidence</p>
                    <ConfidenceMeter value={result.data.confidence_score} size="md" />
                  </div>

                  <div className="flex gap-3 flex-wrap">
                    {infectedArea !== null ? (
                      <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium border bg-primary/10 text-primary border-primary/20">
                        {isSuspiciousWithoutNamedDisease ? "CV estimated abnormal area" : "CV estimated affected area"}: {infectedArea.toFixed(1)}%
                      </span>
                    ) : null}
                    {lesionRegionCount !== null && lesionRegionCount > 0 ? (
                      <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium border bg-muted/40 text-foreground border-border">
                        Marked spots: {lesionRegionCount}
                      </span>
                    ) : null}
                  </div>

                  <div className="flex gap-2 flex-wrap text-sm">
                    <span className="inline-flex items-center gap-1 rounded-full border border-border bg-muted/30 px-3 py-1.5">
                      {leafDetected ? "Leaf isolated" : "Full image used"}
                    </span>
                    <span className="inline-flex items-center gap-1 rounded-full border border-border bg-muted/30 px-3 py-1.5">
                      {branchUsed ? `Branch: ${toTitleCase(branchUsed)}` : "Default branch"}
                    </span>
                    <span className="inline-flex items-center gap-1 rounded-full border border-border bg-muted/30 px-3 py-1.5">
                      {advisory?.advisory_source ? `Explanation: ${toTitleCase(advisory.advisory_source)}` : "Explanation ready"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <PlantAdvisorChat currentResult={result} title="Ask About This Scan" compact />

            {cvImages.length > 0 && (
              <details className="bg-card rounded-2xl border border-border p-5">
                <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-left">
                  <span className="font-semibold text-foreground flex items-center gap-2">
                    <ScanLine className="h-4 w-4 text-primary" /> Computer Vision Evidence
                  </span>
                  <span className="text-sm text-muted-foreground">Show images</span>
                </summary>
                <div className="grid md:grid-cols-3 gap-4 pt-4">
                  {cvImages.map((item) => (
                    <div key={item.url} className="space-y-2">
                      <img src={item.url} alt={item.title} className="w-full h-44 object-cover rounded-xl border border-border bg-muted/20" />
                      <p className="text-sm font-medium text-foreground">{item.title}</p>
                    </div>
                  ))}
                </div>
              </details>
            )}

            {advisory && (
              <div className="grid lg:grid-cols-3 gap-4">
                <div className="bg-card rounded-xl border border-border p-5 space-y-4">
                  <h4 className="font-semibold text-foreground flex items-center gap-2">
                    <Info className="h-4 w-4 text-primary" /> Short Explanation
                  </h4>
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">What happened</p>
                    <p className="text-sm text-foreground">{advisory.what_happened}</p>
                  </div>
                  <div className="pt-4 border-t border-border">
                    <p className="text-sm text-muted-foreground mb-1">Likely cause</p>
                    <p className="text-sm text-foreground">{advisory.likely_cause}</p>
                  </div>
                </div>

                <div className="bg-card rounded-xl border border-border p-5 space-y-4">
                  <h4 className="font-semibold text-foreground flex items-center gap-2">
                    <Leaf className="h-4 w-4 text-primary" /> What To Do
                  </h4>
                  <ul className="space-y-3">
                    {shortTreatmentSteps.map((step, index) => (
                      <li key={`${step}-${index}`} className="flex items-start gap-2">
                        <CheckCircle2 className="h-4 w-4 text-success mt-0.5 shrink-0" />
                        <div>
                          <p className="text-sm text-foreground">{step}</p>
                          {shortActionReasons[index] ? (
                            <p className="text-xs text-muted-foreground mt-1">{shortActionReasons[index]}</p>
                          ) : null}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-card rounded-xl border border-border p-5 space-y-4">
                  <h4 className="font-semibold text-foreground flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-primary" /> Why It Matters
                  </h4>
                  <div>
                    <p className="text-sm text-foreground">{advisory.why_it_matters}</p>
                  </div>
                  <div className="pt-4 border-t border-border">
                    <p className="text-sm text-muted-foreground mb-1">Urgency</p>
                    <p className="text-sm text-foreground">{advisory.urgency_guidance}</p>
                  </div>
                  <div className="pt-4 border-t border-border">
                    <p className="text-sm text-muted-foreground mb-1">Next check</p>
                    <p className="text-sm text-foreground">{advisory.follow_up_recommendation}</p>
                  </div>
                </div>

                {shortPreventionTips.length > 0 ? (
                  <div className="lg:col-span-3 bg-card rounded-xl border border-border p-5">
                    <p className="text-sm font-medium text-foreground mb-3">Keep in mind</p>
                    <div className="flex flex-wrap gap-2">
                      {shortPreventionTips.map((tip, index) => (
                        <span key={`${tip}-${index}`} className="inline-flex rounded-full border border-primary/20 bg-primary/5 px-3 py-1.5 text-sm text-foreground">
                          {tip}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            )}

            <div className="flex flex-wrap gap-3">
              <Button
                className="rounded-xl gradient-hero border-0 text-primary-foreground"
                onClick={handleSave}
                disabled={saving || Boolean(savedDiagnosisId)}
              >
                <Save className="mr-2 h-4 w-4" /> {savedDiagnosisId ? "Saved To History" : saving ? "Saving..." : "Save Result"}
              </Button>
              <Button variant="outline" className="rounded-xl" onClick={reset}>
                <RotateCcw className="mr-2 h-4 w-4" /> Upload Another
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
