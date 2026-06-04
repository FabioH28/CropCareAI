import { useMemo, useState } from "react";
import { Bot, Brain, History, Leaf, Send, Sparkles, Zap } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/lib/auth";
import {
  askAdvisor,
  type AdvisoryPayload,
  type AdvisorReplyPayload,
  type DiagnosisPreviewResponse,
} from "@/lib/api";

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  source?: string;
  model?: string | null;
};

type AdvisorContextImage = {
  title: string;
  url: string;
  source: string;
};

type ModelMode = "fast" | "thinking";

type PlantAdvisorChatProps = {
  currentResult?: DiagnosisPreviewResponse | null;
  title?: string;
  compact?: boolean;
};

function toTitleCase(value: string) {
  return value
    .replace(/__/g, " ")
    .replace(/_/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function buildCurrentDiagnosisContext(result?: DiagnosisPreviewResponse | null) {
  if (!result) {
    return null;
  }

  const advisory = result.meta?.advisory as AdvisoryPayload | undefined;
  const analysis = result.data.raw_prediction_json ?? {};
  const input = typeof analysis.input === "object" && analysis.input !== null ? analysis.input : {};
  const summary = typeof analysis.summary === "object" && analysis.summary !== null ? analysis.summary : {};
  const postprocessing =
    typeof analysis.postprocessing === "object" && analysis.postprocessing !== null ? analysis.postprocessing : {};
  const modelIntegration =
    typeof analysis.model_integration === "object" && analysis.model_integration !== null ? analysis.model_integration : {};
  const outputs = Array.isArray(analysis.outputs) ? analysis.outputs : [];
  const analysisImages = outputs
    .map((item) => {
      if (!item || typeof item !== "object") {
        return null;
      }
      const record = item as Record<string, unknown>;
      const url = typeof record.image_url === "string" ? record.image_url : null;
      if (!url) {
        return null;
      }
      const title =
        (typeof record.title === "string" && record.title) ||
        (typeof record.label === "string" && toTitleCase(record.label)) ||
        "Analysis image";
      return { title, url };
    })
    .filter(Boolean)
    .slice(0, 3);

  return {
    image_url: typeof input.stored_input_url === "string" ? input.stored_input_url : null,
    predicted_crop: result.data.predicted_crop,
    predicted_disease: result.data.predicted_disease ?? null,
    health_status: result.data.health_status,
    confidence_score: result.data.confidence_score,
    severity_level: result.data.severity_level,
    urgency_level: result.data.urgency_level,
    model_version: result.data.model_version,
    cv_summary: summary,
    postprocessing,
    model_integration: modelIntegration,
    analysis_images: analysisImages,
    advisory: advisory
      ? {
          what_happened: advisory.what_happened,
          likely_cause: advisory.likely_cause,
          severity_summary: advisory.severity_summary,
          treatment_steps: advisory.treatment_steps,
          urgency_guidance: advisory.urgency_guidance,
          follow_up_recommendation: advisory.follow_up_recommendation,
        }
      : null,
  };
}

function buildLocalContextImages(currentDiagnosis: Record<string, unknown> | null): AdvisorContextImage[] {
  if (!currentDiagnosis) {
    return [];
  }

  const images: AdvisorContextImage[] = [];
  const imageUrl = currentDiagnosis.image_url;
  if (typeof imageUrl === "string" && imageUrl) {
    images.push({ title: "Current scan", url: imageUrl, source: "current_scan" });
  }

  const analysisImages = currentDiagnosis.analysis_images;
  if (Array.isArray(analysisImages)) {
    for (const item of analysisImages) {
      if (!item || typeof item !== "object") {
        continue;
      }
      const record = item as Record<string, unknown>;
      if (typeof record.url === "string" && record.url) {
        images.push({
          title: typeof record.title === "string" ? record.title : "Analysis image",
          url: record.url,
          source: "current_scan_analysis",
        });
      }
    }
  }

  return images;
}

export function PlantAdvisorChat({ currentResult, title = "Plant Expert Chat", compact = false }: PlantAdvisorChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [modelMode, setModelMode] = useState<ModelMode>("fast");
  const [loading, setLoading] = useState(false);
  const [lastContextSummary, setLastContextSummary] = useState("");
  const [lastContextImages, setLastContextImages] = useState<AdvisorContextImage[]>([]);

  const { token } = useAuth();
  const { toast } = useToast();
  const currentDiagnosis = useMemo(() => buildCurrentDiagnosisContext(currentResult), [currentResult]);
  const localContextImages = useMemo(() => buildLocalContextImages(currentDiagnosis), [currentDiagnosis]);
  const displayedImages = lastContextImages.length > 0 ? lastContextImages : localContextImages;

  const promptButtons = currentDiagnosis
    ? [
        { label: "Current Scan", icon: Leaf, prompt: "Explain this current scan like a plant expert. What does it mean?" },
        { label: "Next Steps", icon: Sparkles, prompt: "What should I do next for this plant, step by step?" },
        { label: "Use History", icon: History, prompt: "Compare this scan with my saved plant history and tell me what matters." },
      ]
    : [
        { label: "Latest Result", icon: Leaf, prompt: "Explain my latest saved diagnosis like a plant expert." },
        { label: "Next Steps", icon: Sparkles, prompt: "What should I do next based on my saved plant history?" },
        { label: "Review History", icon: History, prompt: "Review my plant history and tell me what I should watch." },
      ];

  async function sendMessage(messageText: string) {
    const text = messageText.trim();
    if (!text || loading) {
      return;
    }
    if (!token) {
      toast({
        title: "Sign in required",
        description: "Please sign in before asking the plant expert.",
        variant: "destructive",
      });
      return;
    }

    setDraft("");
    setLoading(true);
    setMessages((current) => [...current, { role: "user", text }]);

    try {
      const response = await askAdvisor(token, text, {
        context_mode: currentDiagnosis ? "current_scan" : "account",
        model_mode: modelMode,
        current_diagnosis: currentDiagnosis,
      });
      const reply: AdvisorReplyPayload = response.data;
      setLastContextSummary(reply.context_summary);
      setLastContextImages(reply.context_images ?? []);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: reply.reply,
          source: reply.source,
          model: reply.llm_model,
        },
      ]);
    } catch (caughtError) {
      toast({
        title: "Advisor unavailable",
        description: caughtError instanceof Error ? caughtError.message : "Could not get a plant expert reply.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="bg-card rounded-2xl border border-border p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
        <div>
          <h3 className="font-semibold text-foreground flex items-center gap-2">
            <Bot className="h-5 w-5 text-primary" /> {title}
          </h3>
          <p className="text-sm text-muted-foreground mt-1">
            {currentDiagnosis ? "Using current scan and saved history." : "Using saved plants and diagnosis history."}
          </p>
        </div>
        {lastContextSummary ? (
          <span className="inline-flex rounded-full border border-primary/20 bg-primary/5 px-3 py-1 text-xs font-medium text-primary">
            {lastContextSummary}
          </span>
        ) : null}
      </div>

      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant={modelMode === "fast" ? "default" : "outline"}
          size="sm"
          className={`rounded-xl ${modelMode === "fast" ? "gradient-hero border-0 text-primary-foreground" : ""}`}
          disabled={loading}
          onClick={() => setModelMode("fast")}
        >
          <Zap className="mr-2 h-4 w-4" /> Fast 3B
        </Button>
        <Button
          type="button"
          variant={modelMode === "thinking" ? "default" : "outline"}
          size="sm"
          className={`rounded-xl ${modelMode === "thinking" ? "gradient-hero border-0 text-primary-foreground" : ""}`}
          disabled={loading}
          onClick={() => setModelMode("thinking")}
        >
          <Brain className="mr-2 h-4 w-4" /> Thinking 8B
        </Button>
      </div>

      <div className="flex flex-wrap gap-2">
        {promptButtons.map((item) => (
          <Button
            key={item.label}
            type="button"
            variant="outline"
            size="sm"
            className="rounded-xl"
            disabled={loading}
            onClick={() => void sendMessage(item.prompt)}
          >
            <item.icon className="mr-2 h-4 w-4" /> {item.label}
          </Button>
        ))}
      </div>

      {displayedImages.length > 0 ? (
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Images available to this chat</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {displayedImages.slice(0, 8).map((image) => (
              <a
                key={`${image.source}-${image.url}`}
                href={image.url}
                target="_blank"
                rel="noreferrer"
                className="group block overflow-hidden rounded-xl border border-border bg-muted/20"
              >
                <img src={image.url} alt={image.title} className="h-24 w-full object-cover transition-transform group-hover:scale-105" />
                <p className="truncate px-2 py-1.5 text-xs font-medium text-foreground">{image.title}</p>
              </a>
            ))}
          </div>
        </div>
      ) : null}

      <div className={`rounded-xl border border-border bg-muted/20 p-3 space-y-3 ${compact ? "max-h-72" : "max-h-96"} overflow-y-auto`}>
        {messages.length === 0 ? (
          <div className="text-sm text-muted-foreground py-4 text-center">Ask about treatment, severity, causes, or what to check next.</div>
        ) : (
          messages.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm ${
                  message.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-background border border-border text-foreground"
                }`}
              >
                <p className="whitespace-pre-wrap leading-relaxed">{message.text}</p>
                {message.role === "assistant" && message.source ? (
                  <p className="text-xs text-muted-foreground mt-2">
                    {toTitleCase(message.source)}
                    {message.model ? ` - ${message.model}` : ""}
                  </p>
                ) : null}
              </div>
            </div>
          ))
        )}
        {loading ? <p className="text-sm text-muted-foreground px-2">Plant expert is thinking...</p> : null}
      </div>

      <form
        className="flex flex-col sm:flex-row gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          void sendMessage(draft);
        }}
      >
        <Textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Ask about this plant..."
          className="min-h-12 rounded-xl resize-none"
        />
        <Button type="submit" className="rounded-xl gradient-hero border-0 text-primary-foreground sm:self-end" disabled={loading || !draft.trim()}>
          <Send className="mr-2 h-4 w-4" /> Ask
        </Button>
      </form>
    </section>
  );
}
