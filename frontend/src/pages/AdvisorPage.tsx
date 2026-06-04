import { Bot } from "lucide-react";

import { PlantAdvisorChat } from "@/components/advisor/PlantAdvisorChat";
import { PageHeader } from "@/components/shared/PageHeader";

export default function AdvisorPage() {
  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <PageHeader
        title="Plant Advisor"
        description="Ask about your saved diagnoses, plant history, and follow-up tasks."
        actions={
          <div className="hidden sm:flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Bot className="h-5 w-5" />
          </div>
        }
      />
      <PlantAdvisorChat />
    </div>
  );
}
