import { useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, Leaf } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { requestPasswordReset } from "@/lib/api";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const handleSubmit = async () => {
    if (!email.trim()) {
      setErrorMessage("Please enter the email address for your account.");
      return;
    }

    setSubmitting(true);
    setErrorMessage("");
    try {
      const response = await requestPasswordReset(email.trim().toLowerCase());
      setSuccessMessage(response.data.note || response.message);
    } catch (caughtError) {
      setErrorMessage(caughtError instanceof Error ? caughtError.message : "Could not submit the reset request.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-background">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-sm space-y-6">
        <div className="text-center">
          <div className="p-3 rounded-2xl bg-primary/10 w-fit mx-auto mb-4">
            <Leaf className="h-8 w-8 text-primary" />
          </div>
          <h1 className="text-2xl font-bold text-foreground">Reset Password</h1>
          <p className="text-sm text-muted-foreground mt-1">
            This local project records reset requests in the database instead of sending real emails.
          </p>
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              placeholder="farmer@example.com"
              className="rounded-xl h-11"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
          </div>
          {successMessage ? (
            <div className="rounded-xl border border-primary/20 bg-primary/5 p-4 text-sm text-muted-foreground">
              {successMessage}
            </div>
          ) : null}
          {errorMessage ? (
            <div className="rounded-xl border border-destructive/20 bg-destructive/5 p-4 text-sm text-destructive">
              {errorMessage}
            </div>
          ) : null}
          <Button
            className="w-full rounded-xl h-11 gradient-hero border-0 text-primary-foreground"
            onClick={() => void handleSubmit()}
            disabled={submitting}
          >
            {submitting ? "Submitting..." : "Send Reset Link"}
          </Button>
        </div>

        <Link to="/login" className="flex items-center justify-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
          <ArrowLeft className="h-4 w-4" /> Back to sign in
        </Link>
      </motion.div>
    </div>
  );
}
