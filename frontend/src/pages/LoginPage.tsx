import { FormEvent, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/lib/auth";
import { Leaf, Eye, EyeOff } from "lucide-react";

export default function LoginPage() {
  const [showPw, setShowPw] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const navigate = useNavigate();
  const location = useLocation();
  const { toast } = useToast();
  const { authReady, isAuthenticated, signIn, signOut, user } = useAuth();

  const redirectTo =
    (location.state as { from?: { pathname?: string } } | null)?.from?.pathname ||
    "/dashboard";

  if (!authReady) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="rounded-xl border border-border bg-card px-6 py-4 text-sm text-muted-foreground">
          Checking your session...
        </div>
      </div>
    );
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");
    setSubmitting(true);

    try {
      await signIn(email, password);
    } catch (exc) {
      const nextError = exc instanceof Error ? exc.message : "Login failed. Please try again.";
      setError(nextError);
      toast({
        title: "Sign-in failed",
        description: nextError,
        variant: "destructive",
      });
      setSubmitting(false);
      return;
    }

    toast({
      title: "Signed in",
      description: "Welcome back. Redirecting to your dashboard.",
    });
    navigate(redirectTo, { replace: true });
  };

  return (
    <div className="min-h-screen flex">
      <div className="hidden lg:flex lg:w-1/2 gradient-hero items-center justify-center p-12">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="text-center max-w-md"
        >
          <Leaf className="h-16 w-16 text-primary-foreground/80 mx-auto mb-6" />
          <h2 className="text-3xl font-bold text-primary-foreground mb-4">
            Welcome Back
          </h2>
          <p className="text-primary-foreground/70">
            Monitor your crops, detect diseases early, and keep your farm thriving with AI-powered insights.
          </p>
        </motion.div>
      </div>
      <div className="flex-1 flex items-center justify-center p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-sm space-y-6"
        >
          <div className="text-center lg:text-left">
            <Link to="/" className="inline-flex items-center gap-2 mb-6">
              <Leaf className="h-7 w-7 text-primary" />
              <span className="text-xl font-bold text-foreground">
                CropCare AI
              </span>
            </Link>
            <h1 className="text-2xl font-bold text-foreground">
              Sign in to your account
            </h1>
          </div>

          {isAuthenticated ? (
            <div className="space-y-4 rounded-2xl border border-border bg-card p-5">
              <div>
                <p className="text-sm font-medium text-foreground">You are already signed in.</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Signed in as {user?.email ?? "your account"}.
                </p>
              </div>
              <div className="flex gap-3">
                <Button
                  type="button"
                  className="flex-1 rounded-xl gradient-hero border-0 text-primary-foreground"
                  onClick={() => navigate(redirectTo, { replace: true })}
                >
                  Continue
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  className="flex-1 rounded-xl"
                  onClick={() => void signOut()}
                >
                  Use Another Account
                </Button>
              </div>
            </div>
          ) : (
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div className="space-y-2">
              <Label>Email</Label>
              <Input
                type="text"
                placeholder="farmer@example.com"
                className="rounded-xl h-11"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Password</Label>
                <Link
                  to="/forgot-password"
                  className="text-xs text-primary hover:underline"
                >
                  Forgot password?
                </Link>
              </div>
              <div className="relative">
                <Input
                  type={showPw ? "text" : "password"}
                  placeholder="password"
                  className="rounded-xl h-11 pr-10"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
                <button
                  type="button"
                  onClick={() => setShowPw(!showPw)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground"
                >
                  {showPw ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>
            {error ? <p className="text-sm text-destructive">{error}</p> : null}
            <Button
              type="submit"
              disabled={submitting}
              className="w-full rounded-xl h-11 gradient-hero border-0 text-primary-foreground"
            >
              {submitting ? "Signing in..." : "Sign In"}
            </Button>
          </form>
          )}
          <p className="text-center text-sm text-muted-foreground">
            Don&apos;t have an account?{" "}
            <Link to="/signup" className="text-primary font-medium hover:underline">
              Sign up
            </Link>
          </p>
        </motion.div>
      </div>
    </div>
  );
}
