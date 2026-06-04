import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Leaf, Eye, EyeOff } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/lib/auth";

export default function SignupPage() {
  const [showPw, setShowPw] = useState(false);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [farmName, setFarmName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const navigate = useNavigate();
  const { toast } = useToast();
  const { authReady, isAuthenticated, signUp, signOut, user } = useAuth();

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

    const fullName = `${firstName} ${lastName}`.trim();
    if (fullName.length < 2) {
      const nextError = "Please enter your first and last name.";
      setError(nextError);
      toast({
        title: "Sign-up failed",
        description: nextError,
        variant: "destructive",
      });
      setSubmitting(false);
      return;
    }
    if (password.trim().length < 8) {
      const nextError = "Password must be at least 8 characters.";
      setError(nextError);
      toast({
        title: "Sign-up failed",
        description: nextError,
        variant: "destructive",
      });
      setSubmitting(false);
      return;
    }

    try {
      await signUp({
        email,
        password,
        fullName,
        farmName,
      });
    } catch (exc) {
      const nextError = exc instanceof Error ? exc.message : "Sign up failed. Please try again.";
      setError(nextError);
      toast({
        title: "Sign-up failed",
        description: nextError,
        variant: "destructive",
      });
      setSubmitting(false);
      return;
    }

    toast({
      title: "Account created",
      description: "Your backend account is ready and saved in the project database.",
    });
    navigate("/dashboard", { replace: true });
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
          <h2 className="text-3xl font-bold text-primary-foreground mb-4">Join CropCare AI</h2>
          <p className="text-primary-foreground/70">
            Start protecting your crops today with AI-powered disease detection and expert treatment advice.
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
              <span className="text-xl font-bold text-foreground">CropCare AI</span>
            </Link>
            <h1 className="text-2xl font-bold text-foreground">Create your account</h1>
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
                  onClick={() => navigate("/dashboard", { replace: true })}
                >
                  Continue
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  className="flex-1 rounded-xl"
                  onClick={() => void signOut()}
                >
                  Create Another Account
                </Button>
              </div>
            </div>
          ) : (
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label>First Name</Label>
                <Input
                  placeholder="Your name"
                  className="rounded-xl h-11"
                  value={firstName}
                  onChange={(event) => setFirstName(event.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>Last Name</Label>
                <Input
                  placeholder="Your last name"
                  className="rounded-xl h-11"
                  value={lastName}
                  onChange={(event) => setLastName(event.target.value)}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Email</Label>
              <Input
                type="email"
                placeholder="farmer@example.com"
                className="rounded-xl h-11"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Farm Name</Label>
              <Input
                placeholder="Green Valley Farm"
                className="rounded-xl h-11"
                value={farmName}
                onChange={(event) => setFarmName(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Password</Label>
              <div className="relative">
                <Input
                  type={showPw ? "text" : "password"}
                  placeholder="At least 8 characters"
                  className="rounded-xl h-11 pr-10"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
                <button
                  type="button"
                  onClick={() => setShowPw(!showPw)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground"
                >
                  {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>
            <div className="rounded-xl border border-primary/20 bg-primary/5 px-4 py-3 text-sm">
              <p className="font-medium text-foreground">Local backend account</p>
              <p className="text-muted-foreground mt-1">
                This sign-up flow now creates a real backend account in the project database. Plant analysis still uses the real backend model.
              </p>
            </div>
            {error ? <p className="text-sm text-destructive">{error}</p> : null}
            <Button
              type="submit"
              disabled={submitting}
              className="w-full rounded-xl h-11 gradient-hero border-0 text-primary-foreground"
            >
              {submitting ? "Creating account..." : "Create Account"}
            </Button>
          </form>
          )}
          <p className="text-center text-sm text-muted-foreground">
            Already have an account?{" "}
            <Link to="/login" className="text-primary font-medium hover:underline">
              Sign in
            </Link>
          </p>
        </motion.div>
      </div>
    </div>
  );
}
