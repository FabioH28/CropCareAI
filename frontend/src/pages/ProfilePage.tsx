import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Camera, MapPin, Shield } from "lucide-react";

import { PageHeader } from "@/components/shared/PageHeader";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/lib/auth";
import { changePassword, fetchCurrentProfile, updateProfile, type ProfilePayload } from "@/lib/api";

type ProfileFormState = {
  full_name: string;
  email: string;
  farm_name: string;
  phone: string;
  location: string;
  avatar_url: string;
};

type PasswordFormState = {
  current_password: string;
  new_password: string;
  confirm_password: string;
};

const initialPasswordForm: PasswordFormState = {
  current_password: "",
  new_password: "",
  confirm_password: "",
};

function buildProfileForm(profile: ProfilePayload): ProfileFormState {
  return {
    full_name: profile.full_name ?? "",
    email: profile.email,
    farm_name: profile.farm_name ?? "",
    phone: profile.phone ?? "",
    location: profile.location ?? "",
    avatar_url: profile.avatar_url ?? "",
  };
}

export default function ProfilePage() {
  const [profile, setProfile] = useState<ProfilePayload | null>(null);
  const [form, setForm] = useState<ProfileFormState | null>(null);
  const [passwordForm, setPasswordForm] = useState<PasswordFormState>(initialPasswordForm);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [savingPassword, setSavingPassword] = useState(false);

  const { token } = useAuth();
  const { toast } = useToast();

  useEffect(() => {
    let cancelled = false;

    async function loadProfile() {
      if (!token) {
        return;
      }

      setLoading(true);
      setError("");
      try {
        const response = await fetchCurrentProfile(token);
        if (!cancelled) {
          setProfile(response.data);
          setForm(buildProfileForm(response.data));
        }
      } catch (caughtError) {
        if (!cancelled) {
          setError(caughtError instanceof Error ? caughtError.message : "Could not load your profile.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadProfile();
    return () => {
      cancelled = true;
    };
  }, [token]);

  const initials = useMemo(() => {
    const source = form?.full_name || profile?.email || "FA";
    return source
      .split(" ")
      .map((part) => part[0] ?? "")
      .join("")
      .slice(0, 2)
      .toUpperCase();
  }, [form?.full_name, profile?.email]);

  const handleSaveProfile = async () => {
    if (!token || !form) {
      return;
    }

    setSavingProfile(true);
    try {
      const response = await updateProfile(token, {
        full_name: form.full_name.trim() || null,
        farm_name: form.farm_name.trim() || null,
        phone: form.phone.trim() || null,
        location: form.location.trim() || null,
        avatar_url: form.avatar_url.trim() || null,
      });
      setProfile(response.data);
      setForm(buildProfileForm(response.data));
      toast({ title: "Profile updated", description: "Your account details were saved to the backend." });
    } catch (caughtError) {
      toast({
        title: "Profile update failed",
        description: caughtError instanceof Error ? caughtError.message : "Could not save your profile.",
        variant: "destructive",
      });
    } finally {
      setSavingProfile(false);
    }
  };

  const handleChangePassword = async () => {
    if (!token) {
      return;
    }

    if (!passwordForm.current_password || !passwordForm.new_password || !passwordForm.confirm_password) {
      toast({
        title: "Missing password fields",
        description: "Fill in the current password and the new password fields.",
        variant: "destructive",
      });
      return;
    }
    if (passwordForm.new_password !== passwordForm.confirm_password) {
      toast({
        title: "Passwords do not match",
        description: "The confirmation password must match the new password.",
        variant: "destructive",
      });
      return;
    }

    setSavingPassword(true);
    try {
      await changePassword(token, {
        current_password: passwordForm.current_password,
        new_password: passwordForm.new_password,
      });
      setPasswordForm(initialPasswordForm);
      toast({ title: "Password updated", description: "Your account password was updated successfully." });
    } catch (caughtError) {
      toast({
        title: "Password update failed",
        description: caughtError instanceof Error ? caughtError.message : "Could not update your password.",
        variant: "destructive",
      });
    } finally {
      setSavingPassword(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-3xl mx-auto space-y-6">
        <PageHeader title="Profile" description="Loading your account..." />
        <div className="bg-card rounded-xl border border-border p-8 text-center text-muted-foreground">
          Loading profile data...
        </div>
      </div>
    );
  }

  if (error || !form) {
    return (
      <div className="max-w-3xl mx-auto space-y-6">
        <PageHeader title="Profile" description="Manage your personal and farm information." />
        <div className="rounded-xl border border-destructive/20 bg-destructive/5 p-5 text-sm text-destructive">
          {error || "Could not load the profile."}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <PageHeader title="Profile" description="Manage the account details stored in your project backend database." />

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-card rounded-xl border border-border p-6">
        <div className="flex flex-col sm:flex-row items-start gap-6">
          <div className="relative">
            <Avatar className="h-20 w-20 ring-4 ring-primary/10">
              <AvatarFallback className="bg-primary text-primary-foreground text-2xl font-bold">{initials}</AvatarFallback>
            </Avatar>
            <div className="absolute bottom-0 right-0 p-1.5 rounded-full bg-primary text-primary-foreground shadow-lg">
              <Camera className="h-3.5 w-3.5" />
            </div>
          </div>
          <div className="flex-1 space-y-4 w-full">
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="full_name">Full name</Label>
                <Input
                  id="full_name"
                  value={form.full_name}
                  onChange={(event) => setForm((current) => (current ? { ...current, full_name: event.target.value } : current))}
                  className="rounded-xl"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input id="email" value={form.email} readOnly className="rounded-xl bg-muted/40" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="phone">Phone</Label>
                <Input
                  id="phone"
                  value={form.phone}
                  onChange={(event) => setForm((current) => (current ? { ...current, phone: event.target.value } : current))}
                  className="rounded-xl"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="avatar_url">Avatar URL</Label>
                <Input
                  id="avatar_url"
                  value={form.avatar_url}
                  onChange={(event) => setForm((current) => (current ? { ...current, avatar_url: event.target.value } : current))}
                  className="rounded-xl"
                  placeholder="Optional image URL"
                />
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="bg-card rounded-xl border border-border p-6">
        <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
          <MapPin className="h-4 w-4 text-primary" /> Farm Information
        </h3>
        <div className="grid sm:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="farm_name">Farm name</Label>
            <Input
              id="farm_name"
              value={form.farm_name}
              onChange={(event) => setForm((current) => (current ? { ...current, farm_name: event.target.value } : current))}
              className="rounded-xl"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="location">Location</Label>
            <Input
              id="location"
              value={form.location}
              onChange={(event) => setForm((current) => (current ? { ...current, location: event.target.value } : current))}
              className="rounded-xl"
            />
          </div>
        </div>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="bg-card rounded-xl border border-border p-6">
        <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
          <Shield className="h-4 w-4 text-primary" /> Security
        </h3>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="current_password">Current password</Label>
            <Input
              id="current_password"
              type="password"
              value={passwordForm.current_password}
              onChange={(event) => setPasswordForm((current) => ({ ...current, current_password: event.target.value }))}
              className="rounded-xl"
            />
          </div>
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="new_password">New password</Label>
              <Input
                id="new_password"
                type="password"
                value={passwordForm.new_password}
                onChange={(event) => setPasswordForm((current) => ({ ...current, new_password: event.target.value }))}
                className="rounded-xl"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="confirm_password">Confirm password</Label>
              <Input
                id="confirm_password"
                type="password"
                value={passwordForm.confirm_password}
                onChange={(event) => setPasswordForm((current) => ({ ...current, confirm_password: event.target.value }))}
                className="rounded-xl"
              />
            </div>
          </div>
          <div className="flex justify-end">
            <Button variant="outline" className="rounded-xl" onClick={() => void handleChangePassword()} disabled={savingPassword}>
              {savingPassword ? "Updating..." : "Update Password"}
            </Button>
          </div>
        </div>
      </motion.div>

      <div className="flex justify-end">
        <Button className="rounded-xl gradient-hero border-0 text-primary-foreground px-8" onClick={() => void handleSaveProfile()} disabled={savingProfile}>
          {savingProfile ? "Saving..." : "Save Changes"}
        </Button>
      </div>
    </div>
  );
}
