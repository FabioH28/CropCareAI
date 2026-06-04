import { Link } from "react-router-dom";
import { motion } from "framer-motion";

import { Button } from "@/components/ui/button";
import { ScanLine, Leaf, History, User, ArrowRight } from "lucide-react";
import heroImg from "@/assets/hero-farm.jpg";
import heroBgImg from "@/assets/hero-main-bg.png";
import leafIcon from "@/assets/leaf-ai-icon.png";

const features = [
  {
    icon: ScanLine,
    title: "Live Diagnosis",
    desc: "Upload a leaf photo and run the real computer-vision and CNN pipeline from the backend.",
  },
  {
    icon: History,
    title: "Saved History",
    desc: "Keep your past scans in one place so you can compare results and track changes over time.",
  },
  {
    icon: Leaf,
    title: "Focused Dashboard",
    desc: "See the diagnosis results that matter without extra management screens getting in the way.",
  },
  {
    icon: User,
    title: "Simple Profile",
    desc: "Manage your account details and password from one straightforward profile page.",
  },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <nav className="fixed top-0 w-full bg-background/80 backdrop-blur-md border-b border-border z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-2">
            <img src={leafIcon} alt="CropCare AI" className="w-8 h-8 rounded-lg" />
            <span className="text-lg font-bold text-foreground">CropCare AI</span>
          </Link>
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Features</a>
            <a href="#how-it-works" className="text-sm text-muted-foreground hover:text-foreground transition-colors">How It Works</a>
          </div>
          <div className="flex items-center gap-3">
            <Link to="/login">
              <Button variant="ghost" className="text-sm">Sign In</Button>
            </Link>
            <Link to="/signup">
              <Button className="text-sm rounded-xl gradient-hero border-0 text-primary-foreground">Get Started</Button>
            </Link>
          </div>
        </div>
      </nav>

      <section className="relative min-h-screen pt-16 px-4 overflow-hidden flex items-center">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${heroBgImg})` }}
        />
        <div className="absolute inset-0 bg-background/15" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.72)_0%,rgba(255,255,255,0.45)_34%,rgba(255,255,255,0.10)_68%)]" />
        <div className="max-w-7xl mx-auto relative w-full">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="text-center max-w-3xl mx-auto"
          >
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-foreground leading-tight mb-6">
              Protect Your Crops with{" "}
              <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                Intelligent Care
              </span>
            </h1>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
              CropCare AI is now a focused plant-diagnosis app: dashboard, live diagnosis, saved history, and your profile.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/login">
                <Button size="lg" className="rounded-xl gradient-hero border-0 text-primary-foreground px-8 text-base h-12">
                  Open CropCare <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link to="/signup">
                <Button size="lg" variant="outline" className="rounded-xl px-8 text-base h-12">
                  Create Account
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      <section id="features" className="py-16 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-[minmax(0,1.05fr)_minmax(320px,0.95fr)] gap-10 lg:gap-14 items-center mb-14">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
              viewport={{ once: true }}
              className="relative"
            >
              <div className="rounded-2xl overflow-hidden shadow-2xl shadow-primary/10 border border-border">
                <img src={heroImg} alt="CropCare AI dashboard showing farm analysis" width={1920} height={1080} className="w-full" />
              </div>
            </motion.div>
            <div className="text-center lg:text-left">
              <h2 className="text-3xl lg:text-4xl font-bold text-foreground mb-4">A Simpler CropCare Experience</h2>
              <p className="text-muted-foreground max-w-xl mx-auto lg:mx-0 text-lg">
                The app is intentionally trimmed to the parts that matter most during diagnosis and presentation.
              </p>
            </div>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-card rounded-2xl border border-border p-6 hover:shadow-lg hover:-translate-y-1 transition-all duration-300"
              >
                <div className="p-3 rounded-xl bg-primary/10 w-fit mb-4">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="font-semibold text-foreground mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section id="how-it-works" className="py-20 px-4 bg-card border-y border-border">
        <div className="max-w-5xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-foreground mb-12">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { step: "1", title: "Open Dashboard", desc: "Sign in and land on a clear overview of saved scans and current plant-health status." },
              { step: "2", title: "Run Diagnosis", desc: "Upload a plant image and let the backend run the CV and deep-learning pipeline." },
              { step: "3", title: "Save To History", desc: "Store the result in SQL so you can review it later from the history page." },
            ].map((step, index) => (
              <motion.div
                key={step.step}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.15 }}
                viewport={{ once: true }}
              >
                <div className="w-12 h-12 rounded-full gradient-hero text-primary-foreground flex items-center justify-center text-lg font-bold mx-auto mb-4">
                  {step.step}
                </div>
                <h3 className="font-semibold text-foreground mb-2">{step.title}</h3>
                <p className="text-sm text-muted-foreground">{step.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <footer className="border-t border-border py-10 px-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <img src={leafIcon} alt="CropCare AI" className="w-6 h-6 rounded" />
            <span className="font-semibold text-foreground">CropCare AI</span>
          </div>
          <p className="text-sm text-muted-foreground">© 2026 CropCare AI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
