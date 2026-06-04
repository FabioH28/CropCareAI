import { useMemo } from "react";
import { Menu } from "lucide-react";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { useLocation, useNavigate } from "react-router-dom";

import { useAuth } from "@/lib/auth";

const pageTitles: Record<string, string> = {
  "/dashboard": "Dashboard",
  "/diagnose": "Diagnose Plant",
  "/history": "Detection History",
  "/profile": "Profile",
};

export function TopNav() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();

  const title = useMemo(() => {
    return pageTitles[location.pathname] || "CropCare AI";
  }, [location.pathname]);

  const initials =
    user?.fullName
      .split(" ")
      .map((part) => part[0])
      .join("")
      .slice(0, 2)
      .toUpperCase() || "FA";

  return (
    <header className="h-16 border-b border-border bg-card/80 backdrop-blur-sm flex items-center justify-between px-4 md:px-6 sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <SidebarTrigger className="text-muted-foreground hover:text-foreground transition-colors">
          <Menu className="h-5 w-5" />
        </SidebarTrigger>
        <h1 className="text-lg font-semibold text-foreground hidden sm:block">{title}</h1>
      </div>
      <div className="flex items-center gap-3">
        <button type="button" onClick={() => navigate("/profile")}>
          <Avatar className="h-9 w-9 cursor-pointer ring-2 ring-primary/20">
            <AvatarFallback className="bg-primary text-primary-foreground text-sm font-semibold">{initials}</AvatarFallback>
          </Avatar>
        </button>
      </div>
    </header>
  );
}
