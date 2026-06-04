import { Outlet } from "react-router-dom";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "./AppSidebar";
import { TopNav } from "./TopNav";
import { MobileNav } from "./MobileNav";

export function AppLayout() {
  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full">
        <AppSidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <TopNav />
          <main className="flex-1 p-4 md:p-6 pb-20 md:pb-6 overflow-auto">
            <Outlet />
          </main>
          <MobileNav />
        </div>
      </div>
    </SidebarProvider>
  );
}
