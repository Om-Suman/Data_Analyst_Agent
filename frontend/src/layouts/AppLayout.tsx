import React, { useState, useEffect, useRef } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Sidebar } from "../components/Sidebar";
import { Navbar } from "../components/Navbar";
import { AICopilotDrawer } from "../components/AICopilotDrawer";

export const AppLayout: React.FC = () => {
  const [isCopilotOpen, setIsCopilotOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const mainRef = useRef<HTMLElement>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const menuRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0, left: 0, behavior: "instant" });
    window.scrollTo({ top: 0, left: 0, behavior: "instant" });
    setMobileOpen(false);
  }, [location.pathname]);
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!mobileOpen) {
      dialog?.close();
      return;
    }
    dialog?.showModal();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const breakpoint = window.matchMedia("(min-width: 1024px)");
    const closeOnDesktop = () => {
      if (breakpoint.matches) setMobileOpen(false);
    };
    breakpoint.addEventListener("change", closeOnDesktop);
    return () => {
      document.body.style.overflow = previousOverflow;
      breakpoint.removeEventListener("change", closeOnDesktop);
      dialog?.close();
      menuRef.current?.focus();
    };
  }, [mobileOpen]);
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setMobileOpen(false);
        setIsCopilotOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);
  return (
    <div className="flex min-h-screen bg-[var(--bg-app)] text-[var(--text-main)]">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:bg-white focus:text-blue-700 focus:p-3"
      >
        Skip to content
      </a>
      <div className="hidden lg:block">
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        />
      </div>
      <dialog
        id="mobile-navigation"
        ref={dialogRef}
        aria-label="Navigation"
        onCancel={() => setMobileOpen(false)}
        onClose={() => setMobileOpen(false)}
        onClick={(e) => {
          if (e.target === e.currentTarget) setMobileOpen(false);
        }}
        className="fixed inset-y-0 left-0 right-auto m-0 h-dvh max-h-none w-80 max-w-[90vw] p-0 bg-white dark:bg-[#0d1322] text-[var(--text-main)] backdrop:bg-slate-950/50"
      >
        <Sidebar mobile onNavigate={() => setMobileOpen(false)} />
      </dialog>
      <div className="flex-1 flex flex-col min-w-0">
        <Navbar
          onToggleCopilot={() => setIsCopilotOpen(!isCopilotOpen)}
          isCopilotOpen={isCopilotOpen}
          onOpenNavigation={() => setMobileOpen(true)}
          menuRef={menuRef}
          mobileOpen={mobileOpen}
        />
        <main
          id="main-content"
          tabIndex={-1}
          ref={mainRef}
          className="flex-1 min-w-0 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto"
        >
          <Outlet />
        </main>
      </div>
      <AICopilotDrawer
        isOpen={isCopilotOpen}
        onClose={() => setIsCopilotOpen(false)}
      />
    </div>
  );
};
