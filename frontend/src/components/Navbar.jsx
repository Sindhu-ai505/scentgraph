import React from "react";
import { Atom, BarChart2, Database, Activity, HelpCircle, Sparkles } from "lucide-react";

export default function Navbar({ activeTab, setActiveTab, onOpenHowItWorks, activeModel, setActiveModel }) {
  const tabs = [
    { id: "predict", label: "Predictor", icon: Atom },
    { id: "compare", label: "Model Comparison", icon: BarChart2 },
    { id: "dataset", label: "Dataset Explorer", icon: Database },
    { id: "training", label: "Training Curves", icon: Activity },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/10 glass-panel">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo */}
        <div 
          className="flex items-center space-x-3 cursor-pointer select-none"
          onClick={() => setActiveTab("predict")}
        >
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-sky-500 to-indigo-600 shadow-glow-cyan">
            <Atom className="w-6 h-6 text-white animate-spin-slow" />
            <span className="absolute -top-1 -right-1 flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
            </span>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-100 to-cyan-300 bg-clip-text text-transparent">
                ScentGraph
              </span>
              <span className="text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                GNN AI
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium">Molecule-to-Odor Predictor</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="hidden md:flex items-center space-x-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-white/10 text-cyan-300 shadow-sm border border-white/10"
                    : "text-slate-400 hover:text-slate-200 hover:bg-white/5"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-cyan-400" : "text-slate-400"}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Action Items */}
        <div className="flex items-center space-x-3">
          {/* Active Model Selector */}
          <div className="hidden sm:flex items-center bg-slate-900/80 border border-white/10 rounded-lg p-0.5 text-xs">
            <button
              onClick={() => setActiveModel("gatv2")}
              className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                activeModel === "gatv2"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              GATv2 (Attn)
            </button>
            <button
              onClick={() => setActiveModel("gcn")}
              className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                activeModel === "gcn"
                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              GCN Baseline
            </button>
          </div>

          {/* How It Works Explainer */}
          <button
            onClick={onOpenHowItWorks}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 hover:text-white bg-white/5 hover:bg-white/10 border border-white/10 transition-colors"
          >
            <HelpCircle className="w-3.5 h-3.5 text-cyan-400" />
            <span>How It Works</span>
          </button>
        </div>
      </div>

      {/* Mobile Navigation Sub-bar */}
      <div className="md:hidden flex items-center justify-around border-t border-white/5 px-2 py-1.5 bg-slate-950/60 overflow-x-auto">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex flex-col items-center py-1 px-2.5 rounded text-[11px] font-medium transition-colors ${
                isActive ? "text-cyan-300" : "text-slate-400"
              }`}
            >
              <Icon className="w-4 h-4 mb-0.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>
    </header>
  );
}
