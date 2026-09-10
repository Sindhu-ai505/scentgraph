import React, { useState } from "react";
import Navbar from "./components/Navbar";
import HowItWorksModal from "./components/HowItWorksModal";
import PredictorScreen from "./pages/PredictorScreen";
import ModelCompareScreen from "./pages/ModelCompareScreen";
import DatasetExplorerScreen from "./pages/DatasetExplorerScreen";
import TrainingDashboardScreen from "./pages/TrainingDashboardScreen";
import { Atom, ExternalLink, Heart } from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const [activeModel, setActiveModel] = useState("gatv2");
  const [howItWorksOpen, setHowItWorksOpen] = useState(false);

  return (
    <div className="min-h-screen flex flex-col bg-[#070a12] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenHowItWorks={() => setHowItWorksOpen(true)}
        activeModel={activeModel}
        setActiveModel={setActiveModel}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        {activeTab === "predict" && (
          <PredictorScreen activeModel={activeModel} setActiveModel={setActiveModel} />
        )}
        {activeTab === "compare" && <ModelCompareScreen />}
        {activeTab === "dataset" && <DatasetExplorerScreen />}
        {activeTab === "training" && <TrainingDashboardScreen />}
      </main>

      {/* Educational Modal */}
      <HowItWorksModal
        isOpen={howItWorksOpen}
        onClose={() => setHowItWorksOpen(false)}
      />

      {/* Footer */}
      <footer className="w-full border-t border-white/10 glass-panel py-6 text-xs text-slate-400 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2">
            <Atom className="w-4 h-4 text-cyan-400" />
            <span className="font-semibold text-slate-300">ScentGraph</span>
            <span>—</span>
            <span>Graph Neural Network Olfactory Predictor</span>
          </div>

          <div className="flex items-center space-x-4 text-slate-400">
            <span>Pyrfume-data</span>
            <span>•</span>
            <span>RDKit</span>
            <span>•</span>
            <span>PyTorch Geometric</span>
            <span>•</span>
            <span>FastAPI</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
