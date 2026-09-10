import React from "react";
import { X, Network, Cpu, Sparkles, BookOpen, Layers, CheckCircle2 } from "lucide-react";

export default function HowItWorksModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in"
      onClick={onClose}
    >
      <div 
        className="relative w-full max-w-3xl max-h-[90vh] overflow-y-auto glass-panel rounded-2xl border border-white/15 p-6 sm:p-8 text-slate-200 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between pb-4 border-b border-white/10">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-cyan-500/20 border border-cyan-500/30 text-cyan-300">
              <Network className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">How ScentGraph Works</h2>
              <p className="text-xs text-slate-400">The science of Graph Neural Networks and molecular olfaction</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="mt-6 space-y-6 text-sm leading-relaxed">
          {/* Section 1 */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/5 space-y-2">
            <div className="flex items-center space-x-2 text-cyan-300 font-semibold">
              <Cpu className="w-4 h-4" />
              <span>1. Molecules as Chemical Graphs</span>
            </div>
            <p className="text-slate-300 text-xs sm:text-sm">
              In organic chemistry, a molecule is not a flat string or a standard image — it is a mathematical graph.
              Each <strong>atom is a node</strong> enriched with 38 physical features (atomic element, hybridization, valence degree, formal charge, aromaticity, and ring membership).
              Each <strong>chemical bond is an edge</strong> encoding single, double, triple, or aromatic character, along with conjugation states.
            </p>
          </div>

          {/* Section 2 */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/5 space-y-2">
            <div className="flex items-center space-x-2 text-purple-300 font-semibold">
              <Layers className="w-4 h-4" />
              <span>2. Graph Attention Message Passing (GATv2)</span>
            </div>
            <p className="text-slate-300 text-xs sm:text-sm">
              Across 4 message-passing rounds, atoms communicate across chemical bonds to learn their neighborhood.
              Our <strong>ScentGATv2 model</strong> uses multi-head attention to learn which chemical bonds transmit the strongest olfactory signals.
              Unlike standard GCNs that average neighbor weights, GATv2 dynamically computes attention coefficients conditioned on bond features.
            </p>
          </div>

          {/* Section 3 */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/5 space-y-2">
            <div className="flex items-center space-x-2 text-rose-300 font-semibold">
              <Sparkles className="w-4 h-4" />
              <span>3. Global Attention Pooling & Olfactory Receptors</span>
            </div>
            <p className="text-slate-300 text-xs sm:text-sm">
              Human noses contain approximately ~400 distinct olfactory receptor types. A single odorant molecule binds to multiple receptors simultaneously,
              producing complex perceptual accords (e.g. Vanillin is both <em>sweet</em>, <em>vanilla</em>, and <em>creamy</em>).
              ScentGraph uses a <strong>learned Global Attention gate</strong> that pinpoints the critical scent-determining functional groups (osmophores) to generate a unified molecular embedding.
            </p>
          </div>

          {/* Section 4 */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/5 space-y-2">
            <div className="flex items-center space-x-2 text-emerald-300 font-semibold">
              <CheckCircle2 className="w-4 h-4" />
              <span>4. Per-Label Decision Thresholds</span>
            </div>
            <p className="text-slate-300 text-xs sm:text-sm">
              Odor labels in nature are severely imbalanced: while common descriptors like "fruity" occur in nearly 40% of molecules, rare scents like "cardamom" occur in fewer than 1%.
              Rather than using a naive, arbitrary 0.5 probability cutoff, ScentGraph performs <strong>per-label F1 threshold optimization</strong> on the validation set, ensuring accurate, high-recall predictions for both common and rare fragrance notes.
            </p>
          </div>

          {/* Section 5 */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/5 space-y-2">
            <div className="flex items-center space-x-2 text-amber-300 font-semibold">
              <BookOpen className="w-4 h-4" />
              <span>5. Real Experimental Training Data</span>
            </div>
            <p className="text-slate-300 text-xs sm:text-sm">
              Trained on <strong>5,070 canonical molecules</strong> across <strong>157 odor descriptors</strong> curated from peer-reviewed scientific repositories in the Pyrfume-data project, including the <em>Leffingwell Odor Dataset</em>, <em>GoodScents Perfumery Database</em>, <em>Flavornet</em>, and <em>AromaDB</em>.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 pt-4 border-t border-white/10 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs tracking-wide transition-all shadow-glow-cyan"
          >
            Got it, explore model
          </button>
        </div>
      </div>
    </div>
  );
}
