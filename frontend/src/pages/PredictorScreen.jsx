import React, { useState, useEffect } from "react";
import { 
  Atom, Sparkles, ArrowRight, CheckCircle2, AlertCircle, 
  Copy, Check, RefreshCw, Sliders, Info, Zap, Flame, Droplets, Wind
} from "lucide-react";
import { predictMolecule, getMoleculeExamples } from "../services/api";
import { getScentConfig } from "../utils/scentColors";

export default function PredictorScreen({ activeModel, setActiveModel }) {
  const [smilesInput, setSmilesInput] = useState("O=Cc1ccc(O)c(OC)c1"); // Default: Vanillin
  const [isValidSmiles, setIsValidSmiles] = useState(true);
  const [examples, setExamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [copied, setCopied] = useState(false);
  const [displayFilter, setDisplayFilter] = useState("positive"); // 'positive', 'top20', 'all'

  // Fetch curated example molecules on mount
  useEffect(() => {
    getMoleculeExamples()
      .then((data) => {
        if (data?.molecules) setExamples(data.molecules);
      })
      .catch((err) => console.error("Failed to load examples:", err));
  }, []);

  // Live client-side SMILES validity sanity check
  useEffect(() => {
    if (!smilesInput.trim()) {
      setIsValidSmiles(false);
      return;
    }
    // Check balanced parentheses, brackets, and reasonable length
    let parenCount = 0;
    let bracketCount = 0;
    let valid = true;
    for (const char of smilesInput) {
      if (char === "(") parenCount++;
      if (char === ")") parenCount--;
      if (char === "[") bracketCount++;
      if (char === "]") bracketCount--;
      if (parenCount < 0 || bracketCount < 0) {
        valid = false;
        break;
      }
    }
    if (parenCount !== 0 || bracketCount !== 0) valid = false;
    setIsValidSmiles(valid && smilesInput.trim().length >= 2);
  }, [smilesInput]);

  // Initial prediction for default molecule
  useEffect(() => {
    handlePredict("O=Cc1ccc(O)c(OC)c1", activeModel);
  }, []);

  // When model changes in navbar, re-run prediction if we have a current SMILES
  useEffect(() => {
    if (predictionData && predictionData.smiles) {
      handlePredict(predictionData.smiles, activeModel);
    }
  }, [activeModel]);

  const handlePredict = async (smilesToPredict = smilesInput, model = activeModel) => {
    const s = smilesToPredict.trim();
    if (!s) return;
    setLoading(true);
    setError(null);

    try {
      const data = await predictMolecule(s, model);
      setPredictionData(data);
    } catch (err) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectExample = (mol) => {
    setSmilesInput(mol.smiles);
    handlePredict(mol.smiles, activeModel);
  };

  const copySmiles = () => {
    if (!predictionData) return;
    navigator.clipboard.writeText(predictionData.smiles);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Filter predictions based on user selection
  const filteredPredictions = React.useMemo(() => {
    if (!predictionData?.predictions) return [];
    if (displayFilter === "positive") {
      const positives = predictionData.predictions.filter((p) => p.is_positive);
      // If none met threshold, show at least top 6
      return positives.length > 0 ? positives : predictionData.predictions.slice(0, 6);
    }
    if (displayFilter === "top20") {
      return predictionData.predictions.slice(0, 20);
    }
    return predictionData.predictions;
  }, [predictionData, displayFilter]);

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      {/* Hero Header */}
      <div className="text-center max-w-3xl mx-auto space-y-3 pt-4">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>Graph Neural Network Odor Intelligence</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold text-white tracking-tight leading-tight">
          What does your molecule <span className="bg-gradient-to-r from-cyan-400 via-sky-300 to-rose-400 bg-clip-text text-transparent">smell like?</span>
        </h1>
        <p className="text-slate-400 text-sm sm:text-base max-w-2xl mx-auto">
          Input any SMILES chemical formula to predict its scent profile across 157 natural and synthetic aroma descriptors, trained on 5,000+ real sensory compounds.
        </p>
      </div>

      {/* SMILES Input Section */}
      <div className="max-w-4xl mx-auto glass-panel rounded-2xl p-5 sm:p-6 shadow-xl border border-white/10 space-y-4">
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center space-y-3 sm:space-y-0 sm:space-x-3">
          {/* Input container */}
          <div className="relative flex-1">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
              <Atom className="w-5 h-5" />
            </div>
            <input
              id="smiles-input"
              type="text"
              value={smilesInput}
              onChange={(e) => setSmilesInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handlePredict(smilesInput, activeModel)}
              placeholder="Enter SMILES string (e.g. O=Cc1ccc(O)c(OC)c1 or CC1=CCC(CC1)C(=C)C)"
              className="w-full pl-11 pr-24 py-3.5 rounded-xl bg-slate-900/90 border border-white/15 text-white font-mono text-xs sm:text-sm focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent transition-all placeholder:text-slate-500"
            />
            {/* Validity indicator */}
            <div className="absolute inset-y-0 right-0 pr-3 flex items-center space-x-1">
              {isValidSmiles ? (
                <span className="flex items-center text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-md border border-emerald-500/30">
                  <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Valid
                </span>
              ) : (
                <span className="flex items-center text-xs font-semibold text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded-md border border-rose-500/30">
                  <AlertCircle className="w-3.5 h-3.5 mr-1" /> Check Syntax
                </span>
              )}
            </div>
          </div>

          {/* Model Switcher Pill */}
          <div className="flex items-center bg-slate-900/90 border border-white/15 rounded-xl p-1 text-xs">
            <button
              onClick={() => setActiveModel("gatv2")}
              className={`px-3 py-2 rounded-lg font-medium transition-all ${
                activeModel === "gatv2"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              GATv2
            </button>
            <button
              onClick={() => setActiveModel("gcn")}
              className={`px-3 py-2 rounded-lg font-medium transition-all ${
                activeModel === "gcn"
                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              GCN
            </button>
          </div>

          {/* Predict Button */}
          <button
            id="predict-submit-btn"
            onClick={() => handlePredict(smilesInput, activeModel)}
            disabled={loading || !isValidSmiles}
            className="px-6 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-sm tracking-wide transition-all shadow-glow-cyan flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-slate-950" />
                <span>Computing...</span>
              </>
            ) : (
              <>
                <span>Predict Scent</span>
                <ArrowRight className="w-4 h-4 text-slate-950" />
              </>
            )}
          </button>
        </div>

        {/* Error message */}
        {error && (
          <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Try an Example Carousel / Grid */}
        <div className="space-y-2 pt-2 border-t border-white/5">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-semibold uppercase tracking-wider text-[11px] text-slate-300">
              Try an Iconic Odorant Molecule:
            </span>
            <span className="text-[11px] text-slate-500">{examples.length} curated standards</span>
          </div>

          <div className="flex items-center space-x-2 overflow-x-auto pb-2 pt-1 scrollbar-thin">
            {examples.map((ex) => {
              const scentCfg = getScentConfig(ex.category);
              const isSelected = smilesInput === ex.smiles;
              return (
                <button
                  key={ex.name}
                  onClick={() => handleSelectExample(ex)}
                  className={`flex-shrink-0 flex items-center space-x-2 px-3 py-1.5 rounded-xl text-xs font-medium border transition-all ${
                    isSelected
                      ? "bg-cyan-500/20 text-cyan-200 border-cyan-400/50 shadow-glow-cyan"
                      : "bg-slate-900/60 hover:bg-slate-800 text-slate-300 border-white/10 hover:border-white/20"
                  }`}
                >
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: scentCfg.barColor }} />
                  <span className="font-semibold">{ex.name}</span>
                  <span className="text-[10px] text-slate-400 hidden sm:inline">({ex.category})</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Loading Skeleton */}
      {loading && (
        <div className="max-w-5xl mx-auto glass-panel rounded-2xl p-8 text-center space-y-4 border border-cyan-500/20 animate-pulse">
          <div className="relative inline-flex items-center justify-center w-16 h-16 rounded-full bg-cyan-500/20 text-cyan-400 mb-2">
            <Atom className="w-10 h-10 animate-spin-slow" />
          </div>
          <h3 className="text-lg font-bold text-white">Running Graph Message Passing</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Propagating 38 node and 7 edge features across atomic bonds using {activeModel === "gatv2" ? "GATv2 multi-head attention" : "GCN convolutions"}...
          </p>
        </div>
      )}

      {/* Prediction Output Section */}
      {!loading && predictionData && (
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: 2D Molecule & Chemical Properties (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            {/* Molecule 2D Structure Card */}
            <div className="glass-panel rounded-2xl p-5 border border-white/10 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs uppercase tracking-wider font-bold text-slate-300 flex items-center space-x-1.5">
                  <Atom className="w-4 h-4 text-cyan-400" />
                  <span>2D Molecular Structure</span>
                </span>
                <button
                  onClick={copySmiles}
                  className="flex items-center space-x-1 text-xs text-slate-400 hover:text-cyan-300 px-2 py-1 rounded bg-white/5 hover:bg-white/10 border border-white/5 transition-colors"
                  title="Copy Canonical SMILES"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? "Copied" : "Copy SMILES"}</span>
                </button>
              </div>

              {/* Molecule Render Image */}
              <div className="relative w-full h-64 rounded-xl bg-slate-950/80 border border-white/5 flex items-center justify-center p-4 overflow-hidden group">
                {predictionData.image_png ? (
                  <img
                    src={predictionData.image_png}
                    alt="2D Molecule"
                    className="max-h-full max-w-full object-contain filter drop-shadow-[0_0_12px_rgba(56,189,248,0.25)] transition-transform group-hover:scale-105 duration-300"
                  />
                ) : (
                  <div className="text-xs text-slate-500">Structure preview unavailable</div>
                )}
                <div className="absolute bottom-2 right-2 text-[10px] font-mono text-slate-400 bg-slate-900/80 px-2 py-0.5 rounded border border-white/5">
                  RDKit 2D Cairo
                </div>
              </div>

              {/* Chemical Stats Grid */}
              <div className="grid grid-cols-2 gap-2.5 text-xs">
                <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                  <div className="text-[11px] text-slate-400 font-medium">Formula</div>
                  <div className="font-mono font-bold text-cyan-300 text-sm">
                    {predictionData.properties?.formula || "N/A"}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                  <div className="text-[11px] text-slate-400 font-medium">Molecular Weight</div>
                  <div className="font-mono font-bold text-white text-sm">
                    {predictionData.properties?.molecular_weight} <span className="text-[10px] text-slate-400">g/mol</span>
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                  <div className="text-[11px] text-slate-400 font-medium">Heavy Atoms</div>
                  <div className="font-mono font-semibold text-slate-200">
                    {predictionData.properties?.heavy_atom_count} atoms
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                  <div className="text-[11px] text-slate-400 font-medium">Rotatable Bonds</div>
                  <div className="font-mono font-semibold text-slate-200">
                    {predictionData.properties?.rotatable_bonds}
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                  <div className="text-[11px] text-slate-400 font-medium">Polar Surface (TPSA)</div>
                  <div className="font-mono font-semibold text-slate-200">
                    {predictionData.properties?.tpsa} Å²
                  </div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                  <div className="text-[11px] text-slate-400 font-medium">Calculated LogP</div>
                  <div className="font-mono font-semibold text-slate-200">
                    {predictionData.properties?.logp}
                  </div>
                </div>
              </div>
            </div>

            {/* Dominant Olfactory Accords Card */}
            <div className="glass-panel rounded-2xl p-5 border border-white/10 space-y-3">
              <div className="text-xs uppercase tracking-wider font-bold text-slate-300 flex items-center space-x-1.5">
                <Sparkles className="w-4 h-4 text-purple-400" />
                <span>Dominant Scent Accords</span>
              </div>
              <p className="text-xs text-slate-400">
                Odor descriptors whose confidence scores exceed the model's optimized validation threshold:
              </p>
              <div className="flex flex-wrap gap-2 pt-1">
                {predictionData.positive_labels && predictionData.positive_labels.length > 0 ? (
                  predictionData.positive_labels.map((lbl) => {
                    const item = predictionData.predictions.find((p) => p.label === lbl);
                    const cfg = getScentConfig(item?.scent_family);
                    return (
                      <span
                        key={lbl}
                        className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border shadow-sm ${cfg.bgClass}`}
                      >
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: cfg.barColor }} />
                        <span className="capitalize">{lbl}</span>
                        <span className="text-[10px] opacity-75 font-mono">
                          ({item ? (item.confidence * 100).toFixed(0) : 0}%)
                        </span>
                      </span>
                    );
                  })
                ) : (
                  <div className="text-xs text-slate-400 italic">
                    Sub-threshold scent bouquet (showing top descriptors below).
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column: Predicted Odor Descriptors & Confidence Bars (7 cols) */}
          <div className="lg:col-span-7 glass-panel rounded-2xl p-5 sm:p-6 border border-white/10 space-y-5">
            {/* Header & Filter Controls */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-4 border-b border-white/10">
              <div>
                <h3 className="text-lg font-bold text-white tracking-tight flex items-center space-x-2">
                  <span>Predicted Odor Profile</span>
                  <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                    {predictionData.model_used.toUpperCase()}
                  </span>
                </h3>
                <p className="text-xs text-slate-400">
                  Sorted by GNN confidence score with per-descriptor threshold markers
                </p>
              </div>

              {/* Filter Buttons */}
              <div className="flex items-center bg-slate-900/80 border border-white/10 rounded-xl p-1 text-xs">
                <button
                  onClick={() => setDisplayFilter("positive")}
                  className={`px-3 py-1 rounded-lg font-medium transition-all ${
                    displayFilter === "positive"
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Positive ({predictionData.positive_labels?.length || 0})
                </button>
                <button
                  onClick={() => setDisplayFilter("top20")}
                  className={`px-3 py-1 rounded-lg font-medium transition-all ${
                    displayFilter === "top20"
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Top 20
                </button>
                <button
                  onClick={() => setDisplayFilter("all")}
                  className={`px-3 py-1 rounded-lg font-medium transition-all ${
                    displayFilter === "all"
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  All 157
                </button>
              </div>
            </div>

            {/* Confidence Bars List */}
            <div className="space-y-3.5 max-h-[600px] overflow-y-auto pr-2 scrollbar-thin">
              {filteredPredictions.map((pred) => {
                const cfg = getScentConfig(pred.scent_family);
                const pct = (pred.confidence * 100).toFixed(1);
                const thPct = (pred.threshold * 100).toFixed(0);

                return (
                  <div
                    key={pred.label}
                    className="p-3 rounded-xl bg-slate-900/50 hover:bg-slate-900/80 border border-white/5 transition-all space-y-1.5"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-2">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: cfg.barColor }} />
                        <span className="font-bold text-white capitalize text-sm">{pred.label}</span>
                        <span className={`text-[10px] px-2 py-0.5 rounded-full border ${cfg.bgClass}`}>
                          {cfg.label}
                        </span>
                      </div>

                      <div className="flex items-center space-x-3">
                        <span className="text-[11px] text-slate-400 font-mono">
                          Threshold: <strong className="text-slate-300">{thPct}%</strong>
                        </span>
                        <span className="font-mono font-bold text-white text-sm">
                          {pct}%
                        </span>
                        {pred.is_positive ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                            Active
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-500 border border-white/5">
                            Sub-threshold
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Progress Bar Container */}
                    <div className="relative w-full h-3 rounded-full bg-slate-950 overflow-hidden border border-white/5">
                      {/* Animated Fill Bar */}
                      <div
                        className={`h-full rounded-full transition-all duration-700 bg-gradient-to-r ${cfg.gradient}`}
                        style={{ width: `${Math.max(2, pred.confidence * 100)}%` }}
                      />
                      {/* Threshold Tick Marker */}
                      <div
                        className="absolute top-0 bottom-0 w-0.5 bg-white/70 z-10"
                        style={{ left: `${pred.threshold * 100}%` }}
                        title={`Decision Threshold: ${thPct}%`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Explanatory Legend */}
            <div className="pt-3 border-t border-white/10 flex flex-wrap items-center justify-between text-[11px] text-slate-400 gap-2">
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-white/70" />
                <span>Vertical white notch indicates optimal per-label decision threshold</span>
              </div>
              <div className="flex items-center space-x-1 text-slate-400">
                <Info className="w-3.5 h-3.5" />
                <span>Multi-label sigmoid outputs conditioned on molecular graph</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
