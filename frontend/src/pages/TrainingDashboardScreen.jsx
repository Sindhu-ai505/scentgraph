import React, { useState, useEffect } from "react";
import { Activity, TrendingUp, Sliders, CheckCircle, ShieldCheck, Flame, GitCommit } from "lucide-react";
import { 
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid 
} from "recharts";
import { getModelComparison } from "../services/api";

export default function TrainingDashboardScreen() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModelComparison()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Format line chart data combining GATv2 and GCN curves by epoch
  const combinedCurves = React.useMemo(() => {
    if (!data?.gatv2_curves) return [];
    const gatCurves = data.gatv2_curves;
    const gcnCurves = data.gcn_curves || [];

    return gatCurves.map((gPoint, idx) => {
      const gcnPoint = gcnCurves[idx] || {};
      return {
        epoch: `Epoch ${gPoint.epoch}`,
        "GATv2 Train Loss": gPoint.train_loss,
        "GATv2 Val Loss": gPoint.val_loss,
        "GCN Train Loss": gcnPoint.train_loss,
        "GCN Val Loss": gcnPoint.val_loss,
        "GATv2 Val Macro-F1": Number((gPoint.val_macro_f1 * 100).toFixed(1)),
        "GCN Val Macro-F1": Number(((gcnPoint.val_macro_f1 || 0) * 100).toFixed(1)),
        "GATv2 Val Micro-F1": Number((gPoint.val_micro_f1 * 100).toFixed(1)),
        "GCN Val Micro-F1": Number(((gcnPoint.val_micro_f1 || 0) * 100).toFixed(1)),
      };
    });
  }, [data]);

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto py-16 text-center space-y-4">
        <div className="inline-flex p-3 rounded-2xl bg-cyan-500/20 text-cyan-400 animate-spin">
          <Activity className="w-8 h-8" />
        </div>
        <p className="text-sm text-slate-400">Loading training curves from logged checkpoints...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-xl mx-auto p-6 glass-panel rounded-2xl border border-rose-500/30 text-rose-300 text-center space-y-2">
        <p className="font-bold">Error loading training dashboard</p>
        <p className="text-xs text-slate-400">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      {/* Title */}
      <div className="text-center max-w-3xl mx-auto space-y-2 pt-4">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Training Dynamics & Convergence
        </h1>
        <p className="text-slate-400 text-xs sm:text-sm">
          Live epoch-by-epoch loss trajectories and Macro-F1 progression recorded during the 15-epoch training run.
        </p>
      </div>

      {/* Line Chart 1: Macro-F1 Progression */}
      <div className="max-w-6xl mx-auto glass-panel rounded-2xl p-5 sm:p-6 border border-white/10 space-y-4 shadow-xl">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              <span>Validation Macro-F1 Learning Curves</span>
            </h3>
            <p className="text-xs text-slate-400">
              Macro-F1 tracks unweighted average across all 157 descriptors, penalizing poor performance on rare notes
            </p>
          </div>
          <span className="text-xs font-mono text-cyan-300 bg-cyan-500/10 px-2.5 py-1 rounded-full border border-cyan-500/30 font-semibold">
            GATv2 Peak: {(data.gatv2_summary.macro_f1 * 100).toFixed(1)}% Test
          </span>
        </div>

        <div className="h-80 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={combinedCurves} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 30]} unit="%" stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", borderColor: "rgba(255,255,255,0.1)", borderRadius: "10px", fontSize: "12px" }}
                formatter={(val) => [`${val}%`, ""]}
              />
              <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "6px" }} />
              <Line 
                type="monotone" 
                dataKey="GATv2 Val Macro-F1" 
                stroke="#38bdf8" 
                strokeWidth={3} 
                dot={{ r: 4, fill: "#38bdf8" }} 
                activeDot={{ r: 6 }} 
              />
              <Line 
                type="monotone" 
                dataKey="GCN Val Macro-F1" 
                stroke="#a855f7" 
                strokeWidth={2.5} 
                strokeDasharray="4 4"
                dot={{ r: 3, fill: "#a855f7" }} 
              />
              <Line 
                type="monotone" 
                dataKey="GATv2 Val Micro-F1" 
                stroke="#34d399" 
                strokeWidth={1.5} 
                dot={false} 
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Line Chart 2: BCE Loss Trajectories */}
      <div className="max-w-6xl mx-auto glass-panel rounded-2xl p-5 sm:p-6 border border-white/10 space-y-4 shadow-xl">
        <div>
          <h3 className="text-base font-bold text-white flex items-center space-x-2">
            <Activity className="w-4 h-4 text-purple-400" />
            <span>Binary Cross-Entropy Loss Trajectory</span>
          </h3>
          <p className="text-xs text-slate-400">
            Training loss vs Validation loss with positive-class reweighting across 15 epochs
          </p>
        </div>

        <div className="h-80 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={combinedCurves} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis domain={[0.2, 0.8]} stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", borderColor: "rgba(255,255,255,0.1)", borderRadius: "10px", fontSize: "12px" }}
              />
              <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "6px" }} />
              <Line 
                type="monotone" 
                dataKey="GATv2 Train Loss" 
                stroke="#f43f5e" 
                strokeWidth={2} 
                dot={false} 
              />
              <Line 
                type="monotone" 
                dataKey="GATv2 Val Loss" 
                stroke="#38bdf8" 
                strokeWidth={2.5} 
                dot={{ r: 3, fill: "#38bdf8" }} 
              />
              <Line 
                type="monotone" 
                dataKey="GCN Val Loss" 
                stroke="#a855f7" 
                strokeWidth={2} 
                strokeDasharray="3 3"
                dot={{ r: 3, fill: "#a855f7" }} 
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Hyperparameter & Setup Specification */}
      <div className="max-w-6xl mx-auto glass-panel rounded-2xl p-6 border border-white/10 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center space-x-2">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <span>Training Protocol & Hyperparameters</span>
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
            <span className="text-slate-400">Optimizer</span>
            <div className="font-mono font-bold text-white">AdamW (lr=1e-3, wd=1e-4)</div>
            <p className="text-[10px] text-slate-500">With ReduceLROnPlateau decay</p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
            <span className="text-slate-400">Loss Function</span>
            <div className="font-mono font-bold text-cyan-300">Class-Weighted BCE</div>
            <p className="text-[10px] text-slate-500">pos_weight clipped to [1.0, 25.0]</p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
            <span className="text-slate-400">Stratification Split</span>
            <div className="font-mono font-bold text-purple-300">Iterative Multi-Label</div>
            <p className="text-[10px] text-slate-500">80% Train / 10% Val / 10% Test</p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
            <span className="text-slate-400">Threshold Optimization</span>
            <div className="font-mono font-bold text-emerald-300">Validation Grid Tuning</div>
            <p className="text-[10px] text-slate-500">Maximizes per-label F1 score</p>
          </div>
        </div>
      </div>
    </div>
  );
}
