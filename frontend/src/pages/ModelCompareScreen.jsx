import React, { useState, useEffect } from "react";
import { 
  BarChart2, ArrowUpRight, ArrowDownRight, Layers, Cpu, 
  Search, Filter, CheckCircle2, TrendingUp, Info
} from "lucide-react";
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid 
} from "recharts";
import { getModelComparison } from "../services/api";
import { getScentConfig } from "../utils/scentColors";

export default function ModelCompareScreen() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortField, setSortField] = useState("support"); // 'support', 'gatv2_f1', 'gcn_f1', 'delta_f1'
  const [sortAsc, setSortAsc] = useState(false);

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

  // Format chart data for Top 10 Best and Top 10 Worst
  const topBestChartData = React.useMemo(() => {
    if (!data?.gatv2_summary?.top_10_best) return [];
    return data.gatv2_summary.top_10_best.map((item) => {
      const gcnMatch = data.gcn_summary.top_10_best.find((g) => g.label === item.label) ||
                       data.comparison_table.find((c) => c.label === item.label);
      return {
        label: item.label,
        "GATv2 (Attention)": Number((item.f1 * 100).toFixed(1)),
        "GCN Baseline": Number(((gcnMatch?.f1 || gcnMatch?.gcn_f1 || 0) * 100).toFixed(1)),
      };
    });
  }, [data]);

  const topWorstChartData = React.useMemo(() => {
    if (!data?.gatv2_summary?.top_10_worst) return [];
    return data.gatv2_summary.top_10_worst.map((item) => {
      const gcnMatch = data.comparison_table.find((c) => c.label === item.label);
      return {
        label: item.label,
        "GATv2 (Attention)": Number((item.f1 * 100).toFixed(1)),
        "GCN Baseline": Number(((gcnMatch?.gcn_f1 || 0) * 100).toFixed(1)),
      };
    });
  }, [data]);

  // Filter and sort comparison table
  const filteredTable = React.useMemo(() => {
    if (!data?.comparison_table) return [];
    let list = data.comparison_table.filter((row) =>
      row.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
      row.scent_family.toLowerCase().includes(searchQuery.toLowerCase())
    );

    list.sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];
      if (typeof valA === "string") {
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return sortAsc ? valA - valB : valB - valA;
    });

    return list;
  }, [data, searchQuery, sortField, sortAsc]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto py-16 text-center space-y-4">
        <div className="inline-flex p-3 rounded-2xl bg-cyan-500/20 text-cyan-400 animate-spin">
          <Layers className="w-8 h-8" />
        </div>
        <p className="text-sm text-slate-400">Loading side-by-side GNN model comparison...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-xl mx-auto p-6 glass-panel rounded-2xl border border-rose-500/30 text-rose-300 text-center space-y-2">
        <p className="font-bold">Error loading model comparison</p>
        <p className="text-xs text-slate-400">{error}</p>
      </div>
    );
  }

  const { gatv2_summary, gcn_summary } = data;

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      {/* Title */}
      <div className="text-center max-w-3xl mx-auto space-y-2 pt-4">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Model Architecture Comparison
        </h1>
        <p className="text-slate-400 text-xs sm:text-sm">
          Comparative empirical evaluation of ScentGATv2 (Graph Attention Network with edge conditioning) versus GCN Baseline across 157 molecular odor descriptors.
        </p>
      </div>

      {/* KPI Comparison Cards */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* GATv2 Card */}
        <div className="glass-panel rounded-2xl p-6 border border-cyan-500/30 relative overflow-hidden group shadow-glow-cyan">
          <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-full blur-2xl group-hover:bg-cyan-500/20 transition-all pointer-events-none" />
          <div className="flex items-center justify-between pb-4 border-b border-white/10">
            <div className="flex items-center space-x-3">
              <div className="p-2.5 rounded-xl bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-lg text-white">ScentGATv2</h3>
                <span className="text-[11px] text-cyan-300 font-semibold">Graph Attention + Edge Conditioning</span>
              </div>
            </div>
            <span className="text-xs font-bold px-2.5 py-1 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
              Champion
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 py-5">
            <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
              <div className="text-[11px] text-slate-400">Test Macro-F1</div>
              <div className="text-2xl font-extrabold font-mono text-cyan-300">
                {(gatv2_summary.macro_f1 * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] text-emerald-400 flex items-center font-medium">
                <ArrowUpRight className="w-3 h-3 mr-0.5" /> +{((gatv2_summary.macro_f1 - gcn_summary.macro_f1) * 100).toFixed(1)}% vs GCN
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
              <div className="text-[11px] text-slate-400">Test Micro-F1</div>
              <div className="text-2xl font-extrabold font-mono text-white">
                {(gatv2_summary.micro_f1 * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] text-slate-400 font-medium">Overall accuracy</div>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
              <div className="text-[11px] text-slate-400">Test BCE Loss</div>
              <div className="text-2xl font-extrabold font-mono text-white">
                {gatv2_summary.test_loss}
              </div>
              <div className="text-[10px] text-emerald-400 font-medium">Lower entropy</div>
            </div>
          </div>

          <div className="text-xs text-slate-400 space-y-1 pt-1 border-t border-white/5">
            <div className="flex justify-between">
              <span>Message Passing:</span>
              <strong className="text-slate-200">4 GATv2Conv rounds + bond features</strong>
            </div>
            <div className="flex justify-between">
              <span>Readout Pooling:</span>
              <strong className="text-slate-200">Global Attention Gating network</strong>
            </div>
          </div>
        </div>

        {/* GCN Baseline Card */}
        <div className="glass-panel rounded-2xl p-6 border border-purple-500/20 relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-2xl group-hover:bg-purple-500/20 transition-all pointer-events-none" />
          <div className="flex items-center justify-between pb-4 border-b border-white/10">
            <div className="flex items-center space-x-3">
              <div className="p-2.5 rounded-xl bg-purple-500/20 text-purple-300 border border-purple-500/30">
                <Layers className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-lg text-white">GCN Baseline</h3>
                <span className="text-[11px] text-purple-300 font-semibold">Graph Convolutional Network</span>
              </div>
            </div>
            <span className="text-xs font-bold px-2.5 py-1 rounded-full bg-purple-500/15 text-purple-300 border border-purple-500/30">
              Standard Baseline
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 py-5">
            <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
              <div className="text-[11px] text-slate-400">Test Macro-F1</div>
              <div className="text-2xl font-extrabold font-mono text-purple-300">
                {(gcn_summary.macro_f1 * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] text-slate-400 font-medium">Averaged across 157 labels</div>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
              <div className="text-[11px] text-slate-400">Test Micro-F1</div>
              <div className="text-2xl font-extrabold font-mono text-white">
                {(gcn_summary.micro_f1 * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] text-slate-400 font-medium">Overall accuracy</div>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
              <div className="text-[11px] text-slate-400">Test BCE Loss</div>
              <div className="text-2xl font-extrabold font-mono text-white">
                {gcn_summary.test_loss}
              </div>
              <div className="text-[10px] text-slate-400 font-medium">Cross-entropy loss</div>
            </div>
          </div>

          <div className="text-xs text-slate-400 space-y-1 pt-1 border-t border-white/5">
            <div className="flex justify-between">
              <span>Message Passing:</span>
              <strong className="text-slate-200">3 GCNConv rounds with skip connections</strong>
            </div>
            <div className="flex justify-between">
              <span>Readout Pooling:</span>
              <strong className="text-slate-200">Concatenated Mean & Max pooling</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Top 10 Best and Worst Odor Descriptors Chart */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top 10 Best Predicted */}
        <div className="glass-panel rounded-2xl p-5 sm:p-6 border border-white/10 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
              <span>Top 10 Highest F1 Odor Descriptors</span>
            </h3>
            <p className="text-xs text-slate-400">
              Descriptors where chemical functional groups provide the clearest graph signals (e.g. sulfur rings, lactones, phenols)
            </p>
          </div>

          <div className="h-72 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topBestChartData} layout="vertical" margin={{ top: 5, right: 20, left: 40, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" horizontal={true} vertical={false} />
                <XAxis type="number" domain={[0, 100]} unit="%" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis dataKey="label" type="category" stroke="#cbd5e1" tick={{ fontSize: 11, fontWeight: 600 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "rgba(255,255,255,0.1)", borderRadius: "10px", fontSize: "12px" }}
                  formatter={(val) => [`${val}%`, ""]}
                />
                <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "6px" }} />
                <Bar dataKey="GATv2 (Attention)" fill="#38bdf8" radius={[0, 4, 4, 0]} />
                <Bar dataKey="GCN Baseline" fill="#a855f7" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top 10 Hardest Odor Descriptors */}
        <div className="glass-panel rounded-2xl p-5 sm:p-6 border border-white/10 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center space-x-2">
              <Info className="w-4 h-4 text-amber-400" />
              <span>Most Challenging Odor Descriptors</span>
            </h3>
            <p className="text-xs text-slate-400">
              Subjective descriptors with high perceptual variance or sparse training instances
            </p>
          </div>

          <div className="h-72 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topWorstChartData} layout="vertical" margin={{ top: 5, right: 20, left: 45, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" horizontal={true} vertical={false} />
                <XAxis type="number" domain={[0, 100]} unit="%" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis dataKey="label" type="category" stroke="#cbd5e1" tick={{ fontSize: 11, fontWeight: 600 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "rgba(255,255,255,0.1)", borderRadius: "10px", fontSize: "12px" }}
                  formatter={(val) => [`${val}%`, ""]}
                />
                <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "6px" }} />
                <Bar dataKey="GATv2 (Attention)" fill="#38bdf8" radius={[0, 4, 4, 0]} />
                <Bar dataKey="GCN Baseline" fill="#a855f7" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Full 157 Descriptors Sortable Table */}
      <div className="max-w-6xl mx-auto glass-panel rounded-2xl p-5 sm:p-6 border border-white/10 space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
          <div>
            <h3 className="text-lg font-bold text-white">All 157 Odor Descriptors Benchmark</h3>
            <p className="text-xs text-slate-400">Click column headers to sort by test F1, sample support, or model advantage (delta)</p>
          </div>

          {/* Search box */}
          <div className="relative w-full sm:w-64">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search descriptors..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 rounded-lg bg-slate-900 border border-white/10 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-400"
            />
          </div>
        </div>

        {/* Responsive Table */}
        <div className="overflow-x-auto max-h-[500px] scrollbar-thin">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-slate-950/90 backdrop-blur-md text-slate-400 border-b border-white/10 select-none">
              <tr>
                <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort("label")}>
                  Descriptor {sortField === "label" && (sortAsc ? "↑" : "↓")}
                </th>
                <th className="py-3 px-3">Fragrance Family</th>
                <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort("support")}>
                  Test Support {sortField === "support" && (sortAsc ? "↑" : "↓")}
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-cyan-300" onClick={() => handleSort("gatv2_f1")}>
                  GATv2 F1 {sortField === "gatv2_f1" && (sortAsc ? "↑" : "↓")}
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-purple-300" onClick={() => handleSort("gcn_f1")}>
                  GCN F1 {sortField === "gcn_f1" && (sortAsc ? "↑" : "↓")}
                </th>
                <th className="py-3 px-3 cursor-pointer hover:text-white" onClick={() => handleSort("delta_f1")}>
                  Delta (GATv2 - GCN) {sortField === "delta_f1" && (sortAsc ? "↑" : "↓")}
                </th>
                <th className="py-3 px-3">GATv2 Threshold</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 font-mono text-[11px]">
              {filteredTable.map((row) => {
                const cfg = getScentConfig(row.scent_family);
                const isGatv2Better = row.delta_f1 > 0;
                const isTied = row.delta_f1 === 0;

                return (
                  <tr key={row.label} className="hover:bg-white/5 transition-colors">
                    <td className="py-2.5 px-3 font-sans font-bold text-white capitalize text-xs">
                      {row.label}
                    </td>
                    <td className="py-2.5 px-3 font-sans">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] border ${cfg.bgClass}`}>
                        {cfg.label}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {row.support} molecules
                    </td>
                    <td className="py-2.5 px-3 font-bold text-cyan-300">
                      {(row.gatv2_f1 * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 font-bold text-purple-300">
                      {(row.gcn_f1 * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 font-bold">
                      {isTied ? (
                        <span className="text-slate-500">0.0%</span>
                      ) : isGatv2Better ? (
                        <span className="text-emerald-400">
                          +{(row.delta_f1 * 100).toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-rose-400">
                          {(row.delta_f1 * 100).toFixed(1)}%
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">
                      {row.gatv2_threshold}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
