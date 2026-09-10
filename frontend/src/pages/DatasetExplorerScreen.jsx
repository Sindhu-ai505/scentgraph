import React, { useState, useEffect } from "react";
import { Database, Search, Filter, Hash, BarChart3, PieChart, Sparkles, ExternalLink } from "lucide-react";
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell 
} from "recharts";
import { getLabels } from "../services/api";
import { getScentConfig, SCENT_FAMILY_CONFIG } from "../utils/scentColors";

export default function DatasetExplorerScreen() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFamily, setSelectedFamily] = useState("all");

  useEffect(() => {
    getLabels()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Top 25 for the frequency chart
  const top25Data = React.useMemo(() => {
    if (!data?.labels) return [];
    return data.labels.slice(0, 25).map((item) => ({
      label: item.label,
      count: item.count,
      percentage: item.percentage,
      scent_family: item.scent_family,
      barColor: getScentConfig(item.scent_family).barColor
    }));
  }, [data]);

  // Filtered vocabulary list
  const filteredLabels = React.useMemo(() => {
    if (!data?.labels) return [];
    return data.labels.filter((item) => {
      const matchesSearch = item.label.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesFamily = selectedFamily === "all" || item.scent_family === selectedFamily;
      return matchesSearch && matchesFamily;
    });
  }, [data, searchQuery, selectedFamily]);

  // Aggregate stats per fragrance family
  const familyDistribution = React.useMemo(() => {
    if (!data?.labels) return {};
    const map = {};
    for (const item of data.labels) {
      const fam = item.scent_family || "other";
      map[fam] = (map[fam] || 0) + item.count;
    }
    return map;
  }, [data]);

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto py-16 text-center space-y-4">
        <div className="inline-flex p-3 rounded-2xl bg-cyan-500/20 text-cyan-400 animate-spin">
          <Database className="w-8 h-8" />
        </div>
        <p className="text-sm text-slate-400">Loading dataset distribution & label vocabulary...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-xl mx-auto p-6 glass-panel rounded-2xl border border-rose-500/30 text-rose-300 text-center space-y-2">
        <p className="font-bold">Error loading dataset</p>
        <p className="text-xs text-slate-400">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      {/* Title */}
      <div className="text-center max-w-3xl mx-auto space-y-2 pt-4">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Dataset Explorer & Label Vocabulary
        </h1>
        <p className="text-slate-400 text-xs sm:text-sm">
          Transparent exploration of the 5,070 canonicalized molecules and 157 filtered odor descriptors curated from Leffingwell, GoodScents, Flavornet, and AromaDB.
        </p>
      </div>

      {/* KPI Tiles */}
      <div className="max-w-6xl mx-auto grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="glass-panel rounded-2xl p-5 border border-white/10 space-y-1">
          <div className="text-xs text-slate-400 font-medium">Total Molecules</div>
          <div className="text-2xl sm:text-3xl font-extrabold font-mono text-cyan-300">
            {data.total_molecules.toLocaleString()}
          </div>
          <div className="text-[11px] text-slate-500">Deduplicated canonical SMILES</div>
        </div>

        <div className="glass-panel rounded-2xl p-5 border border-white/10 space-y-1">
          <div className="text-xs text-slate-400 font-medium">Odor Descriptors</div>
          <div className="text-2xl sm:text-3xl font-extrabold font-mono text-purple-300">
            {data.total_labels}
          </div>
          <div className="text-[11px] text-slate-500">≥ 30 occurrences each</div>
        </div>

        <div className="glass-panel rounded-2xl p-5 border border-white/10 space-y-1">
          <div className="text-xs text-slate-400 font-medium">Avg Labels / Mol</div>
          <div className="text-2xl sm:text-3xl font-extrabold font-mono text-emerald-300">
            4.89
          </div>
          <div className="text-[11px] text-slate-500">Max: 20 descriptors</div>
        </div>

        <div className="glass-panel rounded-2xl p-5 border border-white/10 space-y-1">
          <div className="text-xs text-slate-400 font-medium">Public Sources</div>
          <div className="text-2xl sm:text-3xl font-extrabold font-mono text-amber-300">
            4
          </div>
          <div className="text-[11px] text-slate-500">Pyrfume-data archive</div>
        </div>
      </div>

      {/* Top 25 Frequency Bar Chart */}
      <div className="max-w-6xl mx-auto glass-panel rounded-2xl p-5 sm:p-6 border border-white/10 space-y-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center space-x-2">
            <BarChart3 className="w-4 h-4 text-cyan-400" />
            <span>Top 25 Most Frequent Odor Descriptors in Training Data</span>
          </h3>
          <p className="text-xs text-slate-400">
            Distribution of prevalent chemical scent families across the 5,070 molecules
          </p>
        </div>

        <div className="h-80 w-full pt-3">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={top25Data} margin={{ top: 10, right: 10, left: 0, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis 
                dataKey="label" 
                stroke="#64748b" 
                angle={-45} 
                textAnchor="end" 
                interval={0} 
                tick={{ fontSize: 11, fill: "#cbd5e1" }} 
              />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", borderColor: "rgba(255,255,255,0.1)", borderRadius: "10px", fontSize: "12px" }}
                formatter={(val, name, props) => [`${val} molecules (${props.payload.percentage}%)`, "Frequency"]}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {top25Data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.barColor} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Searchable Vocabulary Table */}
      <div className="max-w-6xl mx-auto glass-panel rounded-2xl p-5 sm:p-6 border border-white/10 space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
          <div>
            <h3 className="text-lg font-bold text-white">Full Label Vocabulary ({filteredLabels.length} of {data.total_labels})</h3>
            <p className="text-xs text-slate-400">Filter by fragrance family or search for specific scent descriptors</p>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center space-y-2 sm:space-y-0 sm:space-x-3 w-full sm:w-auto">
            {/* Family filter dropdown */}
            <select
              value={selectedFamily}
              onChange={(e) => setSelectedFamily(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-slate-900 border border-white/10 text-xs text-white focus:outline-none focus:ring-1 focus:ring-cyan-400"
            >
              <option value="all">All Scent Families</option>
              {Object.keys(SCENT_FAMILY_CONFIG).map((fam) => (
                <option key={fam} value={fam}>
                  {SCENT_FAMILY_CONFIG[fam].label}
                </option>
              ))}
            </select>

            {/* Search Input */}
            <div className="relative w-full sm:w-56">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search vocabulary..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-3 py-1.5 rounded-lg bg-slate-900 border border-white/10 text-xs text-white placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-400"
              />
            </div>
          </div>
        </div>

        {/* Vocabulary Grid Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 max-h-[500px] overflow-y-auto pr-2 scrollbar-thin">
          {filteredLabels.map((item) => {
            const cfg = getScentConfig(item.scent_family);
            return (
              <div
                key={item.label}
                className="p-3 rounded-xl bg-slate-900/50 hover:bg-slate-900/80 border border-white/5 space-y-1.5 transition-all"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-white capitalize text-xs">{item.label}</span>
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: cfg.barColor }} />
                </div>
                <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                  <span>{item.count} molecules</span>
                  <span className="font-bold text-slate-300">{item.percentage}%</span>
                </div>
                <div className={`text-[10px] px-2 py-0.5 rounded-full border text-center ${cfg.bgClass}`}>
                  {cfg.label}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
