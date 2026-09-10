/**
 * Scent Family styling and color configurations
 */

export const SCENT_FAMILY_CONFIG = {
  floral: {
    label: "Floral",
    gradient: "from-pink-500 to-rose-400",
    barColor: "#f43f5e",
    bgClass: "bg-rose-500/10 text-rose-300 border-rose-500/30",
    glowClass: "glow-badge-floral",
    dotColor: "#fb7185",
  },
  woody: {
    label: "Woody",
    gradient: "from-amber-600 to-yellow-600",
    barColor: "#d97706",
    bgClass: "bg-amber-600/10 text-amber-300 border-amber-600/30",
    glowClass: "glow-badge-woody",
    dotColor: "#f59e0b",
  },
  citrus: {
    label: "Citrus",
    gradient: "from-yellow-400 to-lime-400",
    barColor: "#eab308",
    bgClass: "bg-yellow-500/10 text-yellow-300 border-yellow-500/30",
    glowClass: "glow-badge-citrus",
    dotColor: "#facc15",
  },
  fruity: {
    label: "Fruity",
    gradient: "from-pink-500 to-fuchsia-500",
    barColor: "#ec4899",
    bgClass: "bg-pink-500/10 text-pink-300 border-pink-500/30",
    glowClass: "glow-badge-fruity",
    dotColor: "#f472b6",
  },
  sweet: {
    label: "Sweet",
    gradient: "from-purple-500 to-violet-400",
    barColor: "#a855f7",
    bgClass: "bg-purple-500/10 text-purple-300 border-purple-500/30",
    glowClass: "glow-badge-sweet",
    dotColor: "#c084fc",
  },
  green: {
    label: "Green / Herbal",
    gradient: "from-emerald-500 to-teal-400",
    barColor: "#10b981",
    bgClass: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
    glowClass: "glow-badge-green",
    dotColor: "#34d399",
  },
  spicy: {
    label: "Spicy",
    gradient: "from-orange-500 to-amber-500",
    barColor: "#f97316",
    bgClass: "bg-orange-500/10 text-orange-300 border-orange-500/30",
    glowClass: "glow-badge-spicy",
    dotColor: "#fb923c",
  },
  sulfurous: {
    label: "Sulfurous / Savory",
    gradient: "from-red-600 to-rose-700",
    barColor: "#dc2626",
    bgClass: "bg-red-600/10 text-red-300 border-red-600/30",
    glowClass: "glow-badge-sulfurous",
    dotColor: "#f87171",
  },
  earthy: {
    label: "Earthy / Animalic",
    gradient: "from-amber-700 to-stone-600",
    barColor: "#b45309",
    bgClass: "bg-amber-800/15 text-amber-200 border-amber-700/30",
    glowClass: "glow-badge-earthy",
    dotColor: "#d97706",
  },
  fresh: {
    label: "Fresh / Ethereal",
    gradient: "from-cyan-400 to-sky-400",
    barColor: "#06b6d4",
    bgClass: "bg-cyan-500/10 text-cyan-300 border-cyan-500/30",
    glowClass: "glow-badge-fresh",
    dotColor: "#38bdf8",
  },
  other: {
    label: "Other",
    gradient: "from-slate-400 to-gray-400",
    barColor: "#64748b",
    bgClass: "bg-slate-500/10 text-slate-300 border-slate-500/30",
    glowClass: "bg-slate-800 text-slate-300 border border-slate-700",
    dotColor: "#94a3b8",
  }
};

export function getScentConfig(family) {
  return SCENT_FAMILY_CONFIG[family?.toLowerCase()] || SCENT_FAMILY_CONFIG.other;
}
