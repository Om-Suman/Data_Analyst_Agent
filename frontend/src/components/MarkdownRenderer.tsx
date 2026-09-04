import React, { useState } from "react";
import {
  Copy,
  Check,
  TrendingUp,
  Award,
  DollarSign,
  Percent,
} from "lucide-react";

interface MarkdownRendererProps {
  content: string;
  className?: string;
  showKpiCards?: boolean;
}

interface ExtractedKpi {
  label: string;
  value: string;
  subtext?: string;
  icon?: "dollar" | "percent" | "award" | "trend";
}

/**
 * Extracts top numerical KPIs from structured markdown text.
 */
export function extractKpisFromMarkdown(text: string): ExtractedKpi[] {
  const kpis: ExtractedKpi[] = [];
  if (!text) return kpis;

  const lines = text.split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    // Look for lines like "- 🥇 **Technology**: **$836,154.03** (36.4% share)..."
    // or "- **Sales** & **Profit**: **r = +0.98**..."
    const matchRank = trimmed.match(
      /^-\s*([🥇🥈🥉4️⃣5️⃣#\d]+)?\s*\*\*([^*]+)\*\*:\s*\*\*([^*]+)\*\*(?:\s*\(([^)]+)\))?/,
    );
    if (matchRank && kpis.length < 4) {
      const label = matchRank[2].trim();
      const value = matchRank[3].trim();
      const subtext = matchRank[4]
        ? matchRank[4].trim()
        : matchRank[1]
          ? `Rank ${matchRank[1]}`
          : undefined;

      let icon: "dollar" | "percent" | "award" | "trend" = "trend";
      if (value.includes("$")) icon = "dollar";
      else if (value.includes("%")) icon = "percent";
      else if (
        matchRank[1] &&
        (matchRank[1].includes("🥇") ||
          matchRank[1].includes("🥈") ||
          matchRank[1].includes("🥉"))
      )
        icon = "award";

      kpis.push({ label, value, subtext, icon });
    }
  }

  return kpis;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  className = "",
  showKpiCards = true,
}) => {
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

  if (!content) return null;

  const kpis = showKpiCards ? extractKpisFromMarkdown(content) : [];

  const handleCopyCode = (code: string, idx: number) => {
    navigator.clipboard.writeText(code);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  const renderInline = (text: string): React.ReactNode => {
    // Process bold, italic, code, and links
    const parts: React.ReactNode[] = [];
    let remaining = text;
    let key = 0;

    // Pattern for inline code, bold, and italic
    const regex = /(\*\*.*?\*\*|\*.*?\*|`.*?`)/g;
    let match: RegExpExecArray | null;
    let lastIndex = 0;

    while ((match = regex.exec(remaining)) !== null) {
      if (match.index > lastIndex) {
        parts.push(remaining.substring(lastIndex, match.index));
      }

      const token = match[0];
      if (token.startsWith("`") && token.endsWith("`")) {
        parts.push(
          <code
            key={key++}
            className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800/80 text-amber-700 dark:text-amber-300 font-mono text-[11px] border border-slate-200 dark:border-slate-700/50"
          >
            {token.slice(1, -1)}
          </code>,
        );
      } else if (token.startsWith("**") && token.endsWith("**")) {
        const inner = token.slice(2, -2);
        // Highlight numbers/currencies specially
        const isNumOrCurrency =
          /^[\$€£]?\s*[\d,]+(\.\d+)?%?$/.test(inner.trim()) ||
          /^[+-]?[\$€£]?[\d,]+(\.\d+)?%?$/.test(inner.trim());
        parts.push(
          <strong
            key={key++}
            className={
              isNumOrCurrency
                ? "font-bold text-blue-600 dark:text-blue-400 px-0.5"
                : "font-semibold text-slate-900 dark:text-slate-100"
            }
          >
            {inner}
          </strong>,
        );
      } else if (token.startsWith("*") && token.endsWith("*")) {
        parts.push(
          <em key={key++} className="italic text-slate-600 dark:text-slate-400">
            {token.slice(1, -1)}
          </em>,
        );
      }

      lastIndex = regex.lastIndex;
    }

    if (lastIndex < remaining.length) {
      parts.push(remaining.substring(lastIndex));
    }

    return parts.length > 0 ? parts : text;
  };

  // Parse lines and blocks
  const lines = content.split("\n");
  const blocks: React.ReactNode[] = [];
  let inCodeBlock = false;
  let codeBuffer: string[] = [];
  let codeLang = "";
  let inTable = false;
  let tableRows: string[][] = [];
  let blockKey = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Code block toggle
    if (trimmed.startsWith("```")) {
      if (inCodeBlock) {
        const codeText = codeBuffer.join("\n");
        const currentIdx = blockKey;
        blocks.push(
          <div
            key={blockKey++}
            className="my-3 rounded-xl overflow-hidden border border-slate-800 bg-[#080d1a] shadow-inner"
          >
            <div className="flex items-center justify-between px-3.5 py-1.5 bg-slate-900/80 border-b border-slate-800 text-[11px] text-slate-400 font-mono">
              <span className="uppercase tracking-wider font-semibold">
                {codeLang || "output"}
              </span>
              <button
                onClick={() => handleCopyCode(codeText, currentIdx)}
                className="flex items-center gap-1 hover:text-white transition-colors"
                title="Copy code"
              >
                {copiedIdx === currentIdx ? (
                  <Check className="h-3 w-3 text-emerald-400" />
                ) : (
                  <Copy className="h-3 w-3" />
                )}
                <span>{copiedIdx === currentIdx ? "Copied" : "Copy"}</span>
              </button>
            </div>
            <pre className="p-3.5 text-xs font-mono text-emerald-300 overflow-x-auto leading-relaxed">
              <code>{codeText}</code>
            </pre>
          </div>,
        );
        inCodeBlock = false;
        codeBuffer = [];
        codeLang = "";
      } else {
        inCodeBlock = true;
        codeLang = trimmed.replace("```", "").trim();
        codeBuffer = [];
      }
      continue;
    }

    if (inCodeBlock) {
      codeBuffer.push(line);
      continue;
    }

    // Markdown Table parsing
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
      if (!inTable) {
        inTable = true;
        tableRows = [];
      }
      // Skip separator rows like |---|---|
      if (!/^\|(?:\s*:?-+:?\s*\|)+$/.test(trimmed)) {
        const cells = trimmed
          .split("|")
          .slice(1, -1)
          .map((c) => c.trim());
        tableRows.push(cells);
      }
      continue;
    } else if (inTable) {
      // Table closed
      const headers = tableRows[0] || [];
      const bodyRows = tableRows.slice(1);
      blocks.push(
        <div
          key={blockKey++}
          className="my-3 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm"
        >
          <table className="w-full text-left text-xs border-collapse bg-white dark:bg-slate-900/40">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/80">
                {headers.map((h, idx) => (
                  <th
                    key={idx}
                    className="px-3.5 py-2.5 font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider text-[11px]"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800/60">
              {bodyRows.map((row, rIdx) => (
                <tr
                  key={rIdx}
                  className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors"
                >
                  {row.map((cell, cIdx) => (
                    <td
                      key={cIdx}
                      className="px-3.5 py-2 text-slate-800 dark:text-slate-300"
                    >
                      {renderInline(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      inTable = false;
      tableRows = [];
    }

    // Empty lines
    if (!trimmed) {
      continue;
    }

    // Headers
    if (trimmed.startsWith("### ")) {
      const headerText = trimmed.replace("### ", "");
      const isExecSummary = headerText.includes("Executive Summary");
      const isHighlights =
        headerText.includes("Highlights") || headerText.includes("Findings");
      const isObservations =
        headerText.includes("Observations") || headerText.includes("Takeaways");
      const isRecommendations =
        headerText.includes("Recommendations") ||
        headerText.includes("Next Steps");

      let badgeColor =
        "bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-500/20";
      if (isExecSummary)
        badgeColor =
          "bg-indigo-50 dark:bg-indigo-500/10 text-indigo-700 dark:text-indigo-400 border-indigo-200 dark:border-indigo-500/20";
      else if (isHighlights)
        badgeColor =
          "bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/20";
      else if (isObservations)
        badgeColor =
          "bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-500/20";
      else if (isRecommendations)
        badgeColor =
          "bg-purple-50 dark:bg-purple-500/10 text-purple-700 dark:text-purple-400 border-purple-200 dark:border-purple-500/20";

      blocks.push(
        <div
          key={blockKey++}
          className="pt-3 pb-1.5 flex items-center gap-2 border-b border-slate-200 dark:border-slate-800/80 mb-2"
        >
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 dark:text-slate-200 flex items-center gap-2">
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-semibold border ${badgeColor}`}
            >
              {headerText}
            </span>
          </h4>
        </div>,
      );
      continue;
    }

    if (trimmed.startsWith("## ")) {
      blocks.push(
        <h3
          key={blockKey++}
          className="text-sm font-bold text-slate-900 dark:text-white pt-3 pb-1 border-b border-slate-200 dark:border-slate-800 flex items-center gap-2"
        >
          {trimmed.replace("## ", "")}
        </h3>,
      );
      continue;
    }

    if (trimmed.startsWith("# ")) {
      blocks.push(
        <h2
          key={blockKey++}
          className="text-base font-extrabold text-slate-900 dark:text-white pt-2 pb-1"
        >
          {trimmed.replace("# ", "")}
        </h2>,
      );
      continue;
    }

    // Blockquote
    if (trimmed.startsWith(">")) {
      blocks.push(
        <div
          key={blockKey++}
          className="my-2 p-3 rounded-xl bg-blue-50/70 dark:bg-blue-950/20 border-l-4 border-blue-500 text-xs text-slate-700 dark:text-slate-300 leading-relaxed"
        >
          {renderInline(trimmed.replace(/^>\s*/, ""))}
        </div>,
      );
      continue;
    }

    // Bullet points
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      const itemText = trimmed.replace(/^[-*]\s+/, "");
      blocks.push(
        <div
          key={blockKey++}
          className="flex items-start gap-2 py-1 text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed"
        >
          <span className="text-blue-500 dark:text-blue-400 font-bold select-none text-base leading-none">
            •
          </span>
          <span className="flex-1">{renderInline(itemText)}</span>
        </div>,
      );
      continue;
    }

    // Numbered lists
    const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      blocks.push(
        <div
          key={blockKey++}
          className="flex items-start gap-2 py-1 text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed"
        >
          <span className="flex-shrink-0 w-4 h-4 rounded-full bg-slate-100 dark:bg-slate-800 text-[10px] font-mono text-blue-600 dark:text-blue-400 font-bold flex items-center justify-center mt-0.5 border border-slate-300 dark:border-slate-700">
            {numMatch[1]}
          </span>
          <span className="flex-1">{renderInline(numMatch[2])}</span>
        </div>,
      );
      continue;
    }

    // Regular paragraph
    blocks.push(
      <p
        key={blockKey++}
        className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed my-1.5"
      >
        {renderInline(line)}
      </p>,
    );
  }

  return (
    <div className={`space-y-2 ${className}`}>
      {/* Top Visual KPI Metric Cards */}
      {kpis.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pb-3 mb-2 border-b border-slate-200 dark:border-slate-800">
          {kpis.map((kpi, idx) => (
            <div
              key={idx}
              className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800/90 bg-white dark:bg-gradient-to-br dark:from-slate-900/90 dark:to-slate-950/80 shadow-sm flex items-center justify-between gap-3 hover:border-blue-300 dark:hover:border-slate-700/80 transition-all"
            >
              <div className="space-y-0.5 min-w-0">
                <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 truncate uppercase tracking-wider">
                  {kpi.label}
                </p>
                <p className="text-base font-extrabold text-slate-900 dark:text-white tracking-tight">
                  {kpi.value}
                </p>
                {kpi.subtext && (
                  <p className="text-[10px] text-blue-600 dark:text-blue-400 font-medium truncate">
                    {kpi.subtext}
                  </p>
                )}
              </div>
              <div className="p-2 rounded-lg bg-blue-50 dark:bg-blue-500/10 border border-blue-200 dark:border-blue-500/20 text-blue-600 dark:text-blue-400 flex-shrink-0">
                {kpi.icon === "dollar" ? (
                  <DollarSign className="h-4 w-4" />
                ) : kpi.icon === "percent" ? (
                  <Percent className="h-4 w-4" />
                ) : kpi.icon === "award" ? (
                  <Award className="h-4 w-4 text-amber-500 dark:text-amber-400" />
                ) : (
                  <TrendingUp className="h-4 w-4" />
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Main Formatted Markdown Content */}
      <div className="space-y-1">{blocks}</div>
    </div>
  );
};
