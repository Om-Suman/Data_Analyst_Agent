import React, { useState } from 'react';
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  ChevronLeft,
  ChevronRight,
  Search,
  Download,
  SlidersHorizontal,
} from 'lucide-react';
import { useToast } from './Toast';

interface DataTableProps {
  data: Record<string, any>[];
  columns?: string[];
  pageSize?: number;
  className?: string;
  showSearch?: boolean;
  showExport?: boolean;
  emptyMessage?: string;
}

export const DataTable: React.FC<DataTableProps> = ({
  data,
  columns: explicitColumns,
  pageSize: initialPageSize = 10,
  className = '',
  showSearch = true,
  showExport = true,
  emptyMessage = 'No records found',
}) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(initialPageSize);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [density, setDensity] = useState<'compact' | 'standard' | 'relaxed'>('standard');

  const { success } = useToast();

  if (!data || data.length === 0) {
    return (
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white/60 dark:bg-slate-900/40 p-8 text-center text-slate-500 dark:text-slate-400 text-sm">
        {emptyMessage}
      </div>
    );
  }

  const columns = explicitColumns || Object.keys(data[0] || {});

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortCol(col);
      setSortDir('asc');
    }
  };

  const filteredData = data.filter((row) => {
    if (!searchTerm) return true;
    const s = searchTerm.toLowerCase();
    return Object.values(row).some((val) => String(val ?? '').toLowerCase().includes(s));
  });

  const sortedData = [...filteredData].sort((a, b) => {
    if (!sortCol) return 0;
    const aVal = a[sortCol];
    const bVal = b[sortCol];
    if (aVal === bVal) return 0;
    if (aVal === null || aVal === undefined) return 1;
    if (bVal === null || bVal === undefined) return -1;
    const res = aVal > bVal ? 1 : -1;
    return sortDir === 'asc' ? res : -res;
  });

  const totalPages = Math.max(1, Math.ceil(sortedData.length / pageSize));
  const validCurrentPage = Math.min(currentPage, totalPages);
  const startIdx = (validCurrentPage - 1) * pageSize;
  const pageData = sortedData.slice(startIdx, startIdx + pageSize);

  const handleExportCSV = () => {
    const header = columns.join(',');
    const rows = sortedData.map((row) =>
      columns
        .map((col) => {
          const val = row[col];
          if (val === null || val === undefined) return '';
          const str = String(val);
          return str.includes(',') || str.includes('"') || str.includes('\n')
            ? `"${str.replace(/"/g, '""')}"`
            : str;
        })
        .join(',')
    );
    const csvContent = [header, ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'filtered_data.csv');
    document.body.appendChild(link);
    link.click();
    link.remove();
    success('Export CSV', `Exported ${sortedData.length} records.`);
  };

  const densityPadding = {
    compact: 'py-2 px-3 text-xs',
    standard: 'py-3.5 px-4 text-sm',
    relaxed: 'py-5 px-5 text-base',
  }[density];

  return (
    <div className={`space-y-3.5 ${className}`}>
      {/* Control Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        {showSearch ? (
          <div className="relative w-full sm:w-80">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              type="text"
              placeholder={`Filter ${data.length} records across all columns...`}
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700/80 rounded-xl pl-9 pr-4 py-2 text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
            />
          </div>
        ) : <div />}

        <div className="flex items-center gap-2.5 w-full sm:w-auto justify-end">
          {/* Density Switcher */}
          <div className="flex items-center bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 rounded-lg p-0.5 text-xs">
            {(['compact', 'standard', 'relaxed'] as const).map((d) => (
              <button
                key={d}
                onClick={() => setDensity(d)}
                className={`px-2.5 py-1 rounded-md font-medium capitalize transition-all ${
                  density === d
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                {d}
              </button>
            ))}
          </div>

          {showExport && (
            <button
              onClick={handleExportCSV}
              title="Export filtered records to CSV"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 text-xs font-semibold transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
              CSV
            </button>
          )}
        </div>
      </div>

      {/* Table Container */}
      <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/50 shadow-sm">
        <table className="w-full text-left border-collapse">
          <thead className="bg-slate-50 dark:bg-slate-950/70 border-b border-slate-200 dark:border-slate-800 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            <tr>
              {columns.map((col) => {
                const isSorted = sortCol === col;
                return (
                  <th
                    key={col}
                    onClick={() => handleSort(col)}
                    className={`${densityPadding} cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-800/60 transition-colors select-none`}
                  >
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-slate-800 dark:text-slate-200">{col}</span>
                      {isSorted ? (
                        sortDir === 'asc' ? (
                          <ArrowUp className="h-3.5 w-3.5 text-blue-500" />
                        ) : (
                          <ArrowDown className="h-3.5 w-3.5 text-blue-500" />
                        )
                      ) : (
                        <ArrowUpDown className="h-3.5 w-3.5 opacity-30 group-hover:opacity-70" />
                      )}
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-slate-800/70 text-slate-800 dark:text-slate-200">
            {pageData.map((row, rowIdx) => (
              <tr
                key={rowIdx}
                className="hover:bg-blue-50/50 dark:hover:bg-slate-800/40 transition-colors"
              >
                {columns.map((col) => {
                  const val = row[col];
                  return (
                    <td key={col} className={`${densityPadding} whitespace-nowrap`}>
                      {val === null || val === undefined ? (
                        <span className="text-slate-400 dark:text-slate-600 font-mono text-xs">null</span>
                      ) : typeof val === 'boolean' ? (
                        <span
                          className={`px-2 py-0.5 rounded text-xs font-semibold ${
                            val
                              ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
                              : 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300'
                          }`}
                        >
                          {String(val)}
                        </span>
                      ) : typeof val === 'number' ? (
                        <span className="font-mono">{val.toLocaleString()}</span>
                      ) : (
                        <span>{String(val)}</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500 dark:text-slate-400 pt-1">
        <div className="flex items-center gap-2">
          <span>
            Showing <span className="font-bold text-slate-800 dark:text-slate-200">{startIdx + 1}</span> to{' '}
            <span className="font-bold text-slate-800 dark:text-slate-200">
              {Math.min(startIdx + pageSize, sortedData.length)}
            </span>{' '}
            of <span className="font-bold text-slate-800 dark:text-slate-200">{sortedData.length}</span> rows
          </span>

          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setCurrentPage(1);
            }}
            className="ml-2 bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 rounded px-2 py-1 text-slate-700 dark:text-slate-300"
          >
            {[10, 25, 50, 100].map((size) => (
              <option key={size} value={size}>
                {size} / page
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={validCurrentPage === 1}
            className="p-1.5 rounded-lg border border-slate-300 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-40 disabled:pointer-events-none transition-colors"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="px-2 font-medium">
            Page {validCurrentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={validCurrentPage === totalPages}
            className="p-1.5 rounded-lg border border-slate-300 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-40 disabled:pointer-events-none transition-colors"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
