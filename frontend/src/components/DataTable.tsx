import React, { useState } from 'react';
import { ArrowUpDown, ChevronLeft, ChevronRight, Search } from 'lucide-react';

interface DataTableProps {
  data: Record<string, any>[];
  columns?: string[];
  pageSize?: number;
  className?: string;
  showSearch?: boolean;
  emptyMessage?: string;
}

export const DataTable: React.FC<DataTableProps> = ({
  data,
  columns: explicitColumns,
  pageSize = 10,
  className = '',
  showSearch = true,
  emptyMessage = 'No data available',
}) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  if (!data || data.length === 0) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/30 p-8 text-center text-slate-500">
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

  const totalPages = Math.ceil(sortedData.length / pageSize) || 1;
  const startIndex = (currentPage - 1) * pageSize;
  const pageSlice = sortedData.slice(startIndex, startIndex + pageSize);

  return (
    <div className={`space-y-3 ${className}`}>
      {showSearch && (
        <div className="flex items-center justify-between gap-4">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
            <input
              type="text"
              placeholder="Search table rows..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              className="w-full bg-slate-900/70 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500"
            />
          </div>
          <div className="text-xs text-slate-400">
            Showing <span className="text-slate-200 font-medium">{Math.min(startIndex + 1, sortedData.length)}</span> -{' '}
            <span className="text-slate-200 font-medium">{Math.min(startIndex + pageSize, sortedData.length)}</span> of{' '}
            <span className="text-slate-200 font-medium">{sortedData.length}</span> rows
          </div>
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-900/40">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="bg-slate-800/60 text-[11px] uppercase tracking-wider text-slate-400 font-semibold sticky top-0 border-b border-slate-800">
            <tr>
              {columns.map((col) => (
                <th
                  key={col}
                  onClick={() => handleSort(col)}
                  className="px-4 py-3 cursor-pointer select-none hover:text-white transition-colors"
                >
                  <div className="flex items-center gap-1.5">
                    <span>{col}</span>
                    <ArrowUpDown className={`h-3 w-3 ${sortCol === col ? 'text-blue-400' : 'text-slate-600'}`} />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {pageSlice.map((row, rowIdx) => (
              <tr
                key={rowIdx}
                className="hover:bg-slate-800/30 transition-colors even:bg-slate-900/20"
              >
                {columns.map((col) => (
                  <td key={col} className="px-4 py-2.5 whitespace-nowrap text-slate-200">
                    {row[col] === null || row[col] === undefined ? (
                      <span className="text-slate-600 italic">null</span>
                    ) : typeof row[col] === 'boolean' ? (
                      <span
                        className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-mono ${
                          row[col] ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
                        }`}
                      >
                        {String(row[col])}
                      </span>
                    ) : typeof row[col] === 'number' ? (
                      <span className="font-mono">{row[col].toLocaleString()}</span>
                    ) : (
                      String(row[col])
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between pt-2 text-xs text-slate-400">
          <div>
            Page {currentPage} of {totalPages}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="p-1.5 rounded bg-slate-800/70 border border-slate-700 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-300"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="p-1.5 rounded bg-slate-800/70 border border-slate-700 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-300"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
