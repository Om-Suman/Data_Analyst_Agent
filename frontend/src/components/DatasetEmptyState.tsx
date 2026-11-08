import { ArrowRight, Database, FileText } from "lucide-react";
import { Link } from "react-router-dom";
import { useDataset } from "../context/DatasetContext";

export function DatasetEmptyState({
  title,
  description,
  document = false,
}: {
  title: string;
  description: string;
  document?: boolean;
}) {
  const { loading, datasets, setActiveDataset } = useDataset();
  const available = datasets.filter((dataset) => dataset.is_text === document);
  const resource = document ? "document" : "dataset";
  const Icon = document ? FileText : Database;

  return (
    <section className="space-y-6">
      <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
      {loading ? <p role="status">Loading your workspace…</p> : (
        <div className="mx-auto flex max-w-2xl flex-col items-center rounded-2xl border border-slate-200 bg-[var(--bg-card)] px-5 py-9 text-center sm:px-10 sm:py-12 dark:border-slate-800">
          <span className="grid h-12 w-12 place-items-center rounded-xl bg-blue-50 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
            <Icon className="h-6 w-6" aria-hidden="true" />
          </span>
          <h2 className="mt-4 text-xl font-semibold tracking-tight">
            {available.length ? `Choose a ${resource}` : `Start with a ${resource}`}
          </h2>
          <p className="mt-2 max-w-md text-sm leading-relaxed text-slate-600 dark:text-slate-400">{description}</p>
          {available.length > 0 ? (
            <select aria-label={`Select a ${resource}`} value="" onChange={(event) => void setActiveDataset(event.target.value)} className="mt-5 min-h-11 w-full max-w-xs rounded-xl border border-slate-300 bg-[var(--bg-card)] px-3 text-sm dark:border-slate-700">
              <option value="" disabled>Select an existing {resource}</option>
              {available.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}
            </select>
          ) : null}
          <div className="mt-4 flex flex-wrap items-center justify-center gap-x-4 gap-y-1">
            <Link to="/upload" className={available.length
              ? "inline-flex min-h-11 items-center text-sm font-semibold text-blue-700 underline underline-offset-4 dark:text-blue-300"
              : "inline-flex min-h-11 items-center gap-2 rounded-xl bg-blue-600 px-4 text-sm font-semibold text-white hover:bg-blue-700"}>
              {document ? "Upload a document" : "Upload a dataset"}
              {!available.length && <ArrowRight className="h-4 w-4" aria-hidden="true" />}
            </Link>
            {!document && !available.length && (
              <Link to="/upload#samples" className="inline-flex min-h-11 items-center text-sm font-semibold text-blue-700 underline underline-offset-4 dark:text-blue-300">
                Try a sample dataset
              </Link>
            )}
          </div>
          {!available.length && (
            <p className="mt-3 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
              {document ? "Searchable PDF, DOCX, TXT, or an image with text" : "CSV, Excel, JSON, or SQLite"}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
