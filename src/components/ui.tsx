// src/components/ui.tsx
// Information-first minimalist primitives (Linear & Notion style)
import React from 'react';
import { Link } from 'react-router-dom';

export const fmtDateTime = (s?: string | null): string => {
  if (!s) return '—';
  const d = new Date(s);
  if (isNaN(d.getTime())) return s;
  return d.toLocaleString('ar-SA-u-nu-latn', { dateStyle: 'medium', timeStyle: 'short' });
};

export const fmtDay = (s?: string | null): string => {
  if (!s) return '—';
  const d = new Date(s);
  if (isNaN(d.getTime())) return s;
  return d.toLocaleDateString('ar-SA-u-nu-latn', { dateStyle: 'medium' });
};

export const Panel: React.FC<{
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  icon?: React.ReactNode;
  right?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}> = ({ title, subtitle, icon, right, className = '', children }) => (
  <section className={`panel p-5 space-y-4 ${className}`}>
    {(title || right) && (
      <header className="flex items-start justify-between gap-3 pb-1 border-b border-stone-100 dark:border-slate-800/60">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {icon && <span className="text-sky-600 dark:text-sky-400">{icon}</span>}
            {title && <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100">{title}</h3>}
          </div>
          {subtitle && <p className="text-xs text-stone-500 dark:text-slate-400 mt-0.5 leading-relaxed">{subtitle}</p>}
        </div>
        {right}
      </header>
    )}
    {children}
  </section>
);

export type Tone = 'brand' | 'critical' | 'high' | 'medium' | 'low' | 'plain' | 'neutral';

const toneClasses: Record<Tone, string> = {
  brand: 'bg-sky-50 text-sky-700 border-sky-200 dark:bg-sky-950/40 dark:text-sky-300 dark:border-sky-800/60',
  critical: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950/40 dark:text-red-300 dark:border-red-800/60',
  high: 'bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950/40 dark:text-orange-300 dark:border-orange-800/60',
  medium: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-800/60',
  low: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-800/60',
  plain: 'bg-stone-50 text-stone-700 border-stone-200 dark:bg-slate-800/60 dark:text-stone-300 dark:border-slate-700',
  neutral: 'bg-stone-100 text-stone-600 border-stone-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700',
};

export const Chip: React.FC<{ tone?: Tone; className?: string; children: React.ReactNode }> = ({
  tone = 'neutral',
  className = '',
  children,
}) => (
  <span
    className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-mono border font-medium ${toneClasses[tone]} ${className}`}
  >
    {children}
  </span>
);

export type Epistemic = 'fact' | 'inference' | 'hypothesis' | 'unknown';

const epiStyle: Record<Epistemic, { cls: string; label: string }> = {
  fact: {
    cls: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-800/60',
    label: 'حقيقة موثّقة',
  },
  inference: {
    cls: 'bg-sky-50 text-sky-700 border-sky-200 dark:bg-sky-950/40 dark:text-sky-300 dark:border-sky-800/60',
    label: 'استنتاج تحليلي',
  },
  hypothesis: {
    cls: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-800/60',
    label: 'فرضية مقترحة',
  },
  unknown: {
    cls: 'bg-stone-100 text-stone-600 border-stone-200 dark:bg-slate-800 dark:text-slate-400 dark:border-slate-700',
    label: 'غير محسوم',
  },
};

export const EpistemicChip: React.FC<{ kind: Epistemic; label?: string }> = ({ kind, label }) => (
  <span
    className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-mono border font-medium ${epiStyle[kind].cls}`}
  >
    <span className="w-1.5 h-1.5 rounded-full bg-current opacity-80" />
    {label || epiStyle[kind].label}
  </span>
);

export const KpiTile: React.FC<{
  label: string;
  value: React.ReactNode;
  hint?: string;
  icon?: React.ReactNode;
  tone?: Tone;
  to?: string;
}> = ({ label, value, hint, icon, tone = 'plain', to }) => {
  const content = (
    <div className="panel p-4 flex flex-col justify-between hover:border-stone-300 dark:hover:border-slate-700 transition-colors">
      <div className="flex items-center justify-between text-xs text-stone-500 dark:text-slate-400">
        <span>{label}</span>
        {icon && <span className="text-sky-600 dark:text-sky-400">{icon}</span>}
      </div>
      <div className="my-2">
        <div className="num-mono text-2xl font-bold text-stone-900 dark:text-stone-100">{value}</div>
        {hint && <div className="text-[11px] text-stone-500 dark:text-slate-400 mt-0.5">{hint}</div>}
      </div>
    </div>
  );
  return to ? <Link to={to}>{content}</Link> : content;
};

export const PageHeader: React.FC<{
  kicker?: string;
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
}> = ({ kicker, title, subtitle, right }) => (
  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-stone-200 dark:border-slate-800">
    <div>
      {kicker && (
        <span className="text-xs font-mono text-sky-600 dark:text-sky-400 font-semibold block mb-0.5">
          {kicker}
        </span>
      )}
      <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-stone-900 dark:text-stone-100">{title}</h1>
      {subtitle && <p className="text-xs text-stone-600 dark:text-slate-400 mt-1 max-w-2xl">{subtitle}</p>}
    </div>
    {right && <div className="flex items-center gap-2">{right}</div>}
  </div>
);

export const Unavailable: React.FC<{
  title?: string;
  note?: string;
  className?: string;
}> = ({
  title = 'لا تتوفر معلومات كافية',
  note = 'لا توجد بيانات موثقة في المصادر المتاحة حالياً.',
  className = '',
}) => (
  <div
    className={`rounded-lg bg-stone-50 dark:bg-slate-900/50 border border-dashed border-stone-300 dark:border-slate-800 p-4 text-center ${className}`}
  >
    <div className="text-xs font-semibold text-stone-600 dark:text-slate-300">{title}</div>
    {note && <div className="text-[11px] text-stone-500 dark:text-slate-500 mt-1 leading-relaxed">{note}</div>}
  </div>
);

export const LoadingBlock: React.FC<{ label?: string }> = ({ label = 'جارٍ التحميل…' }) => (
  <div className="panel p-8 text-center text-xs text-stone-500 dark:text-slate-400 animate-pulse font-mono">
    {label}
  </div>
);

export const ErrorBlock: React.FC<{ message: string; onRetry?: () => void }> = ({ message, onRetry }) => (
  <div className="panel p-6 text-center border-red-200 dark:border-red-900/40 bg-red-50/30 dark:bg-red-950/10">
    <div className="text-xs font-semibold text-red-600 dark:text-red-400">تعذّر تحميل البيانات</div>
    <div className="text-xs text-stone-600 dark:text-slate-400 mt-1">{message}</div>
    {onRetry && (
      <button
        onClick={onRetry}
        className="mt-3 px-3 py-1.5 rounded-lg bg-white dark:bg-slate-800 text-stone-800 dark:text-stone-200 border border-stone-200 dark:border-slate-700 text-xs font-medium hover:bg-stone-50 dark:hover:bg-slate-700 transition-colors"
      >
        إعادة المحاولة
      </button>
    )}
  </div>
);
