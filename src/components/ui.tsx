// src/components/ui.tsx
// Shared command-center primitives. All values passed in must come from real API data.
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
  title?: string;
  subtitle?: string;
  icon?: React.ReactNode;
  right?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}> = ({ title, subtitle, icon, right, className = '', children }) => (
  <section className={`panel p-5 space-y-4 ${className}`}>
    {(title || right) && (
      <header className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {icon && <span className="text-brand">{icon}</span>}
            {title && <h3 className="text-base font-semibold text-ink">{title}</h3>}
          </div>
          {subtitle && <p className="text-xs text-ink-variant mt-0.5 leading-relaxed">{subtitle}</p>}
        </div>
        {right}
      </header>
    )}
    {children}
  </section>
);

const toneText: Record<string, string> = {
  brand: 'text-brand',
  critical: 'text-threat-critical',
  high: 'text-threat-high',
  medium: 'text-threat-medium',
  low: 'text-threat-low',
  plain: 'text-ink',
};
const toneBar: Record<string, string> = {
  brand: 'bg-brand',
  critical: 'bg-threat-critical',
  high: 'bg-threat-high',
  medium: 'bg-threat-medium',
  low: 'bg-threat-low',
  plain: 'bg-surface-bright',
};

export type Tone = keyof typeof toneText;

export const KpiTile: React.FC<{
  label: string;
  value: React.ReactNode;
  hint?: string;
  icon?: React.ReactNode;
  tone?: Tone;
  /** 0..100, only render when the ratio is a real computed ratio */
  ratio?: number | null;
  to?: string;
}> = ({ label, value, hint, icon, tone = 'plain', ratio, to }) => {
  const body = (
    <div className="panel p-4 flex flex-col justify-between h-full hover:bg-surface-container transition-colors">
      <div className="flex items-center justify-between text-ink-outline">
        <span className="label-mono">{label}</span>
        {icon && <span className="text-brand">{icon}</span>}
      </div>
      <div className="my-3">
        <div className={`num-mono text-3xl font-bold tracking-tight ${toneText[tone]}`}>{value}</div>
        {hint && <div className="text-xs text-ink-variant mt-1">{hint}</div>}
      </div>
      {ratio != null && (
        <div className="w-full bg-surface-high h-1.5 rounded-full overflow-hidden">
          <div className={`${toneBar[tone]} h-full rounded-full`} style={{ width: `${Math.max(0, Math.min(100, ratio))}%` }} />
        </div>
      )}
    </div>
  );
  return to ? <Link to={to} className="block">{body}</Link> : body;
};

export type Epistemic = 'fact' | 'inference' | 'hypothesis' | 'unknown';

const epiStyle: Record<Epistemic, { cls: string; label: string }> = {
  fact: { cls: 'bg-epistemic-fact/10 text-epistemic-fact', label: 'حقيقة موثّقة' },
  inference: { cls: 'bg-epistemic-inference/10 text-epistemic-inference', label: 'استنتاج' },
  hypothesis: { cls: 'bg-epistemic-hypothesis/10 text-epistemic-hypothesis', label: 'فرضية' },
  unknown: { cls: 'bg-epistemic-unknown/15 text-epistemic-unknown', label: 'غير محسوم' },
};

export const EpistemicChip: React.FC<{ kind: Epistemic; label?: string }> = ({ kind, label }) => (
  <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${epiStyle[kind].cls}`}>
    <span className="w-1.5 h-1.5 rounded-full bg-current" />
    {label || epiStyle[kind].label}
  </span>
);

export const Chip: React.FC<{ tone?: Tone | 'neutral'; children: React.ReactNode }> = ({ tone = 'neutral', children }) => {
  const cls =
    tone === 'neutral'
      ? 'bg-surface-container text-ink-variant'
      : tone === 'brand'
      ? 'bg-brand/10 text-brand'
      : tone === 'critical'
      ? 'bg-threat-critical/15 text-threat-critical'
      : tone === 'high'
      ? 'bg-threat-high/15 text-threat-high'
      : tone === 'medium'
      ? 'bg-threat-medium/15 text-threat-medium'
      : tone === 'low'
      ? 'bg-threat-low/10 text-threat-low'
      : 'bg-surface-container text-ink';
  return <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono ${cls}`}>{children}</span>;
};

export const Unavailable: React.FC<{ title?: string; note?: string }> = ({
  title = 'غير متوفر بعد',
  note = 'لا يوجد مصدر بيانات حقيقي لهذا القسم حتى الآن، لذلك لا نعرض أرقامًا مختلقة.',
}) => (
  <div className="rounded-lg bg-surface-lowest border border-dashed border-edge-strong p-4 text-center">
    <div className="text-sm font-semibold text-ink-variant">{title}</div>
    <div className="text-xs text-ink-outline mt-1 leading-relaxed">{note}</div>
  </div>
);

export const LoadingBlock: React.FC<{ label?: string }> = ({ label = 'جارٍ التحميل…' }) => (
  <div className="panel p-8 text-center text-sm text-ink-variant animate-pulse">{label}</div>
);

export const ErrorBlock: React.FC<{ message: string; onRetry?: () => void }> = ({ message, onRetry }) => (
  <div className="panel p-6 text-center border-threat-critical/40">
    <div className="text-sm text-threat-critical font-semibold">تعذّر تحميل البيانات</div>
    <div className="text-xs text-ink-variant mt-1">{message}</div>
    {onRetry && (
      <button onClick={onRetry} className="mt-3 px-3 py-1.5 rounded-lg bg-surface-container-high text-brand text-xs">
        إعادة المحاولة
      </button>
    )}
  </div>
);

export const PageHeader: React.FC<{
  kicker: string;
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
}> = ({ kicker, title, subtitle, right }) => (
  <div className="flex flex-col xl:flex-row xl:items-end justify-between gap-4">
    <div>
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-brand/10 text-brand label-mono !text-brand">
        <span className="w-1.5 h-1.5 rounded-full bg-brand animate-pulse" />
        {kicker}
      </span>
      <h1 className="text-2xl font-semibold tracking-tight text-ink mt-2">{title}</h1>
      {subtitle && <p className="text-sm text-ink-variant mt-1 max-w-3xl leading-relaxed">{subtitle}</p>}
    </div>
    {right && <div className="flex items-center gap-2 flex-wrap">{right}</div>}
  </div>
);
