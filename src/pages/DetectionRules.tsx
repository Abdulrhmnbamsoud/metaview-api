// src/pages/DetectionRules.tsx
// Proposed detection rules. They are hypotheses derived from evidence and are NOT tested until test_results exist.
import React, { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ExternalLink, FlaskConical } from 'lucide-react';
import { apiService } from '../services/api';
import { SaudiRulesResponse, SaudiRule } from '../types/saudi';
import {
  Panel,
  Chip,
  EpistemicChip,
  Unavailable,
  LoadingBlock,
  ErrorBlock,
  PageHeader,
  KpiTile,
  fmtDay,
} from '../components/ui';

const statusLabel: Record<string, string> = {
  proposed_untested: 'مقترحة — لم تُختبر',
  tested: 'مُختبرة',
};

const RuleCard: React.FC<{ r: SaudiRule }> = ({ r }) => {
  const [open, setOpen] = useState(false);
  const tested = r.test_results.length > 0;
  return (
    <div className="panel p-5 space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1.5 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="num-mono text-xs font-semibold text-brand">{r.rule_id}</span>
            <Chip tone={tested ? 'low' : 'medium'}>{statusLabel[r.status] || r.status}</Chip>
            {r.is_inference && <EpistemicChip kind="inference" label="استنتاج النظام" />}
            <Chip>{r.category_ar}</Chip>
          </div>
          <h3 className="text-base font-semibold text-ink">{r.name}</h3>
        </div>
        <button
          onClick={() => setOpen(!open)}
          className="px-3 py-1.5 rounded-lg bg-surface-container-high text-xs text-brand hover:bg-surface-bright transition-colors bg-surface-high"
        >
          {open ? 'إخفاء التفاصيل' : 'عرض المنطق والأدلة'}
        </button>
      </div>

      <p className="text-sm text-ink-variant leading-relaxed">
        <span className="label-mono ml-2">الفرضية</span>
        {r.hypothesis}
      </p>

      <div className="p-3 rounded-lg bg-surface-canvas font-mono text-xs text-brand leading-relaxed overflow-x-auto" dir="ltr">
        {r.logic}
      </div>

      <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
        <span className="label-mono">بيانات مطلوبة من البنك:</span>
        {r.required_data.map((d) => (
          <Chip key={d}>{d}</Chip>
        ))}
      </div>

      {open && (
        <div className="space-y-4 pt-3 border-t border-edge-subtle">
          <div className="text-xs text-ink-variant leading-relaxed">{r.evidence.note}</div>

          <div>
            <div className="label-mono mb-2">خطة التحقق</div>
            <ul className="space-y-1 text-[13px] text-ink-variant list-disc pr-5">
              {r.validation_plan.map((v, i) => (
                <li key={i}>{v}</li>
              ))}
            </ul>
          </div>

          <div>
            <div className="label-mono mb-2">الحوادث الداعمة ({r.evidence.incident_count})</div>
            {r.evidence.incidents.length === 0 ? (
              <Unavailable title="لا حوادث داعمة" note="القاعدة مبنية على تحذيرات فقط." />
            ) : (
              <div className="space-y-2">
                {r.evidence.incidents.map((i) => (
                  <div key={i.id} className="p-3 rounded-lg bg-surface-lowest space-y-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <Link to="/incident-registry" className="text-sm text-ink hover:text-brand">
                        {i.title}
                      </Link>
                      <span className="text-[11px] text-ink-outline">{fmtDay(i.first_reported_at)}</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {i.article_urls.slice(0, 4).map((u, idx) => (
                        <a key={u} href={u} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[11px] text-brand hover:underline">
                          <ExternalLink className="w-3 h-3" /> مصدر {idx + 1}
                        </a>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {r.evidence.warning_articles.length > 0 && (
            <div>
              <div className="label-mono mb-2">أخبار تحذير داعمة ({r.evidence.warning_article_count})</div>
              <div className="space-y-1">
                {r.evidence.warning_articles.map((w) => (
                  <a key={w.threat_id} href={w.url} target="_blank" rel="noreferrer" className="block text-[13px] text-ink-variant hover:text-brand">
                    • {w.title}
                  </a>
                ))}
              </div>
            </div>
          )}

          <div>
            <div className="label-mono mb-2">نتائج الاختبار</div>
            {tested ? (
              <pre className="p-3 rounded-lg bg-surface-lowest text-[11px] text-ink-variant overflow-x-auto" dir="ltr">
                {JSON.stringify(r.test_results, null, 2)}
              </pre>
            ) : (
              <Unavailable title="لم تُشغَّل اختبارات" note="لا تُعرض دقة أو إنذارات كاذبة قبل الاختبار على بيانات البنك الفعلية." />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export const DetectionRules: React.FC = () => {
  const [data, setData] = useState<SaudiRulesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await apiService.getSaudiRules());
    } catch (e: any) {
      setError(e?.message || 'خطأ غير معروف');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (loading && !data) return <LoadingBlock label="جارٍ تحميل القواعد…" />;
  if (error && !data) return <ErrorBlock message={error} onRetry={load} />;
  if (!data) return null;

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="طبقة الكشف (Detection)"
        title="قواعد الكشف المقترحة"
        subtitle="قواعد مشتقة من أدلة الحوادث والتحذيرات. تبقى فرضيات حتى تُختبر على بيانات حقيقية."
      />
      <div className="panel p-3 text-xs text-ink-variant leading-relaxed flex items-start gap-2">
        <FlaskConical className="w-4 h-4 text-threat-medium shrink-0 mt-0.5" />
        <span>{data.note}</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <KpiTile label="إجمالي القواعد" value={data.summary.total} tone="brand" />
        <KpiTile label="مُختبرة" value={data.summary.tested} tone={data.summary.tested ? 'low' : 'plain'} />
        <KpiTile label="مقترحة بلا اختبار" value={data.summary.proposed_untested} tone="medium" />
      </div>
      {data.rules.length === 0 ? (
        <Unavailable title="لا قواعد بعد" note="لم تُشتق قواعد من الحوادث المخزّنة." />
      ) : (
        <div className="space-y-4">
          {data.rules.map((r) => (
            <RuleCard key={r.rule_id} r={r} />
          ))}
        </div>
      )}
    </div>
  );
};

export default DetectionRules;
