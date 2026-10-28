// src/pages/FraudPatterns.tsx
// Fraud Pattern Explorer: patterns extracted from incidents/warnings, with the evidence behind each one.
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { ExternalLink, Network, ListChecks, Quote } from 'lucide-react';
import { apiService } from '../services/api';
import { SaudiPattern, SaudiPatternsResponse, SaudiRule } from '../types/saudi';
import {
  Panel,
  Chip,
  EpistemicChip,
  Unavailable,
  LoadingBlock,
  ErrorBlock,
  PageHeader,
  fmtDay,
} from '../components/ui';

export const FraudPatterns: React.FC = () => {
  const [data, setData] = useState<SaudiPatternsResponse | null>(null);
  const [rules, setRules] = useState<SaudiRule[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [q, setQ] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [p, r] = await Promise.all([apiService.getSaudiPatterns(), apiService.getSaudiRules()]);
      setData(p);
      setRules(r.rules);
      setSelected((cur) => cur || p.patterns[0]?.key || null);
    } catch (e: any) {
      setError(e?.message || 'خطأ غير معروف');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const handleRefresh = () => load();
    window.addEventListener('metaview:refresh', handleRefresh);
    return () => {
      window.removeEventListener('metaview:refresh', handleRefresh);
    };
  }, [load]);

  const filtered = useMemo(() => {
    if (!data) return [];
    const t = q.trim();
    return t ? data.patterns.filter((p) => p.label_ar.includes(t) || p.key.includes(t)) : data.patterns;
  }, [data, q]);

  if (loading && !data) return <LoadingBlock label="جارٍ تحميل الأنماط…" />;
  if (error && !data) return <ErrorBlock message={error} onRetry={load} />;
  if (!data) return null;

  const pat: SaudiPattern | undefined = data.patterns.find((p) => p.key === selected);
  const linkedRules = pat ? rules.filter((r) => r.evidence.patterns.some((x) => x.key === pat.key)) : [];
  const totalChannelMentions = data.channels.reduce((a, c) => a + c.count, 0) || 1;

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="سجل الأنماط (Topology Registry)"
        title="مستكشف أنماط الاحتيال"
        subtitle="النمط أسلوب احتيالي مستخرج من حادثة أو أكثر. كل نمط يعرض أدلته من نصوص المصادر، والقواعد المقترحة المرتبطة به."
        right={
          <>
            <Chip tone="brand">{data.patterns.length} نمط</Chip>
            <Chip>{rules.length} قاعدة مقترحة</Chip>
          </>
        }
      />

      {data.note && (
        <div className="panel p-3 text-xs text-ink-variant leading-relaxed">{data.note}</div>
      )}

      {data.patterns.length === 0 ? (
        <Unavailable title="لا أنماط بعد" note="لم تُستخرج أنماط من الحوادث والتحذيرات المخزّنة حتى الآن." />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Selector */}
          <div className="lg:col-span-4 space-y-3">
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="ابحث في الأنماط…"
              className="w-full bg-surface-lowest border border-edge-strong rounded-lg px-3 py-2 text-sm text-ink placeholder:text-ink-outline outline-none focus:border-brand"
            />
            <div className="space-y-2">
              {filtered.map((p, idx) => {
                const active = p.key === selected;
                return (
                  <button
                    key={`${p.key}-${idx}`}
                    onClick={() => setSelected(p.key)}
                    className={`w-full text-right p-3 rounded-lg transition-colors border ${
                      active ? 'bg-surface-high border-brand/50' : 'bg-surface-container border-transparent hover:bg-surface-high'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-semibold text-ink">{p.label_ar}</span>
                      <span className="num-mono text-[11px] text-brand">{p.incident_count} حادثة</span>
                    </div>
                    <div className="flex items-center gap-1.5 mt-1.5">
                      <Chip>{p.warning_article_count} تحذير</Chip>
                      {p.last_seen && <span className="text-[10px] text-ink-outline">آخر ظهور {fmtDay(p.last_seen)}</span>}
                    </div>
                  </button>
                );
              })}
              {filtered.length === 0 && <Unavailable title="لا نتائج" note="لا نمط يطابق البحث." />}
            </div>
          </div>

          {/* Dossier */}
          <div className="lg:col-span-8 space-y-6">
            {pat && (
              <>
                <div className="panel p-6 space-y-4 relative overflow-hidden">
                  <div className="absolute -left-10 -top-10 w-56 h-56 bg-brand/5 rounded-full blur-3xl pointer-events-none" />
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="num-mono text-xs px-2 py-0.5 rounded bg-brand-container text-surface-canvas font-semibold">{pat.key}</span>
                    <EpistemicChip kind="fact" label={`الأساس: ${pat.support}`} />
                    <Chip>{pat.origin}</Chip>
                  </div>
                  <h2 className="text-xl font-semibold text-ink">{pat.label_ar}</h2>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-edge-subtle">
                    <div>
                      <div className="label-mono">حوادث مرتبطة</div>
                      <div className="num-mono text-2xl font-bold text-ink">{pat.incident_count}</div>
                    </div>
                    <div>
                      <div className="label-mono">أخبار تحذير</div>
                      <div className="num-mono text-2xl font-bold text-ink">{pat.warning_article_count}</div>
                    </div>
                    <div>
                      <div className="label-mono">أول ظهور</div>
                      <div className="text-sm text-ink mt-2">{fmtDay(pat.first_seen)}</div>
                    </div>
                    <div>
                      <div className="label-mono">آخر ظهور</div>
                      <div className="text-sm text-ink mt-2">{fmtDay(pat.last_seen)}</div>
                    </div>
                  </div>
                  {pat.channels.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="label-mono">القنوات:</span>
                      {pat.channels.map((c) => (
                        <Chip key={c}>{c}</Chip>
                      ))}
                    </div>
                  )}
                  {pat.categories.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="label-mono">التصنيفات:</span>
                      {pat.categories.map((c) => (
                        <Chip key={c.label_ar} tone="brand">
                          {c.label_ar} ({c.incidents})
                        </Chip>
                      ))}
                    </div>
                  )}
                </div>

                <Panel title="الحوادث المرتبطة" icon={<Network className="w-4 h-4" />}>
                  {pat.incidents.length === 0 ? (
                    <Unavailable title="لا حوادث مرتبطة" note="النمط مدعوم بتحذيرات فقط." />
                  ) : (
                    <div className="space-y-2">
                      {pat.incidents.map((i) => (
                        <Link
                          key={i.id}
                          to="/incident-registry"
                          className="p-3 rounded-lg bg-surface-container hover:bg-surface-high transition-colors flex items-center justify-between gap-3"
                        >
                          <span className="text-sm text-ink leading-relaxed">{i.title}</span>
                          <span className="num-mono text-[11px] text-ink-variant shrink-0">{i.article_count} خبر</span>
                        </Link>
                      ))}
                    </div>
                  )}
                </Panel>

                <Panel
                  title="الأدلة من نصوص المصادر"
                  subtitle="اقتباسات حرفية؛ لا تُضاف أي معلومة خارج النص."
                  icon={<Quote className="w-4 h-4" />}
                >
                  {pat.evidence.length === 0 ? (
                    <Unavailable title="لا اقتباسات" />
                  ) : (
                    <div className="space-y-3">
                      {pat.evidence.map((e, idx) => (
                        <div key={idx} className="p-3 rounded-lg bg-surface-lowest space-y-2">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-semibold text-ink">{e.title}</span>
                            <Chip>{e.type}</Chip>
                          </div>
                          <p className="text-[13px] text-ink-variant leading-relaxed border-r-2 border-brand/50 pr-3">«{e.quote}»</p>
                          <a
                            href={e.url}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1 text-[11px] text-brand hover:underline"
                          >
                            <ExternalLink className="w-3 h-3" /> {e.source}
                          </a>
                        </div>
                      ))}
                    </div>
                  )}
                </Panel>

                <Panel
                  title="قواعد الكشف المقترحة لهذا النمط"
                  icon={<ListChecks className="w-4 h-4" />}
                >
                  {linkedRules.length === 0 ? (
                    <Unavailable title="لا قواعد مرتبطة" note="لم تُشتق قاعدة مقترحة لهذا النمط." />
                  ) : (
                    <div className="space-y-2">
                      {linkedRules.map((r) => (
                        <div key={r.rule_id} className="p-3 rounded-lg bg-surface-container space-y-1.5">
                          <div className="flex items-center gap-2">
                            <span className="num-mono text-xs text-brand font-semibold">{r.rule_id}</span>
                            <EpistemicChip kind="hypothesis" label="فرضية — لم تُختبر" />
                          </div>
                          <div className="text-sm text-ink">{r.name}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </Panel>
              </>
            )}
          </div>
        </div>
      )}

      {/* Channel matrix */}
      {data.channels.length > 0 && (
        <Panel title="توزيع القنوات المذكورة في المصادر" subtitle="عدد مرات ذكر القناة في الأخبار المصنّفة كاحتيال.">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3">
            {data.channels.map((c) => (
              <div key={c.channel} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-ink">{c.channel}</span>
                  <span className="num-mono text-ink-variant">{c.count}</span>
                </div>
                <div className="w-full bg-surface-container h-1.5 rounded-full overflow-hidden">
                  <div className="bg-brand h-full rounded-full" style={{ width: `${(c.count / totalChannelMentions) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
};

export default FraudPatterns;
