// src/pages/ExecutiveOverview.tsx
// Executive intelligence overview. Every figure is read from the backend; nothing is invented.
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ShieldCheck,
  Network,
  Landmark,
  ListChecks,
  Radio,
  FileDown,
  Eye,
  AlertTriangle,
  Clock,
} from 'lucide-react';
import { apiService } from '../services/api';
import {
  SaudiOverview,
  SaudiIncidentReport,
  SaudiIncident,
  SaudiPatternsResponse,
  SaudiRulesResponse,
  SaudiSourceInfo,
  SaudiFraudItem,
} from '../types/saudi';
import {
  Panel,
  KpiTile,
  Chip,
  EpistemicChip,
  Unavailable,
  LoadingBlock,
  ErrorBlock,
  PageHeader,
  fmtDateTime,
} from '../components/ui';
import { timeAgo } from '../utils/formatDate';

const CT_LABEL: Record<string, string> = {
  confirmed_incident: 'حوادث مؤكدة',
  reported_incident: 'حوادث مبلَّغ عنها',
  fraud_warning: 'تحذيرات احتيالية',
  awareness_campaign: 'حملات توعية',
  general_cybersecurity: 'أمن سيبراني عام (مستبعد)',
  unrelated: 'غير ذي صلة (مستبعد)',
};
const CT_TONE: Record<string, string> = {
  confirmed_incident: 'bg-threat-critical',
  reported_incident: 'bg-threat-high',
  fraud_warning: 'bg-threat-medium',
  awareness_campaign: 'bg-brand',
  general_cybersecurity: 'bg-epistemic-unknown',
  unrelated: 'bg-epistemic-unknown',
};

const dayKey = (s: string) => s.slice(0, 10);

export const ExecutiveOverview: React.FC = () => {
  const [ov, setOv] = useState<SaudiOverview | null>(null);
  const [report, setReport] = useState<SaudiIncidentReport | null>(null);
  const [incidents, setIncidents] = useState<SaudiIncident[]>([]);
  const [patterns, setPatterns] = useState<SaudiPatternsResponse | null>(null);
  const [rules, setRules] = useState<SaudiRulesResponse | null>(null);
  const [sources, setSources] = useState<SaudiSourceInfo[]>([]);
  const [news, setNews] = useState<SaudiFraudItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [o, r, inc, p, ru, so, n] = await Promise.all([
        apiService.getSaudiOverview(),
        apiService.getSaudiIncidentReport(),
        apiService.getSaudiIncidents({ page_size: 5 }),
        apiService.getSaudiPatterns(),
        apiService.getSaudiRules(),
        apiService.getSaudiSources(),
        apiService.getSaudiNews({ page_size: 100 }),
      ]);
      setOv(o);
      setReport(r);
      setIncidents(inc.items);
      setPatterns(p);
      setRules(ru);
      setSources(so.sources);
      setNews(n.items);
    } catch (e: any) {
      setError(e?.message || 'خطأ غير معروف');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const h = () => load();
    window.addEventListener('metaview:refresh', h);
    return () => window.removeEventListener('metaview:refresh', h);
  }, [load]);

  // Real timeline: stored Saudi fraud articles per publication day.
  const timeline = useMemo(() => {
    const m = new Map<string, number>();
    news.forEach((i) => {
      const d = i.published_at || i.fetched_at;
      if (d) m.set(dayKey(d), (m.get(dayKey(d)) || 0) + 1);
    });
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [news]);

  if (loading && !ov) return <LoadingBlock label="جارٍ تحميل النظرة التنفيذية…" />;
  if (error && !ov) return <ErrorBlock message={error} onRetry={load} />;
  if (!ov || !report) return null;

  const maxDay = Math.max(1, ...timeline.map(([, c]) => c));
  const byType = report.articles.by_content_type || {};
  const typeTotal = Object.values(byType).reduce((a, b) => a + b, 0) || 1;
  const sourcesBad = sources.filter((s) => s.last_status && s.last_status !== 'ok');
  const sourcesOkRatio = ov.sources_total ? (ov.sources_ok / ov.sources_total) * 100 : null;
  const incRatio = report.incidents.independent_total
    ? (report.incidents.confirmed / report.incidents.independent_total) * 100
    : null;

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="القيادة التنفيذية • السوق المالي السعودي"
        title="النظرة التنفيذية لاستخبارات الاحتيال"
        subtitle="ملخّص حيّ مبني بالكامل على بيانات المنصة: الأخبار الأصلية، الحوادث المستقلة، الأنماط، وقواعد الكشف المقترحة. لا يُعرض أي رقم غير موجود في قاعدة البيانات."
        right={
          <>
            <Chip tone="low">حالة البيانات: مباشرة</Chip>
            <a
              href={`${apiService.getBaseUrl()}/saudi-fraud/incidents/report?format=markdown`}
              target="_blank"
              rel="noreferrer"
              className="px-3 py-1.5 rounded-lg bg-brand text-brand-on text-sm font-semibold flex items-center gap-1.5 hover:bg-brand-container transition-colors"
            >
              <FileDown className="w-4 h-4" />
              <span>تقرير الحوادث</span>
            </a>
          </>
        }
      />

      {/* KPI tiles */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <KpiTile
          label="حوادث مستقلة"
          value={report.incidents.independent_total}
          hint={`${report.incidents.confirmed} مؤكدة • ${report.incidents.reported} مبلَّغ عنها`}
          icon={<ShieldCheck className="w-4 h-4" />}
          tone="brand"
          ratio={incRatio}
          to="/incident-registry"
        />
        <KpiTile
          label="أخبار خلف الحوادث"
          value={report.incidents.articles_behind_incidents}
          hint={`${report.incidents.multi_article} حادثة متعددة المصادر`}
          icon={<Landmark className="w-4 h-4" />}
          tone="plain"
          to="/incident-registry"
        />
        <KpiTile
          label="أنماط مستخرجة"
          value={patterns ? patterns.patterns.length : '—'}
          hint="من حوادث وتحذيرات موثّقة"
          icon={<Network className="w-4 h-4" />}
          tone="high"
          to="/patterns"
        />
        <KpiTile
          label="قواعد كشف مقترحة"
          value={rules ? rules.summary.total : '—'}
          hint={rules ? `${rules.summary.tested} مُختبرة • ${rules.summary.proposed_untested} بلا اختبار` : undefined}
          icon={<ListChecks className="w-4 h-4" />}
          tone="brand"
          to="/rules"
        />
        <KpiTile
          label="مصادر الرصد"
          value={`${ov.sources_ok}/${ov.sources_total}`}
          hint={ov.last_fetch_at ? `آخر جلب ${timeAgo(ov.last_fetch_at)}` : 'لم يُجلب بعد'}
          icon={<Radio className="w-4 h-4" />}
          tone={sourcesBad.length ? 'medium' : 'low'}
          ratio={sourcesOkRatio}
          to="/sources"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 space-y-6">
          {/* Timeline */}
          <Panel
            title="الأخبار الاحتيالية السعودية حسب يوم النشر"
            subtitle={`آخر ${news.length} خبرًا مخزّنًا — تاريخ النشر كما ورد من المصدر.`}
            icon={<Clock className="w-4 h-4" />}
          >
            {timeline.length === 0 ? (
              <Unavailable title="لا أخبار مخزّنة" note="لم تُسجَّل أخبار احتيال سعودية بعد." />
            ) : (
              <div className="rounded-lg bg-surface-lowest p-4">
                <div className="flex items-end gap-1.5 h-40" dir="ltr">
                  {timeline.map(([d, c]) => (
                    <div key={d} className="flex-1 min-w-[10px] flex flex-col items-center justify-end gap-1 group" title={`${d}: ${c}`}>
                      <span className="text-[9px] font-mono text-ink-outline opacity-0 group-hover:opacity-100">{c}</span>
                      <div
                        className="w-full rounded-t bg-brand/70 group-hover:bg-brand transition-colors"
                        style={{ height: `${Math.max(4, (c / maxDay) * 100)}%` }}
                      />
                    </div>
                  ))}
                </div>
                <div className="flex justify-between text-[10px] font-mono text-ink-outline mt-2" dir="ltr">
                  <span>{timeline[0][0]}</span>
                  <span>{timeline[timeline.length - 1][0]}</span>
                </div>
              </div>
            )}
          </Panel>

          {/* Content type breakdown */}
          <Panel
            title="توزيع الأخبار حسب نوع المحتوى"
            subtitle="الفصل بين الخبر والحادثة والتحذير والتوعية — الحادثة قد تغطيها عدة أخبار."
            icon={<AlertTriangle className="w-4 h-4" />}
          >
            <div className="space-y-3">
              {Object.entries(byType)
                .filter(([, c]) => c > 0)
                .sort((a, b) => b[1] - a[1])
                .map(([k, c]) => (
                  <div key={k} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-ink">{CT_LABEL[k] || k}</span>
                      <span className="num-mono text-ink-variant">{c}</span>
                    </div>
                    <div className="w-full bg-surface-container h-2 rounded-full overflow-hidden">
                      <div className={`${CT_TONE[k] || 'bg-surface-bright'} h-full rounded-full`} style={{ width: `${(c / typeTotal) * 100}%` }} />
                    </div>
                  </div>
                ))}
            </div>
          </Panel>

          {/* Priority incidents */}
          <Panel
            title="أحدث الحوادث المستقلة"
            subtitle="كل حادثة تجمع الأخبار التي تتناول الواقعة نفسها بناءً على الوقائع (الجهة، الموقع، العدد)."
            right={<Link to="/incident-registry" className="text-xs text-brand hover:underline">سجل الحوادث الكامل ←</Link>}
          >
            {incidents.length === 0 ? (
              <Unavailable title="لا حوادث بعد" note="لم تُستخرج حوادث مستقلة من الأخبار المخزّنة." />
            ) : (
              <div className="space-y-3">
                {incidents.map((i) => (
                  <div key={i.id} className="p-4 rounded-lg bg-surface-container hover:bg-surface-high transition-colors space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <Chip tone={i.record_type === 'confirmed_incident' ? 'critical' : 'high'}>{i.record_type_ar}</Chip>
                      <span className="num-mono text-[11px] text-brand">{i.ref}</span>
                      <Chip>{i.review_status_ar}</Chip>
                    </div>
                    <div className="text-sm font-semibold text-ink leading-relaxed">{i.title}</div>
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <EpistemicChip kind="fact" label={`${i.article_count} خبر • ${i.source_count} مصدر`} />
                        {i.locations.slice(0, 3).map((l) => (
                          <Chip key={l}>{l}</Chip>
                        ))}
                      </div>
                      <Link
                        to="/incident-registry"
                        className="text-xs text-ink-variant hover:text-brand flex items-center gap-1"
                      >
                        <Eye className="w-3.5 h-3.5" /> فتح الملف
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        </div>

        <div className="lg:col-span-4 space-y-6">
          {/* Review queue */}
          <Panel title="الانضباط المعرفي" subtitle="ما هو مؤكد، وما ينتظر مراجعة بشرية.">
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-surface-lowest">
                <div className="label-mono">تنتظر المراجعة</div>
                <div className="num-mono text-2xl font-bold text-threat-medium">{report.incidents.needs_review}</div>
                <div className="text-[11px] text-ink-outline">حوادث</div>
              </div>
              <div className="p-3 rounded-lg bg-surface-lowest">
                <div className="label-mono">اقتراحات دمج</div>
                <div className="num-mono text-2xl font-bold text-brand">{report.pending_merge_suggestions}</div>
                <div className="text-[11px] text-ink-outline">معلّقة</div>
              </div>
            </div>
            <div className="flex flex-wrap gap-1.5">
              <EpistemicChip kind="fact" label="الوقائع من نص المصدر فقط" />
              <EpistemicChip kind="inference" label="التحليل مُعلَّم كاستنتاج" />
              <EpistemicChip kind="hypothesis" label="القواعد فرضيات غير مُختبرة" />
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs text-ink-variant">
              <div className="p-2.5 rounded-lg bg-surface-container">
                تحذيرات منفصلة: <span className="num-mono text-ink">{report.separate_from_incidents.fraud_warnings}</span>
              </div>
              <div className="p-2.5 rounded-lg bg-surface-container">
                حملات توعية: <span className="num-mono text-ink">{report.separate_from_incidents.awareness_campaigns}</span>
              </div>
            </div>
          </Panel>

          {/* Source integrity */}
          <Panel
            title="سلامة المصادر"
            subtitle="حالة آخر جلب لكل مصدر."
            icon={<Radio className="w-4 h-4" />}
            right={<Link to="/sources" className="text-xs text-brand hover:underline">التفاصيل ←</Link>}
          >
            {sources.length === 0 ? (
              <Unavailable title="لا مصادر مسجّلة" />
            ) : (
              <div className="space-y-2">
                {sources.map((s) => {
                  const ok = s.last_status === 'ok';
                  const never = !s.last_status;
                  return (
                    <div key={s.code} className="p-2.5 rounded-lg bg-surface-container flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5 min-w-0">
                        <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${never ? 'bg-epistemic-unknown' : ok ? 'bg-threat-low' : 'bg-threat-critical'}`} />
                        <div className="min-w-0">
                          <div className="text-[13px] font-semibold text-ink truncate">{s.name}</div>
                          <div className="text-[10px] font-mono text-ink-outline">
                            {s.official ? 'رسمي' : 'إعلامي'} • {s.stored} مخزّن
                          </div>
                        </div>
                      </div>
                      <div className="text-left shrink-0">
                        <div className={`text-[11px] font-mono ${never ? 'text-ink-outline' : ok ? 'text-threat-low' : 'text-threat-critical'}`}>
                          {never ? 'لم يُجلب' : ok ? 'OK' : s.last_status}
                        </div>
                        <div className="text-[10px] text-ink-outline">{s.last_success_at ? timeAgo(s.last_success_at) : '—'}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
            <div className="text-[11px] text-ink-outline">
              آخر جلب: {fmtDateTime(ov.last_fetch_at)} • دورة التحديث كل {ov.refresh_every_hours} ساعة
            </div>
          </Panel>

          {/* Honest gaps */}
          <Panel title="قنوات غير مربوطة بعد">
            <Unavailable
              title="تيليمتري البنوك والويب المظلم"
              note="لا يوجد ربط ببيانات البنوك الداخلية أو مصادر الويب المظلم، لذلك لا تُعرض أرقام لهما."
            />
          </Panel>
        </div>
      </div>
    </div>
  );
};

export default ExecutiveOverview;
