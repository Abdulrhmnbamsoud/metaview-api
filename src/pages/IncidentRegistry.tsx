// src/pages/IncidentRegistry.tsx
// سجل الحوادث — كل حادثة مستقلة مع كل التغطيات الصحفية المرتبطة بها وسبب الربط.
import React, { useCallback, useEffect, useState } from 'react';
import {
  FolderSearch,
  ExternalLink,
  Search,
  X,
  AlertTriangle,
  CheckCircle2,
  Layers,
  Newspaper,
  ClipboardCheck,
  FlaskConical,
  ShieldAlert,
  ListChecks,
} from 'lucide-react';
import { apiService } from '../services/api';
import {
  SaudiIncident,
  SaudiIncidentReport,
  SaudiOverview,
  SaudiFraudItem,
  MergeSuggestion,
  IncidentArticleRef,
} from '../types/saudi';

type Tab = 'incidents' | 'review' | 'warnings' | 'awareness';

const PAGE_SIZE = 10;

const fmtDate = (s?: string | null) => {
  if (!s) return '—';
  const d = new Date(s);
  if (isNaN(d.getTime())) return s;
  return d.toLocaleString('ar-SA-u-nu-latn', { dateStyle: 'medium', timeStyle: 'short' });
};

const bankBadge: Record<string, { label: string; cls: string }> = {
  high: { label: 'صلة بنكية: عالية', cls: 'bg-red-500/10 text-red-300 border-red-500/30' },
  medium: { label: 'صلة بنكية: متوسطة', cls: 'bg-amber-500/10 text-amber-300 border-amber-500/30' },
  low: { label: 'صلة بنكية: منخفضة', cls: 'bg-white/5 text-gray-400 border-white/10' },
};

const recordBadge: Record<string, string> = {
  confirmed_incident: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
  reported_incident: 'bg-sky-500/10 text-sky-300 border-sky-500/30',
};

const reviewBadge: Record<string, string> = {
  needs_review: 'bg-amber-500/10 text-amber-300 border-amber-500/40',
  human_confirmed: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
  auto_merged: 'bg-white/5 text-gray-300 border-white/10',
  single_source: 'bg-white/5 text-gray-400 border-white/10',
};

const basisLabel = (b?: IncidentArticleRef['match_basis']) => {
  if (!b) return '—';
  if (b.type === 'seed') return 'المقال المرجعي للحادثة';
  if (b.type === 'human_approved') return 'دمج أكّده مراجع بشري';
  const facts = [...(b.strong || []), ...(b.context || [])];
  return facts.length ? `حقائق مشتركة: ${facts.join('، ')}` : 'تطابق حقائق';
};

export const IncidentRegistry: React.FC = () => {
  const [tab, setTab] = useState<Tab>('incidents');
  const [overview, setOverview] = useState<SaudiOverview | null>(null);
  const [report, setReport] = useState<SaudiIncidentReport | null>(null);
  const [incidents, setIncidents] = useState<SaudiIncident[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [recordType, setRecordType] = useState('');
  const [reviewStatus, setReviewStatus] = useState('');
  const [scope, setScope] = useState<'saudi' | 'all'>('saudi');
  const [windowDays, setWindowDays] = useState<number>(30);
  const [fraudType, setFraudType] = useState<string>('');
  const [query, setQuery] = useState('');
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SaudiIncident | null>(null);
  const [suggestions, setSuggestions] = useState<MergeSuggestion[]>([]);
  const [noteReview, setNoteReview] = useState('');
  const [sideItems, setSideItems] = useState<SaudiFraudItem[]>([]);
  const [sideLoading, setSideLoading] = useState(false);

  useEffect(() => {
    apiService.getSaudiIncidentReport().then(setReport).catch(() => {});
  }, []);

  const loadIncidents = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [res, ovRes] = await Promise.all([
        apiService.getSaudiIncidents({
          scope,
          window_days: windowDays > 0 ? windowDays : undefined,
          fraud_type: fraudType || undefined,
          record_type: recordType || undefined,
          review_status: reviewStatus || undefined,
          q: search || undefined,
          page,
          page_size: PAGE_SIZE,
        }),
        apiService.getSaudiOverview({
          scope,
          window_days: windowDays > 0 ? windowDays : undefined,
          fraud_type: fraudType || undefined,
        }).catch(() => null),
      ]);
      setIncidents(res.items);
      setTotal(res.total);
      if (ovRes) setOverview(ovRes);
    } catch (e: any) {
      setError(e.message || 'تعذر تحميل سجل الحوادث');
    } finally {
      setLoading(false);
    }
  }, [scope, windowDays, fraudType, recordType, reviewStatus, search, page]);

  useEffect(() => {
    loadIncidents();
    const handleRefresh = () => loadIncidents();
    window.addEventListener('metaview:refresh', handleRefresh);
    return () => {
      window.removeEventListener('metaview:refresh', handleRefresh);
    };
  }, [loadIncidents]);

  useEffect(() => {
    if (tab === 'review') {
      apiService
        .getSaudiMergeSuggestions('pending')
        .then((r) => {
          setSuggestions(r.items);
          setNoteReview(r.note);
        })
        .catch(() => {});
    }
    if (tab === 'warnings' || tab === 'awareness') {
      setSideLoading(true);
      apiService
        .getSaudiNews({ content_type: tab === 'warnings' ? 'fraud_warning' : 'awareness_campaign', page_size: 50 })
        .then((r) => setSideItems(r.items))
        .catch((e) => setError(e.message || 'تعذر التحميل'))
        .finally(() => setSideLoading(false));
    }
  }, [tab]);

  const openDetail = async (inc: SaudiIncident) => {
    setSelected(inc);
    try {
      setSelected(await apiService.getSaudiIncident(inc.id));
    } catch {
      /* keep summary */
    }
  };

  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const inc = report?.incidents;

  const stats = [
    { label: 'حوادث مستقلة (وفق النطاق)', value: total, icon: FolderSearch },
    { label: 'مقالات خلف الحوادث', value: inc?.articles_behind_incidents ?? '…', icon: Newspaper },
    { label: 'مؤكدة / مُبلَّغ عنها', value: inc ? `${inc.confirmed} / ${inc.reported}` : '…', icon: CheckCircle2 },
    { label: 'تحتاج مراجعة بشرية', value: inc?.needs_review ?? '…', icon: ClipboardCheck },
    { label: 'تحذيرات (خارج الحوادث)', value: report?.separate_from_incidents.fraud_warnings ?? '…', icon: ShieldAlert },
    { label: 'توعية (خارج الحوادث)', value: report?.separate_from_incidents.awareness_campaigns ?? '…', icon: Layers },
  ];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Title */}
      <div className="pb-4 border-b border-white/5">
        <h1 className="text-2xl font-black text-white royal-gradient-text flex items-center gap-3">
          <FolderSearch className="w-7 h-7 text-[#4cd7f6]" />
          سجل الحوادث (Incident Registry)
        </h1>
        <p className="text-xs text-gray-400 mt-1 leading-relaxed">
          الحادثة غير الخبر: قد تغطي عدة مقالات حادثة واحدة. كل مقال أصلي محفوظ ومرتبط برابطه، والدمج يعتمد على
          الوقائع (العدد، المبالغ، المضبوطات، الجنسيات، الموقع، الجهة) وليس العنوان وحده.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
        {stats.map((s) => {
          const Icon = s.icon;
          return (
            <div key={s.label} className="sovereign-card rounded-2xl p-4 border border-white/10">
              <div className="flex items-center justify-between text-gray-400 text-[11px]">
                <span>{s.label}</span>
                <Icon className="w-4 h-4 text-[#4cd7f6]" />
              </div>
              <div className="text-2xl font-black text-white mt-2 font-mono">{s.value}</div>
            </div>
          );
        })}
      </div>
      {report && (
        <div className="text-[11px] text-gray-500 font-mono flex flex-wrap gap-x-6 gap-y-1">
          <span>مقالات سعودية محفوظة: {report.articles.saudi_fraud_articles}</span>
          <span>مقالات لكل حادثة: {report.incidents.articles_per_incident}</span>
          <span>مستبعد (أمن سيبراني عام): {report.excluded_from_indicators.general_cybersecurity}</span>
          <span>مستبعد (غير متعلق): {report.excluded_from_indicators.unrelated}</span>
        </div>
      )}

      {error && (
        <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" /> {error}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 border-b border-white/5 overflow-x-auto">
        {(
          [
            ['incidents', 'الحوادث'],
            ['review', `مراجعة الدمج${overview?.merge_suggestions_pending ? ` (${overview.merge_suggestions_pending})` : ''}`],
            ['warnings', 'التحذيرات'],
            ['awareness', 'التوعية'],
          ] as [Tab, string][]
        ).map(([k, label]) => (
          <button
            key={k}
            id={`incident-tab-${k}`}
            onClick={() => setTab(k)}
            className={`px-4 py-2.5 text-xs font-semibold border-b-2 whitespace-nowrap transition-colors ${
              tab === k ? 'border-[#4cd7f6] text-[#acedff]' : 'border-transparent text-gray-400 hover:text-white'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* INCIDENTS */}
      {tab === 'incidents' && (
        <div className="space-y-4">
          <div className="flex flex-wrap gap-3">
            <form
              className="relative flex-1 min-w-[220px]"
              onSubmit={(e) => {
                e.preventDefault();
                setSearch(query.trim());
                setPage(1);
              }}
            >
              <input
                id="incident-search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="بحث في عناوين الحوادث…"
                className="w-full bg-[#0F172A] border border-[#4cd7f6]/20 focus:border-[#4cd7f6]/60 py-2.5 pr-10 pl-3 rounded-xl text-xs text-white placeholder-gray-500 outline-none"
              />
              <Search className="w-4 h-4 text-gray-500 absolute right-3 top-1/2 -translate-y-1/2" />
            </form>
            <select
              id="incident-scope-filter"
              value={scope}
              onChange={(e) => { setScope(e.target.value as 'saudi' | 'all'); setPage(1); }}
              className="bg-[#0F172A] border border-white/10 rounded-xl text-xs text-gray-200 px-3 py-2.5"
            >
              <option value="saudi">🇸🇦 المملكة العربية السعودية</option>
              <option value="all">🌐 كافة النطاقات (العالم والكل)</option>
            </select>
            <select
              id="incident-window-filter"
              value={windowDays}
              onChange={(e) => { setWindowDays(Number(e.target.value)); setPage(1); }}
              className="bg-[#0F172A] border border-white/10 rounded-xl text-xs text-gray-200 px-3 py-2.5"
            >
              <option value={30}>آخر 30 يومًا (المعياري)</option>
              <option value={90}>آخر 90 يومًا</option>
              <option value={365}>آخر سنة (365 يومًا)</option>
              <option value={0}>كامل السجل التاريخي</option>
            </select>
            <select
              id="incident-fraud-type-filter"
              value={fraudType}
              onChange={(e) => { setFraudType(e.target.value); setPage(1); }}
              className="bg-[#0F172A] border border-white/10 rounded-xl text-xs text-gray-200 px-3 py-2.5"
            >
              <option value="">كافة أنواع الاحتيال</option>
              <option value="gov_impersonation">انتحال جهات حكومية</option>
              <option value="bank_impersonation">انتحال صفة بنكية</option>
              <option value="phishing">تصيد احتيالي</option>
              <option value="investment_fraud">احتيال استثماري</option>
              <option value="job_fraud">احتيال توظيفي</option>
            </select>
            <select
              value={recordType}
              onChange={(e) => { setRecordType(e.target.value); setPage(1); }}
              className="bg-[#0F172A] border border-white/10 rounded-xl text-xs text-gray-200 px-3 py-2.5"
            >
              <option value="">كل أنواع السجل</option>
              <option value="confirmed_incident">واقعة مؤكدة</option>
              <option value="reported_incident">واقعة مُبلَّغ عنها</option>
            </select>
            <select
              value={reviewStatus}
              onChange={(e) => { setReviewStatus(e.target.value); setPage(1); }}
              className="bg-[#0F172A] border border-white/10 rounded-xl text-xs text-gray-200 px-3 py-2.5"
            >
              <option value="">كل حالات المراجعة</option>
              <option value="needs_review">تحتاج مراجعة</option>
              <option value="auto_merged">دُمجت تلقائياً</option>
              <option value="human_confirmed">أكّدها مراجع</option>
              <option value="single_source">مصدر واحد</option>
            </select>
          </div>

          {loading ? (
            <div className="text-center text-gray-500 text-xs py-12">جاري التحميل…</div>
          ) : incidents.length === 0 ? (
            <div className="sovereign-card rounded-2xl p-8 border border-white/10 text-center text-xs text-gray-400 leading-relaxed">
              لا توجد حوادث مطابقة. لا تُعرض أي بيانات غير حقيقية.
            </div>
          ) : (
            <div className="grid gap-4">
              {incidents.map((it) => {
                const bb = it.bank_relevance ? bankBadge[it.bank_relevance] : null;
                return (
                  <article
                    key={it.id}
                    className="sovereign-card rounded-2xl p-5 border border-white/10 hover:border-[#4cd7f6]/40 transition-colors"
                  >
                    <div className="flex flex-wrap items-center gap-2 mb-3">
                      <span className="px-2.5 py-1 rounded-lg text-[10px] font-mono font-bold bg-black/40 text-gray-300 border border-white/10">
                        {it.ref}
                      </span>
                      <span className={`px-2.5 py-1 rounded-lg text-[10px] font-bold border ${recordBadge[it.record_type] || ''}`}>
                        {it.record_type_ar}
                      </span>
                      <span className="px-2.5 py-1 rounded-lg text-[10px] font-bold bg-[#4cd7f6]/15 text-[#acedff] border border-[#4cd7f6]/30">
                        {it.fraud_category_ar}
                      </span>
                      <span className={`px-2.5 py-1 rounded-lg text-[10px] border ${reviewBadge[it.review_status] || ''}`}>
                        {it.review_status_ar}
                      </span>
                      {bb && <span className={`px-2.5 py-1 rounded-lg text-[10px] border ${bb.cls}`}>{bb.label}</span>}
                    </div>
                    <h3
                      className="text-base font-bold text-white leading-snug cursor-pointer hover:text-[#acedff]"
                      onClick={() => openDetail(it)}
                    >
                      {it.title}
                    </h3>
                    {it.summary && <p className="text-xs text-gray-400 leading-relaxed mt-2 line-clamp-3">{it.summary}</p>}
                    <div className="flex flex-wrap gap-x-5 gap-y-1 mt-3 text-[11px] text-gray-400">
                      <span>تغطيات مرتبطة: <b className="text-gray-200 font-mono">{it.article_count}</b></span>
                      <span>مصادر مختلفة: <b className="text-gray-200 font-mono">{it.source_count}</b></span>
                      {it.locations.length > 0 && <span>الموقع: <b className="text-gray-200">{it.locations.join('، ')}</b></span>}
                      {it.agencies.length > 0 && <span>الجهات: <b className="text-gray-200">{it.agencies.join('، ')}</b></span>}
                      <span>أول نشر: {fmtDate(it.first_reported_at)}</span>
                    </div>
                    <div className="flex flex-wrap items-center justify-between gap-3 mt-4 pt-3 border-t border-white/5">
                      <ul className="text-[11px] text-gray-400 space-y-1 flex-1 min-w-[240px]">
                        {it.articles.slice(0, 3).map((a) => (
                          <li key={a.article_id} className="flex items-center gap-2">
                            <ExternalLink className="w-3 h-3 text-[#4cd7f6] shrink-0" />
                            <a href={a.url} target="_blank" rel="noopener noreferrer" className="hover:text-[#acedff] truncate">
                              {a.source || a.domain || 'مصدر'} — {a.title}
                            </a>
                          </li>
                        ))}
                        {it.articles.length > 3 && <li className="text-gray-500">+ {it.articles.length - 3} تغطيات أخرى</li>}
                      </ul>
                      <button
                        onClick={() => openDetail(it)}
                        className="px-3 py-1.5 rounded-lg bg-[#4cd7f6]/15 border border-[#4cd7f6]/40 text-[#acedff] text-[11px]"
                      >
                        تفاصيل الحادثة وسبب الدمج
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          )}

          {pages > 1 && (
            <div className="flex items-center justify-center gap-3 text-xs text-gray-300">
              <button disabled={page <= 1} onClick={() => setPage(page - 1)} className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 disabled:opacity-40">السابق</button>
              <span className="font-mono">{page} / {pages}</span>
              <button disabled={page >= pages} onClick={() => setPage(page + 1)} className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 disabled:opacity-40">التالي</button>
            </div>
          )}
        </div>
      )}

      {/* REVIEW QUEUE */}
      {tab === 'review' && (
        <div className="space-y-4">
          <div className="p-4 rounded-2xl bg-[#0F172A] border border-[#4cd7f6]/20 flex gap-3 text-xs text-gray-300 leading-relaxed">
            <ClipboardCheck className="w-5 h-5 text-[#4cd7f6] shrink-0" />
            <div>
              {noteReview || 'اقتراحات دمج غير حاسمة تنتظر قراراً بشرياً. لا يُنفَّذ أي دمج غير مؤكد تلقائياً.'}
              <div className="text-[11px] text-gray-500 mt-1">
                قرار المراجع (موافقة / رفض) يُسجَّل عبر واجهة الإدارة المحمية فقط، وليس من المتصفح.
              </div>
            </div>
          </div>
          {suggestions.length === 0 ? (
            <div className="sovereign-card rounded-2xl p-8 border border-white/10 text-center text-xs text-gray-400">
              لا توجد حالات دمج معلّقة. جميع الحوادث إمّا مدموجة بأدلة قوية أو منفصلة.
            </div>
          ) : (
            suggestions.map((s) => (
              <div key={s.id} className="sovereign-card rounded-2xl p-5 border border-amber-500/30 space-y-3">
                <div className="flex justify-between text-[11px] text-gray-400">
                  <span className="font-mono">اقتراح #{s.id}</span>
                  <span className="font-mono text-amber-300">درجة التشابه: {s.score}</span>
                </div>
                <div className="grid md:grid-cols-2 gap-3 text-xs">
                  {[s.article_a, s.article_b].map((a, i) => (
                    <div key={i} className="p-3 rounded-xl bg-black/40 border border-white/5 space-y-1">
                      <div className="text-gray-200 font-semibold leading-snug">{a.headline || `مقال ${a.id}`}</div>
                      <div className="text-[11px] text-gray-500">{a.source} • {fmtDate(a.published_at)}</div>
                      {a.url && (
                        <a href={a.url} target="_blank" rel="noopener noreferrer" dir="ltr" className="text-[11px] text-[#acedff] break-all hover:underline">
                          {a.url}
                        </a>
                      )}
                    </div>
                  ))}
                </div>
                <pre dir="ltr" className="text-[11px] font-mono bg-black/60 border border-white/5 rounded-lg p-3 whitespace-pre-wrap text-amber-200">
                  {JSON.stringify(s.why_not_merged, null, 2)}
                </pre>
              </div>
            ))
          )}
        </div>
      )}

      {/* WARNINGS / AWARENESS */}
      {(tab === 'warnings' || tab === 'awareness') && (
        <div className="space-y-4">
          <div className="p-4 rounded-2xl bg-[#0F172A] border border-[#4cd7f6]/20 text-xs text-gray-300 leading-relaxed">
            {tab === 'warnings'
              ? 'التحذيرات تنبّه من أسلوب أو تهديد ولا تصف واقعة بعينها، لذلك لا تُحتسب حوادث.'
              : 'حملات التوعية (مثل حملة «شيّك») تثقيفية ولا تصف واقعة احتيال، لذلك لا تُحتسب حوادث.'}
          </div>
          {sideLoading ? (
            <div className="text-center text-gray-500 text-xs py-12">جاري التحميل…</div>
          ) : sideItems.length === 0 ? (
            <div className="sovereign-card rounded-2xl p-8 border border-white/10 text-center text-xs text-gray-400">لا عناصر.</div>
          ) : (
            sideItems.map((it) => (
              <div key={it.id} className="sovereign-card rounded-2xl p-4 border border-white/10 flex flex-wrap items-center justify-between gap-3">
                <div className="min-w-[240px] flex-1">
                  <div className="text-sm font-bold text-white leading-snug">{it.headline || it.title}</div>
                  <div className="text-[11px] text-gray-500 mt-1">{it.source} • {fmtDate(it.published_at)}</div>
                </div>
                {it.original_url && (
                  <a href={it.original_url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-[#4cd7f6]/15 border border-[#4cd7f6]/40 text-[#acedff] text-[11px]">
                    المصدر الأصلي <ExternalLink className="w-3 h-3" />
                  </a>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* DETAIL MODAL */}
      {selected && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-start justify-center p-4 overflow-y-auto" onClick={() => setSelected(null)}>
          <div className="w-full max-w-3xl bg-[#0a0b12] border border-[#4cd7f6]/30 rounded-2xl p-6 space-y-5 my-8" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex flex-wrap gap-2">
                  <span className="px-2.5 py-1 rounded-lg text-[10px] font-mono bg-black/40 text-gray-300 border border-white/10">{selected.ref}</span>
                  <span className={`px-2.5 py-1 rounded-lg text-[10px] font-bold border ${recordBadge[selected.record_type] || ''}`}>{selected.record_type_ar}</span>
                  <span className={`px-2.5 py-1 rounded-lg text-[10px] border ${reviewBadge[selected.review_status] || ''}`}>{selected.review_status_ar}</span>
                </div>
                <h2 className="text-lg font-bold text-white mt-3 leading-snug">{selected.title}</h2>
                <p className="text-[11px] text-gray-500 mt-1">
                  {selected.article_count} تغطية • {selected.source_count} مصدر • أول نشر {fmtDate(selected.first_reported_at)}
                </p>
              </div>
              <button onClick={() => setSelected(null)} className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-300">
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Coverage + merge reasons */}
            <section className="space-y-2">
              <h3 className="text-xs font-bold text-white flex items-center gap-2">
                <Newspaper className="w-4 h-4 text-[#4cd7f6]" /> التغطيات المرتبطة وسبب ربط كل واحدة
              </h3>
              <ul className="space-y-2">
                {selected.articles.map((a) => (
                  <li key={a.article_id} className="p-3 rounded-lg bg-black/40 border border-white/5 text-[11px] space-y-1">
                    <div className="flex justify-between gap-2">
                      <b className="text-gray-200">{a.source || a.domain}</b>
                      <span className="text-gray-500">{fmtDate(a.published_at)}</span>
                    </div>
                    <div className="text-gray-300 leading-snug">{a.title}</div>
                    <div className="text-amber-200/80">سبب الربط: {basisLabel(a.match_basis)}</div>
                    <a href={a.url} target="_blank" rel="noopener noreferrer" dir="ltr" className="flex items-center gap-1 text-[#acedff] break-all hover:underline">
                      <ExternalLink className="w-3 h-3 shrink-0" /> {a.url}
                    </a>
                  </li>
                ))}
              </ul>
              {selected.match_basis && <p className="text-[10px] text-gray-500 leading-relaxed">{selected.match_basis}</p>}
            </section>

            {(selected.pending_merge_suggestions || []).length > 0 && (
              <section className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/30 text-[11px] text-amber-200">
                <AlertTriangle className="w-4 h-4 inline ml-1" />
                توجد {selected.pending_merge_suggestions!.length} اقتراحات دمج معلّقة لهذه الحادثة، تنتظر قراراً بشرياً (تبويب «مراجعة الدمج»).
              </section>
            )}

            {/* Facts */}
            {selected.facts && (
              <section className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/20 space-y-2">
                <h3 className="text-xs font-bold text-emerald-300 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4" /> حقائق وردت في المصادر
                </h3>
                <div className="text-[11px] text-gray-300 flex flex-wrap gap-x-6 gap-y-1">
                  {selected.facts.persons_count != null && <span>عدد الأشخاص: <b className="text-white">{selected.facts.persons_count}</b></span>}
                  {selected.facts.nationalities.length > 0 && <span>الجنسيات: <b className="text-white">{selected.facts.nationalities.map((n) => n.label).join('، ')}</b></span>}
                  {selected.facts.amounts.length > 0 && <span>مبالغ: <b className="text-white font-mono">{selected.facts.amounts.map((a) => a.value.toLocaleString('en-US')).join(' / ')}</b></span>}
                  {selected.facts.seized_items.length > 0 && <span>مضبوطات: <b className="text-white">{selected.facts.seized_items.map((n) => n.label).join('، ')}</b></span>}
                  {selected.facts.locations.length > 0 && <span>المواقع: <b className="text-white">{selected.facts.locations.map((n) => n.label).join('، ')}</b></span>}
                  {selected.facts.agencies.length > 0 && <span>الجهات: <b className="text-white">{selected.facts.agencies.map((n) => n.label).join('، ')}</b></span>}
                </div>
                {selected.facts.fact_sentences.slice(0, 4).map((s, i) => (
                  <p key={i} className="text-[11px] text-gray-300 leading-relaxed border-r-2 border-emerald-500/40 pr-2">
                    {s.text} <a href={s.url} target="_blank" rel="noopener noreferrer" className="text-[#acedff] hover:underline">— {s.source}</a>
                  </p>
                ))}
              </section>
            )}

            {/* Methods */}
            {selected.methods && selected.methods.length > 0 && (
              <section className="space-y-2">
                <h3 className="text-xs font-bold text-white flex items-center gap-2">
                  <FlaskConical className="w-4 h-4 text-[#4cd7f6]" /> أساليب ذُكرت صراحة (مع الاقتباس)
                </h3>
                {selected.methods.map((m) => (
                  <div key={m.tag} className="p-3 rounded-lg bg-black/40 border border-white/5 text-[11px]">
                    <b className="text-[#acedff]">{m.label}</b>
                    {m.quotes.slice(0, 1).map((q, i) => (
                      <p key={i} className="text-gray-400 mt-1 leading-relaxed">«{q.quote}» <a href={q.url} target="_blank" rel="noopener noreferrer" className="text-[#acedff] hover:underline">— {q.source}</a></p>
                    ))}
                  </div>
                ))}
              </section>
            )}

            {/* Indicators */}
            {selected.indicators && (
              <section className="space-y-2">
                <h3 className="text-xs font-bold text-white flex items-center gap-2">
                  <ListChecks className="w-4 h-4 text-[#4cd7f6]" /> مؤشرات المخاطر
                </h3>
                <div className="grid md:grid-cols-2 gap-3 text-[11px]">
                  <div className="p-3 rounded-lg bg-black/40 border border-white/5">
                    <b className="text-gray-200">مذكورة في المصدر</b>
                    <ul className="list-disc pr-5 mt-1 text-gray-400">
                      {selected.indicators.mentioned_in_source.length
                        ? selected.indicators.mentioned_in_source.map((x, i) => <li key={i}>{x.label}</li>)
                        : <li>لا شيء</li>}
                    </ul>
                  </div>
                  <div className="p-3 rounded-lg bg-black/40 border border-white/5">
                    <b className="text-gray-200">مقترحة من النظام (استنتاج)</b>
                    <ul className="list-disc pr-5 mt-1 text-gray-400">
                      {selected.indicators.suggested_by_system.items.map((x, i) => <li key={i}>{x}</li>)}
                    </ul>
                  </div>
                </div>
              </section>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default IncidentRegistry;
