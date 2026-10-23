// src/pages/SaudiFraud.tsx
import React, { useCallback, useEffect, useState } from 'react';
import {
  ShieldAlert,
  RefreshCw,
  ExternalLink,
  Search,
  X,
  Clock,
  Radio,
  Landmark,
  ListChecks,
  FlaskConical,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { apiService } from '../services/api';
import {
  SaudiOverview,
  SaudiFraudItem,
  SaudiSourceInfo,
  SaudiPatternsResponse,
  SaudiRulesResponse,
} from '../types/saudi';

type Tab = 'news' | 'patterns' | 'rules' | 'sources';

const PAGE_SIZE = 15;

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

const eventLabel: Record<string, string> = {
  confirmed_incident: 'واقعة مؤكدة',
  reported_incident: 'واقعة مُبلَّغ عنها',
  warning: 'تحذير',
  guidance: 'توعية',
};

export const SaudiFraud: React.FC = () => {
  const [tab, setTab] = useState<Tab>('news');
  const [overview, setOverview] = useState<SaudiOverview | null>(null);
  const [items, setItems] = useState<SaudiFraudItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [category, setCategory] = useState('');
  const [bank, setBank] = useState('');
  const [query, setQuery] = useState('');
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SaudiFraudItem | null>(null);
  const [patterns, setPatterns] = useState<SaudiPatternsResponse | null>(null);
  const [rules, setRules] = useState<SaudiRulesResponse | null>(null);
  const [sources, setSources] = useState<SaudiSourceInfo[]>([]);

  const loadOverview = useCallback(async () => {
    try {
      const o = await apiService.getSaudiOverview();
      setOverview(o);
      return o;
    } catch (e: any) {
      setError(e.message || 'تعذر تحميل الملخص');
      return null;
    }
  }, []);

  const loadNews = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.getSaudiNews({
        category: category || undefined,
        bank_relevance: bank || undefined,
        q: search || undefined,
        page,
        page_size: PAGE_SIZE,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (e: any) {
      setError(e.message || 'تعذر تحميل الأخبار');
    } finally {
      setLoading(false);
    }
  }, [category, bank, search, page]);

  useEffect(() => {
    loadOverview();
  }, [loadOverview]);

  useEffect(() => {
    loadNews();
  }, [loadNews]);

  useEffect(() => {
    if (tab === 'patterns' && !patterns) apiService.getSaudiPatterns().then(setPatterns).catch(() => {});
    if (tab === 'rules' && !rules) apiService.getSaudiRules().then(setRules).catch(() => {});
    if (tab === 'sources') apiService.getSaudiSources().then((r) => setSources(r.sources)).catch(() => {});
  }, [tab, patterns, rules]);

  // Poll (read-only) while a server-side ingestion is running
  useEffect(() => {
    if (!overview?.ingest_running) return;
    const t = setInterval(async () => {
      const o = await loadOverview();
      if (o && !o.ingest_running) {
        setPatterns(null);
        setRules(null);
        loadNews();
      }
    }, 5000);
    return () => clearInterval(t);
  }, [overview?.ingest_running, loadOverview, loadNews]);

  const openDetail = async (it: SaudiFraudItem) => {
    setSelected(it);
    try {
      setSelected(await apiService.getSaudiNewsItem(it.id));
    } catch {
      /* keep summary view */
    }
  };

  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-white/5">
        <div>
          <h1 className="text-2xl font-black text-white royal-gradient-text flex items-center gap-3">
            <ShieldAlert className="w-7 h-7 text-[#c5a059]" />
            استخبارات الاحتيال السعودية (Saudi Fraud Intelligence)
          </h1>
          <p className="text-xs text-gray-400 mt-1">
            أخبار احتيال حقيقية من مصادر سعودية، مع روابط المصدر الأصلي وتحليل منفصل بين الحقائق المنقولة واستنتاجات النظام.
          </p>
        </div>
        <div
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold border ${
            overview?.ingest_running
              ? 'bg-[#c5a059]/20 border-[#c5a059]/40 text-[#e0c58a]'
              : 'bg-white/5 border-white/10 text-gray-300'
          }`}
          title="السحب اليدوي إجراء إداري من الخادم فقط"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-[#c5a059] ${overview?.ingest_running ? 'animate-spin' : ''}`} />
          {overview?.ingest_running
            ? 'جاري السحب...'
            : overview?.auto_ingest_enabled
              ? `تحديث تلقائي كل ${overview.refresh_every_hours} ساعات`
              : 'التحديث التلقائي متوقف'}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'حوادث مستقلة', value: overview?.incidents_total ?? '…', icon: ShieldAlert },
          { label: 'مقالات (حوادث + تحذيرات + توعية)', value: overview?.total_items ?? '…', icon: Clock },
          { label: 'حوادث بصلة بنكية عالية', value: overview?.high_bank_relevance ?? '…', icon: Landmark },
          {
            label: 'مصادر تعمل / الإجمالي',
            value: overview ? `${overview.sources_ok} / ${overview.sources_total}` : '…',
            icon: Radio,
          },
        ].map((s) => {
          const Icon = s.icon;
          return (
            <div key={s.label} className="sovereign-card rounded-2xl p-4 border border-white/10">
              <div className="flex items-center justify-between text-gray-400 text-[11px]">
                <span>{s.label}</span>
                <Icon className="w-4 h-4 text-[#c5a059]" />
              </div>
              <div className="text-2xl font-black text-white mt-2 font-mono">{s.value}</div>
            </div>
          );
        })}
      </div>
      <div className="text-[11px] text-gray-500 font-mono flex flex-wrap gap-x-6 gap-y-1">
        <span>آخر سحب: {fmtDate(overview?.last_fetch_at)}</span>
        <span>
          التحديث التلقائي: {overview?.auto_ingest_enabled ? `كل ${overview.refresh_every_hours} ساعات` : 'غير مفعّل'}
        </span>
      </div>

      {error && (
        <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" /> {error}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 border-b border-white/5">
        {(
          [
            ['news', 'آخر الأخبار'],
            ['patterns', 'الأنماط'],
            ['rules', 'قواعد الكشف المقترحة'],
            ['sources', 'المصادر'],
          ] as [Tab, string][]
        ).map(([k, label]) => (
          <button
            key={k}
            onClick={() => setTab(k)}
            className={`px-4 py-2.5 text-xs font-semibold border-b-2 transition-colors ${
              tab === k ? 'border-[#c5a059] text-[#e0c58a]' : 'border-transparent text-gray-400 hover:text-white'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* NEWS */}
      {tab === 'news' && (
        <div className="space-y-4">
          <div className="flex flex-wrap gap-2 items-center">
            <button
              onClick={() => { setCategory(''); setPage(1); }}
              className={`px-3 py-1.5 rounded-lg text-[11px] border ${
                !category ? 'bg-[#c5a059]/20 border-[#c5a059]/50 text-[#e0c58a]' : 'bg-white/5 border-white/10 text-gray-300'
              }`}
            >
              الكل ({overview?.total_items ?? 0})
            </button>
            {overview?.categories.filter((c) => c.count > 0).map((c) => (
              <button
                key={c.key}
                onClick={() => { setCategory(c.key); setPage(1); }}
                className={`px-3 py-1.5 rounded-lg text-[11px] border ${
                  category === c.key
                    ? 'bg-[#c5a059]/20 border-[#c5a059]/50 text-[#e0c58a]'
                    : 'bg-white/5 border-white/10 text-gray-300 hover:border-[#c5a059]/30'
                }`}
              >
                {c.label_ar} ({c.count})
              </button>
            ))}
          </div>
          <div className="flex flex-wrap gap-3">
            <form
              className="relative flex-1 min-w-[220px]"
              onSubmit={(e) => { e.preventDefault(); setSearch(query.trim()); setPage(1); }}
            >
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="بحث في عناوين وملخصات الاحتيال..."
                className="w-full bg-[#0d0f18] border border-[#c5a059]/20 focus:border-[#c5a059]/60 py-2.5 pr-10 pl-3 rounded-xl text-xs text-white placeholder-gray-500 outline-none"
              />
              <Search className="w-4 h-4 text-gray-500 absolute right-3 top-1/2 -translate-y-1/2" />
            </form>
            <select
              value={bank}
              onChange={(e) => { setBank(e.target.value); setPage(1); }}
              className="bg-[#0d0f18] border border-white/10 rounded-xl text-xs text-gray-200 px-3 py-2.5"
            >
              <option value="">كل مستويات الصلة البنكية</option>
              <option value="high">عالية</option>
              <option value="medium">متوسطة</option>
              <option value="low">منخفضة</option>
            </select>
          </div>

          {loading ? (
            <div className="text-center text-gray-500 text-xs py-12">جاري التحميل…</div>
          ) : items.length === 0 ? (
            <div className="sovereign-card rounded-2xl p-8 border border-white/10 text-center text-xs text-gray-400 leading-relaxed">
              لا توجد أخبار احتيال سعودية مطابقة حاليًا. اضغط «سحب يدوي الآن» لتشغيل الجمع من المصادر،
              ولن تُعرض أي بيانات غير حقيقية.
            </div>
          ) : (
            <div className="grid gap-4">
              {items.map((it) => {
                const bb = it.bank_relevance ? bankBadge[it.bank_relevance] : null;
                return (
                  <article
                    key={it.id}
                    className="sovereign-card rounded-2xl p-5 border border-white/10 hover:border-[#c5a059]/40 transition-colors"
                  >
                    <div className="flex flex-wrap items-center gap-2 mb-3">
                      <span className="px-2.5 py-1 rounded-lg text-[10px] font-bold bg-[#c5a059]/15 text-[#e0c58a] border border-[#c5a059]/30">
                        {it.fraud_category_ar}
                      </span>
                      {it.content_type_ar && (
                        <span className="px-2.5 py-1 rounded-lg text-[10px] bg-white/5 text-gray-300 border border-white/10">
                          {it.content_type_ar}
                        </span>
                      )}
                      {it.incident_id ? (
                        <Link to="/incident-registry" className="px-2.5 py-1 rounded-lg text-[10px] bg-sky-500/10 text-sky-300 border border-sky-500/30 hover:bg-sky-500/20">
                          ضمن حادثة #{it.incident_id}
                        </Link>
                      ) : null}
                      {!it.content_type_ar && it.event_type && (
                        <span className="px-2.5 py-1 rounded-lg text-[10px] bg-white/5 text-gray-300 border border-white/10">
                          {eventLabel[it.event_type] || it.event_type}
                        </span>
                      )}
                      {bb && <span className={`px-2.5 py-1 rounded-lg text-[10px] border ${bb.cls}`}>{bb.label}</span>}
                    </div>
                    <h3
                      className="text-base font-bold text-white leading-snug cursor-pointer hover:text-[#e0c58a]"
                      onClick={() => openDetail(it)}
                    >
                      {it.headline || it.title}
                    </h3>
                    {it.summary && <p className="text-xs text-gray-400 leading-relaxed mt-2 line-clamp-3">{it.summary}</p>}
                    <div className="flex flex-wrap items-center justify-between gap-3 mt-4 pt-3 border-t border-white/5 text-[11px] text-gray-400">
                      <div className="flex flex-wrap gap-x-4 gap-y-1">
                        <span>المصدر: <b className="text-gray-200">{it.source || '—'}</b></span>
                        <span>نُشر: {fmtDate(it.published_at)}</span>
                        <span>سُحب: {fmtDate(it.fetched_at)}</span>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => openDetail(it)}
                          className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-[#c5a059]/15 border border-white/10 text-gray-200"
                        >
                          التحليل الاستخباراتي
                        </button>
                        {it.original_url && (
                          <a
                            href={it.original_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-[#c5a059]/15 border border-[#c5a059]/40 text-[#e0c58a]"
                          >
                            المصدر الأصلي <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                      </div>
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

      {/* PATTERNS */}
      {tab === 'patterns' && (
        <div className="space-y-4">
          <div className="p-4 rounded-2xl bg-[#0d0f18] border border-[#c5a059]/20 text-xs text-gray-300 leading-relaxed">
            {patterns?.note || 'الأنماط أساليب ذُكرت صراحة في المصادر؛ العدّ بالحوادث المستقلة وليس بالمقالات.'}
          </div>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="sovereign-card rounded-2xl p-5 border border-white/10 space-y-3">
              <h2 className="text-sm font-bold text-white">أساليب الاحتيال المذكورة في المصادر</h2>
              {patterns?.patterns.length ? patterns.patterns.map((p) => (
                <div key={p.key} className="p-3 rounded-xl bg-black/40 border border-white/5 space-y-1">
                  <div className="flex justify-between text-xs">
                    <b className="text-[#e0c58a]">{p.label_ar}</b>
                    <span className="font-mono text-gray-300">{p.incident_count} حادثة • {p.warning_article_count} تحذير</span>
                  </div>
                  {p.evidence[0] && (
                    <p className="text-[11px] text-gray-400 leading-relaxed">
                      «{p.evidence[0].quote}»{' '}
                      <a href={p.evidence[0].url} target="_blank" rel="noopener noreferrer" className="text-[#e0c58a] hover:underline">— {p.evidence[0].source}</a>
                    </p>
                  )}
                </div>
              )) : <p className="text-xs text-gray-500">لا أنماط بعد.</p>}
            </div>
            <div className="sovereign-card rounded-2xl p-5 border border-white/10 space-y-3">
              <h2 className="text-sm font-bold text-white">القنوات المذكورة في المصادر</h2>
              {patterns?.channels.length ? patterns.channels.map((c) => (
                <div key={c.channel} className="flex justify-between text-xs p-2 rounded-lg bg-black/40 border border-white/5">
                  <span className="text-gray-200">{c.channel}</span>
                  <span className="font-mono text-gray-300">{c.count}</span>
                </div>
              )) : <p className="text-xs text-gray-500">لا قنوات بعد.</p>}
            </div>
          </div>
        </div>
      )}

      {/* RULES */}
      {tab === 'rules' && (
        <div className="space-y-4">
          <div className="p-4 rounded-2xl bg-[#0d0f18] border border-[#c5a059]/20 flex gap-3 text-xs text-gray-300 leading-relaxed">
            <FlaskConical className="w-5 h-5 text-[#c5a059] shrink-0" />
            {rules?.note || 'قواعد مقترحة من النظام (فرضيات) وغير مُختبرة على بيانات فعلية.'}
          </div>
          {rules?.rules.length ? rules.rules.map((r) => (
            <div key={r.rule_id} className="sovereign-card rounded-2xl p-5 border border-white/10 space-y-3">
              <div className="flex flex-wrap justify-between gap-2">
                <h3 className="font-bold text-white text-sm">{r.name}</h3>
                <span className="text-[10px] font-mono px-2 py-1 rounded-lg bg-amber-500/10 text-amber-300 border border-amber-500/30">
                  {r.rule_id} • {r.status} • {r.evidence.incident_count} حادثة داعمة
                </span>
              </div>
              <p className="text-xs text-gray-400">{r.hypothesis}</p>
              <pre dir="ltr" className="text-[11px] font-mono bg-black/60 border border-white/5 rounded-lg p-3 whitespace-pre-wrap text-emerald-300">{r.logic}</pre>
              <div className="text-[11px] text-gray-400">
                <b className="text-gray-200">البيانات المطلوبة للاختبار:</b>
                <ul className="list-disc pr-5 mt-1 space-y-0.5">{(r.required_data || []).map((d, i) => <li key={i}>{d}</li>)}</ul>
              </div>
              <div className="text-[11px] text-gray-400">
                <b className="text-gray-200">الحوادث الداعمة:</b>
                <ul className="list-disc pr-5 mt-1 space-y-0.5">
                  {r.evidence.incidents.map((i) => (
                    <li key={i.id}>{i.title} <span className="text-gray-500">({i.article_count} تغطية)</span></li>
                  ))}
                  {r.evidence.incidents.length === 0 && <li>لا حوادث؛ مبنية على {r.evidence.warning_article_count} تحذير فقط</li>}
                </ul>
              </div>
              <p className="text-[10px] text-gray-500">{r.evidence.note}</p>
            </div>
          )) : <p className="text-xs text-gray-500">لا قواعد بعد.</p>}
        </div>
      )}

      {/* SOURCES */}
      {tab === 'sources' && (
        <div className="sovereign-card rounded-2xl border border-white/10 overflow-x-auto">
          <table className="w-full text-xs text-right">
            <thead className="text-gray-400 border-b border-white/10">
              <tr>
                <th className="p-3">المصدر</th><th className="p-3">النوع</th><th className="p-3">الحالة</th>
                <th className="p-3">آخر نجاح</th><th className="p-3">عناصر</th><th className="p-3">محفوظ</th><th className="p-3">الخطأ</th>
              </tr>
            </thead>
            <tbody>
              {sources.map((s) => (
                <tr key={s.code} className="border-b border-white/5">
                  <td className="p-3 text-gray-200">{s.name}{s.official ? <span className="mr-2 text-[10px] text-[#e0c58a]">رسمي</span> : null}</td>
                  <td className="p-3 font-mono text-gray-400">{s.kind}</td>
                  <td className="p-3">
                    <span className={`inline-flex items-center gap-1 ${s.last_status === 'ok' ? 'text-emerald-400' : 'text-red-400'}`}>
                      {s.last_status === 'ok' ? <CheckCircle2 className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
                      {s.last_status || '—'}
                    </span>
                  </td>
                  <td className="p-3 text-gray-400">{fmtDate(s.last_success_at)}</td>
                  <td className="p-3 font-mono">{s.last_entries}</td>
                  <td className="p-3 font-mono">{s.stored}</td>
                  <td className="p-3 text-red-300/80 max-w-[240px] truncate" title={s.last_error || ''}>{s.last_error || ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* DETAIL MODAL */}
      {selected && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-start justify-center p-4 overflow-y-auto" onClick={() => setSelected(null)}>
          <div className="w-full max-w-3xl bg-[#0a0b12] border border-[#c5a059]/30 rounded-2xl p-6 space-y-5 my-8" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <span className="px-2.5 py-1 rounded-lg text-[10px] font-bold bg-[#c5a059]/15 text-[#e0c58a] border border-[#c5a059]/30">{selected.fraud_category_ar}</span>
                <h2 className="text-lg font-bold text-white mt-3 leading-snug">{selected.headline || selected.title}</h2>
                <p className="text-[11px] text-gray-500 mt-1">{selected.source} • نُشر {fmtDate(selected.published_at)} • سُحب {fmtDate(selected.fetched_at)}</p>
              </div>
              <button onClick={() => setSelected(null)} className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-gray-300"><X className="w-4 h-4" /></button>
            </div>

            {selected.original_url && (
              <a href={selected.original_url} target="_blank" rel="noopener noreferrer" dir="ltr" className="flex items-center gap-2 text-[11px] text-[#e0c58a] break-all hover:underline">
                <ExternalLink className="w-3.5 h-3.5 shrink-0" /> {selected.original_url}
              </a>
            )}

            {/* Facts */}
            <section className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/20 space-y-2">
              <h3 className="text-xs font-bold text-emerald-300 flex items-center gap-2"><CheckCircle2 className="w-4 h-4" /> حقائق وردت في المصدر</h3>
              {selected.verified_facts?.incident_summary && <p className="text-xs text-gray-200 leading-relaxed">{selected.verified_facts.incident_summary}</p>}
              {(selected.verified_facts?.reported_method_sentences || []).map((m: string, i: number) => (
                <p key={i} className="text-[11px] text-gray-300 leading-relaxed border-r-2 border-emerald-500/40 pr-2">{m}</p>
              ))}
              <div className="text-[11px] text-gray-400 flex flex-wrap gap-x-6 gap-y-1">
                {selected.channels?.length > 0 && <span>القنوات: <b className="text-gray-200">{selected.channels.join('، ')}</b></span>}
                {selected.target_group?.length > 0 && <span>الفئة المستهدفة: <b className="text-gray-200">{selected.target_group.join('، ')}</b></span>}
                {(selected.verified_facts?.agencies_mentioned || []).length > 0 && <span>الجهات: <b className="text-gray-200">{selected.verified_facts.agencies_mentioned.join('، ')}</b></span>}
              </div>
              <p className="text-[10px] text-gray-500">أساس التحليل: {selected.verified_facts?.text_basis === 'title+summary_only' ? 'العنوان والملخص فقط' : 'نص المقال'}</p>
            </section>

            {/* Inference */}
            <section className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/20 space-y-2">
              <h3 className="text-xs font-bold text-amber-300 flex items-center gap-2"><FlaskConical className="w-4 h-4" /> استنتاجات النظام (ليست حقائق مؤكدة)</h3>
              {selected.modus_operandi && <p className="text-xs text-gray-200 leading-relaxed">{selected.modus_operandi}</p>}
              {selected.analytical_assessment?.bank_relevance?.reason && (
                <p className="text-[11px] text-gray-300">الصلة بالبنوك السعودية ({selected.bank_relevance}): {selected.analytical_assessment.bank_relevance.reason}</p>
              )}
              <p className="text-[11px] text-gray-400 font-mono">
                ثقة المصدر {selected.evidence_confidence} • ثقة التحليل {selected.analysis_confidence}
              </p>
              {selected.analytical_assessment?.limitations && <p className="text-[10px] text-gray-500">{selected.analytical_assessment.limitations}</p>}
            </section>

            {/* Indicators */}
            {selected.detection_indicators && (
              <section className="space-y-2">
                <h3 className="text-xs font-bold text-white flex items-center gap-2"><ListChecks className="w-4 h-4 text-[#c5a059]" /> مؤشرات المخاطر القابلة للرصد</h3>
                <div className="grid md:grid-cols-2 gap-3 text-[11px]">
                  <div className="p-3 rounded-lg bg-black/40 border border-white/5">
                    <b className="text-gray-200">مذكورة في المصدر</b>
                    <ul className="list-disc pr-5 mt-1 text-gray-400">{selected.detection_indicators.mentioned_in_source.map((x, i) => <li key={i}>{x}</li>)}</ul>
                  </div>
                  <div className="p-3 rounded-lg bg-black/40 border border-white/5">
                    <b className="text-gray-200">مقترحة من النظام</b>
                    <ul className="list-disc pr-5 mt-1 text-gray-400">{selected.detection_indicators.suggested_by_system.map((x, i) => <li key={i}>{x}</li>)}</ul>
                  </div>
                </div>
              </section>
            )}

            {/* Rule */}
            {selected.detection_rule && (
              <section className="p-4 rounded-xl bg-black/40 border border-white/10 space-y-2">
                <div className="flex justify-between items-center">
                  <h3 className="text-xs font-bold text-white">{selected.detection_rule.name}</h3>
                  <span className="text-[10px] font-mono text-amber-300">{selected.detection_rule.rule_id} • {selected.detection_rule.status}</span>
                </div>
                <p className="text-[11px] text-gray-400">{selected.detection_rule.hypothesis}</p>
                <pre dir="ltr" className="text-[11px] font-mono whitespace-pre-wrap text-emerald-300 bg-black/60 rounded-lg p-3">{selected.detection_rule.logic}</pre>
                <div className="text-[11px] text-gray-400">
                  <b className="text-gray-200">البيانات المطلوبة للاختبار:</b>
                  <ul className="list-disc pr-5 mt-1">{selected.detection_rule.required_data.map((x, i) => <li key={i}>{x}</li>)}</ul>
                  <b className="text-gray-200 block mt-2">خطة التحقق:</b>
                  <ul className="list-disc pr-5 mt-1">{selected.detection_rule.validation.map((x, i) => <li key={i}>{x}</li>)}</ul>
                </div>
              </section>
            )}

            {selected.preventive_recommendations?.items?.length ? (
              <section className="text-[11px] text-gray-400">
                <b className="text-gray-200">توصيات وقائية (قالب النظام):</b>
                <ul className="list-disc pr-5 mt-1 space-y-0.5">{selected.preventive_recommendations.items.map((x, i) => <li key={i}>{x}</li>)}</ul>
              </section>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
};

export default SaudiFraud;
