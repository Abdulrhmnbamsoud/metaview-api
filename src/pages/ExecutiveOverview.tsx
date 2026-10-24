// src/pages/ExecutiveOverview.tsx
// Home Page: Information-First search-centric experience (Linear/Notion style)
import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  Search,
  ShieldAlert,
  Network,
  AlertTriangle,
  Clock,
  RefreshCw,
  ExternalLink,
  ArrowRight,
  CheckCircle2,
  MapPin,
  Building2,
  ChevronLeft,
} from 'lucide-react';
import { apiService } from '../services/api';
import {
  SaudiOverview,
  SaudiIncidentReport,
  SaudiIncident,
  SaudiPatternsResponse,
  SaudiPattern,
} from '../types/saudi';
import { Chip, EpistemicChip, LoadingBlock, ErrorBlock, fmtDay } from '../components/ui';
import { IncidentDetailModal } from '../components/IncidentDetailModal';
import { timeAgo } from '../utils/formatDate';

export const ExecutiveOverview: React.FC = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [ov, setOv] = useState<SaudiOverview | null>(null);
  const [report, setReport] = useState<SaudiIncidentReport | null>(null);
  const [incidents, setIncidents] = useState<SaudiIncident[]>([]);
  const [patterns, setPatterns] = useState<SaudiPattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Selected incident for detail view
  const [selectedIncident, setSelectedIncident] = useState<SaudiIncident | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [o, r, inc, p] = await Promise.all([
        apiService.getSaudiOverview(),
        apiService.getSaudiIncidentReport(),
        apiService.getSaudiIncidents({ page_size: 6 }),
        apiService.getSaudiPatterns(),
      ]);
      setOv(o);
      setReport(r);
      setIncidents(inc.items);
      setPatterns(p.patterns);
    } catch (e: any) {
      setError(e?.message || 'خطأ في جلب البيانات');
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

  const handleRefresh = async () => {
    if (refreshing) return;
    setRefreshing(true);
    try {
      await apiService.runIngest(15);
      await load();
    } catch (e) {
      console.error(e);
    } finally {
      setRefreshing(false);
    }
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/explore?q=${encodeURIComponent(searchQuery.trim())}`);
    } else {
      navigate('/explore');
    }
  };

  if (loading && !ov) return <LoadingBlock label="جارٍ تحميل لوحة الاستخبارات…" />;
  if (error && !ov) return <ErrorBlock message={error} onRetry={load} />;
  if (!ov || !report) return null;

  const reviewItems = incidents.filter((i) => i.needs_review || i.review_status === 'needs_review');

  return (
    <div className="space-y-8 max-w-4xl mx-auto py-2">
      {/* 1. Unobtrusive last-updated & status strip */}
      <div className="flex items-center justify-between text-xs text-stone-500 dark:text-slate-400 font-mono">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500" />
          <span>المصادر: {ov.sources_ok}/{ov.sources_total} نشطة</span>
          <span>•</span>
          <span>آخر تحديث: {timeAgo(ov.last_fetch_at)}</span>
        </div>

        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center gap-1.5 hover:text-stone-800 dark:hover:text-stone-200 transition-colors"
          title="مزامنة فورية"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin text-sky-600' : ''}`} />
          <span>{refreshing ? 'جارٍ المزامنة…' : 'تحديث'}</span>
        </button>
      </div>

      {/* 2. Primary Search Box (Linear / Modern Search style) */}
      <div className="space-y-3 text-center sm:text-right">
        <div className="space-y-1">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-stone-900 dark:text-stone-100">
            استخبارات الاحتيال المالي
          </h1>
          <p className="text-sm text-stone-600 dark:text-slate-400">
            البحث المباشر والوصول السريع للأدلة، الحوادث، وأنماط الاستهداف في القطاع البنكي
          </p>
        </div>

        <form onSubmit={handleSearchSubmit} className="relative mt-4">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="ابحث عن حادثة احتيال، أسلوب، جهة، أو مؤشر خطر..."
            className="w-full bg-white dark:bg-[#111726] border border-stone-300 dark:border-slate-800 rounded-2xl py-4 pr-12 pl-28 text-base text-stone-900 dark:text-stone-100 placeholder:text-stone-400 dark:placeholder:text-slate-500 shadow-sm focus:border-sky-500 dark:focus:border-sky-500 focus:ring-4 focus:ring-sky-500/10 outline-none transition-all"
            autoFocus
          />
          <Search className="w-5 h-5 text-stone-400 dark:text-slate-500 absolute top-1/2 -translate-y-1/2 right-4 pointer-events-none" />
          <button
            type="submit"
            className="absolute top-1/2 -translate-y-1/2 left-3 px-4 py-2 rounded-xl bg-stone-900 dark:bg-stone-100 text-white dark:text-stone-900 font-semibold text-xs hover:bg-stone-800 dark:hover:bg-white transition-colors"
          >
            بحث
          </button>
        </form>

        {/* Quick query chips */}
        <div className="flex flex-wrap items-center gap-1.5 pt-1 text-xs text-stone-500 dark:text-slate-400">
          <span className="text-[11px] font-mono">اقتراحات سريعة:</span>
          {['الرياض', 'سريع', 'انتحال جهات حكومية', 'حسابات وسيطة (Mule)', 'مدى'].map((tag) => (
            <button
              key={tag}
              type="button"
              onClick={() => navigate(`/explore?q=${encodeURIComponent(tag)}`)}
              className="px-2 py-0.5 rounded-md bg-stone-100 dark:bg-slate-800/80 hover:bg-stone-200 dark:hover:bg-slate-700 text-stone-700 dark:text-slate-300 text-xs transition-colors"
            >
              {tag}
            </button>
          ))}
        </div>
      </div>

      {/* 3. Items Requiring Analyst Review */}
      {reviewItems.length > 0 ? (
        <section className="p-4 rounded-xl bg-amber-50/70 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/40 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-amber-900 dark:text-amber-200 font-semibold text-xs">
              <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400" />
              <span>تنبيه المراجعة: {reviewItems.length} حوادث بانتظار قرار المحلل البشري</span>
            </div>
            <Link to="/explore?type=incident" className="text-xs text-amber-800 dark:text-amber-300 font-medium hover:underline">
              عرض في الاستكشاف ←
            </Link>
          </div>
          <div className="space-y-1.5 pt-1">
            {reviewItems.map((item) => (
              <div
                key={item.id}
                onClick={() => setSelectedIncident(item)}
                className="flex items-center justify-between p-2 rounded-lg bg-white/80 dark:bg-slate-900/60 border border-amber-200/60 dark:border-amber-900/30 text-xs cursor-pointer hover:bg-white transition-colors"
              >
                <span className="text-stone-800 dark:text-stone-200 font-medium">{item.title}</span>
                <span className="text-stone-400 font-mono text-[11px]">{item.ref}</span>
              </div>
            ))}
          </div>
        </section>
      ) : (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-emerald-50/60 dark:bg-emerald-950/20 border border-emerald-200/60 dark:border-emerald-900/30 text-xs text-emerald-800 dark:text-emerald-300">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
          <span>حالة التحقق: كافة الحوادث المستقلة مدققة ومطابقة، ولا توجد اقتراحات دمج معلّقة حالياً.</span>
        </div>
      )}

      {/* 4. Important Recent Incidents */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-sky-600 dark:text-sky-400" />
            <h2 className="text-base font-bold text-stone-900 dark:text-stone-100">
              أهم الحوادث المرصودة حديثاً
            </h2>
          </div>
          <Link to="/explore?type=incident" className="text-xs text-sky-600 dark:text-sky-400 hover:underline font-medium">
            عرض كافة الحوادث ({report.incidents.independent_total}) ←
          </Link>
        </div>

        <div className="space-y-2.5">
          {incidents.slice(0, 4).map((i) => (
            <div
              key={i.id}
              onClick={() => setSelectedIncident(i)}
              className="panel p-4 hover:border-sky-300 dark:hover:border-sky-800/80 transition-all cursor-pointer space-y-2 group"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-semibold text-stone-600 dark:text-slate-400">
                    {i.ref}
                  </span>
                  <Chip tone={i.record_type === 'confirmed_incident' ? 'critical' : 'high'}>
                    {i.record_type_ar}
                  </Chip>
                  {i.bank_relevance === 'high' && (
                    <Chip tone="critical">صلة بنكية عالية</Chip>
                  )}
                </div>
                <span className="text-xs font-mono text-stone-400 dark:text-slate-500">
                  {fmtDay(i.first_reported_at)}
                </span>
              </div>

              <div className="flex items-start justify-between gap-3">
                <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100 group-hover:text-sky-600 dark:group-hover:text-sky-400 transition-colors leading-snug">
                  {i.title}
                </h3>
                <ChevronLeft className="w-4 h-4 text-stone-400 group-hover:text-sky-600 dark:group-hover:text-sky-400 shrink-0 mt-0.5 transition-transform group-hover:-translate-x-1" />
              </div>

              <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-xs text-stone-500 dark:text-slate-400">
                <div className="flex items-center gap-3">
                  {i.locations.length > 0 && (
                    <span className="flex items-center gap-1">
                      <MapPin className="w-3 h-3 text-sky-600 dark:text-sky-400" />
                      {i.locations.join('، ')}
                    </span>
                  )}
                  {i.agencies.length > 0 && (
                    <span className="flex items-center gap-1">
                      <Building2 className="w-3 h-3" />
                      {i.agencies[0]}
                    </span>
                  )}
                </div>

                <span className="font-mono text-[11px] bg-stone-100 dark:bg-slate-800 px-2 py-0.5 rounded text-stone-600 dark:text-slate-300">
                  {i.article_count} مصادر إخبارية مؤيدة
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 5. Newly Identified Fraud Patterns */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Network className="w-4 h-4 text-sky-600 dark:text-sky-400" />
            <h2 className="text-base font-bold text-stone-900 dark:text-stone-100">
              أنماط الاحتيال المكتشفة
            </h2>
          </div>
          <Link to="/explore?type=pattern" className="text-xs text-sky-600 dark:text-sky-400 hover:underline font-medium">
            عرض كافة الأنماط ({patterns.length}) ←
          </Link>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {patterns.slice(0, 4).map((p) => (
            <Link
              key={p.key}
              to={`/explore?q=${encodeURIComponent(p.label_ar)}`}
              className="panel p-3.5 hover:border-sky-300 dark:hover:border-sky-800/80 transition-all space-y-1.5 group"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-stone-800 dark:text-stone-200 group-hover:text-sky-600 dark:group-hover:text-sky-400 transition-colors">
                  {p.label_ar}
                </span>
                <span className="font-mono text-xs text-sky-600 dark:text-sky-400 font-bold">
                  {p.incident_count} حوادث
                </span>
              </div>
              <p className="text-[11px] text-stone-500 dark:text-slate-400 line-clamp-1">
                القنوات: {p.channels.join('، ') || 'قنوات مصرفية متعددة'}
              </p>
            </Link>
          ))}
        </div>
      </section>

      {/* Incident Detail Modal */}
      <IncidentDetailModal
        incident={selectedIncident}
        onClose={() => setSelectedIncident(null)}
      />
    </div>
  );
};

export default ExecutiveOverview;
