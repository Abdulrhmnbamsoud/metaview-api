// src/pages/SourcesAndSystem.tsx
// الوجهة الرابعة: المصادر والنظام — إدارة مصادر الرصد، تشخيص النشر، والتحكم في الاستيعاب
import React, { useCallback, useEffect, useState } from 'react';
import {
  Radio,
  HeartPulse,
  Activity,
  HardDrive,
  Timer,
  CheckCircle2,
  AlertTriangle,
  Play,
  RefreshCw,
  ExternalLink,
} from 'lucide-react';
import { apiService } from '../services/api';
import { SaudiSourceInfo, SaudiRuntime } from '../types/saudi';
import { HealthStatus, IngestRunResult } from '../types';
import { Chip, LoadingBlock, ErrorBlock, Unavailable, fmtDateTime } from '../components/ui';
import { timeAgo } from '../utils/formatDate';
import { useAuth } from '../context/AuthContext';

type Tab = 'sources' | 'system' | 'ingest';

export const SourcesAndSystem: React.FC = () => {
  const { can, role } = useAuth();
  const canTriggerIngest = can('ingestion.trigger') || role === 'admin';
  const [tab, setTab] = useState<Tab>('sources');
  const [sources, setSources] = useState<SaudiSourceInfo[]>([]);
  const [runtime, setRuntime] = useState<SaudiRuntime | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Ingest trigger state
  const [isIngesting, setIsIngesting] = useState(false);
  const [lastRunResult, setLastRunResult] = useState<IngestRunResult | null>(null);
  const [ingestMsg, setIngestMsg] = useState<string | null>(null);

  // Filter state (must be declared with all hooks before any early returns)
  const [sourceKindFilter, setSourceKindFilter] = useState<'all' | 'x_social' | 'official' | 'news'>('all');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [srcRes, rtRes, hlthRes] = await Promise.all([
        apiService.getSaudiSources(),
        apiService.getSaudiRuntime(),
        apiService.getHealth(),
      ]);
      setSources(srcRes.sources || []);
      setRuntime(rtRes);
      setHealth(hlthRes);
    } catch (e: any) {
      setError(e?.message || 'خطأ في جلب بيانات النظام');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleRunIngest = async () => {
    if (isIngesting) return;
    setIsIngesting(true);
    setIngestMsg(null);
    try {
      const res = await apiService.runIngest(20);
      setLastRunResult(res);
      setIngestMsg(`اكتملت دورة الاستيعاب بنجاح: تم جلب ${res.inserted_total} مقالاً جديداً.`);
      await load();
    } catch (e: any) {
      setIngestMsg(`تعذر إكمال الاستيعاب: ${e.message}`);
    } finally {
      setIsIngesting(false);
    }
  };

  if (loading && !runtime) return <LoadingBlock label="جارٍ تحميل معلومات المصادر والنظام…" />;
  if (error && !runtime) return <ErrorBlock message={error} onRetry={load} />;

  const filteredSources = sources.filter((s) => {
    if (sourceKindFilter === 'x_social') return s.kind === 'x_social';
    if (sourceKindFilter === 'official') return s.official === 1;
    if (sourceKindFilter === 'news') return s.kind !== 'x_social' && s.official !== 1;
    return true;
  });

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="space-y-1 pb-3 border-b border-stone-200 dark:border-slate-800">
        <h1 className="text-2xl font-bold tracking-tight text-stone-900 dark:text-stone-100">
          المصادر وحالة النظام
        </h1>
        <p className="text-sm text-stone-600 dark:text-slate-400">
          متابعة مصادر الأخبار والبلاغات، صحة خادم البيانات، والتحكم في دورات الاستيعاب
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-stone-200 dark:border-slate-800 text-xs">
        {[
          { key: 'sources', label: `مصادر الرصد (${sources.length})`, icon: Radio },
          { key: 'system', label: 'صحة النظام والتخزين', icon: HeartPulse },
          { key: 'ingest', label: 'مراقبة الاستيعاب الفوري', icon: Activity },
        ].map((t) => {
          const Icon = t.icon;
          const active = tab === t.key;
          return (
            <button
              key={t.key}
              onClick={() => setTab(t.key as Tab)}
              className={`flex items-center gap-2 px-4 py-2.5 font-medium border-b-2 transition-colors ${
                active
                  ? 'border-sky-600 text-sky-600 dark:border-sky-400 dark:text-sky-400'
                  : 'border-transparent text-stone-500 hover:text-stone-800 dark:text-slate-400 dark:hover:text-slate-200'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{t.label}</span>
            </button>
          );
        })}
      </div>

      {/* TAB 1: Sources */}
      {tab === 'sources' && (
        <div className="space-y-4">
          {/* Sub-filter chips */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-1.5 text-xs">
              {[
                { key: 'all', label: `كافة المصادر (${sources.length})` },
                { key: 'x_social', label: `منصات X والتواصل (${sources.filter((s) => s.kind === 'x_social').length})` },
                { key: 'official', label: `جهات رسمية (${sources.filter((s) => s.official === 1).length})` },
                { key: 'news', label: `صحف وموجزات (${sources.filter((s) => s.kind !== 'x_social' && s.official !== 1).length})` },
              ].map((f) => (
                <button
                  key={f.key}
                  onClick={() => setSourceKindFilter(f.key as any)}
                  className={`px-3 py-1 rounded-lg border text-xs font-medium transition-colors ${
                    sourceKindFilter === f.key
                      ? 'bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border-sky-300 dark:border-sky-800 font-semibold'
                      : 'border-stone-200 dark:border-slate-800 text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-stone-200'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>

            <span className="text-xs text-stone-400 font-mono">
              إجمالي {filteredSources.length} مصدر
            </span>
          </div>

          <div className="panel overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-right text-xs">
                <thead>
                  <tr className="bg-stone-50 dark:bg-slate-900/60 text-stone-500 dark:text-slate-400 border-b border-stone-200 dark:border-slate-800 font-mono text-[11px]">
                    <th className="py-3 px-4">اسم المصدر والمعرّف</th>
                    <th className="py-3 px-4">النوع</th>
                    <th className="py-3 px-4">نطاق التغطية</th>
                    <th className="py-3 px-4">حالة الاتصال</th>
                    <th className="py-3 px-4">آخر جلب ناجح</th>
                    <th className="py-3 px-4">المقالات المخزنة</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100 dark:divide-slate-800/60">
                  {filteredSources.map((s) => {
                    const isX = s.kind === 'x_social';
                    const isReqInt = s.last_status === 'requires_integration';
                    const ok = s.last_status === 'ok';
                    const never = !s.last_status;
                    return (
                      <tr key={s.code} className="hover:bg-stone-50/60 dark:hover:bg-slate-800/40">
                        <td className="py-3 px-4 font-semibold text-stone-900 dark:text-stone-100">
                          <div className="flex flex-col gap-0.5">
                            <div className="flex items-center gap-1.5">
                              <span>{s.name}</span>
                              <a
                                href={s.url}
                                target="_blank"
                                rel="noreferrer"
                                className="text-stone-400 hover:text-sky-600 p-0.5"
                                title="فتح الرابط الأصلي للمصدر"
                              >
                                <ExternalLink className="w-3 h-3" />
                              </a>
                            </div>
                            {s.handle && (
                              <span className="text-[11px] font-mono text-stone-500 dark:text-slate-400 font-normal" dir="ltr">
                                {s.handle}
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          {isX ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-black text-white dark:bg-stone-800 dark:text-stone-200">
                              <span className="font-bold">𝕏</span>
                              <span>شبكات اجتماعية</span>
                            </span>
                          ) : (
                            <Chip tone={s.official ? 'brand' : 'plain'}>
                              {s.official ? 'رسمي حكومي' : 'صحيفة إعلامية'}
                            </Chip>
                          )}
                        </td>
                        <td className="py-3 px-4">
                          <span className="inline-flex items-center gap-1 text-[11px] font-mono text-stone-600 dark:text-slate-300">
                            {s.country === 'SA' ? '🇸🇦 السعودية' : '🌐 العالمي'}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          {isReqInt ? (
                            <div className="space-y-0.5">
                              <span className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-400 font-mono font-medium text-[11px]">
                                <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                                يتطلب ربط API (Requires Integration)
                              </span>
                              <p className="text-[10px] text-stone-500 dark:text-slate-400 max-w-xs leading-tight">
                                {s.last_error || 'يتطلب إعداد X API Bearer Token للوصول المصرح.'}
                              </p>
                            </div>
                          ) : never ? (
                            <span className="text-stone-400 text-[11px] font-mono">لم يُجلب</span>
                          ) : ok ? (
                            <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-mono font-medium text-[11px]">
                              <CheckCircle2 className="w-3.5 h-3.5" />
                              يعمل (OK)
                            </span>
                          ) : (
                            <div className="space-y-0.5">
                              <span className="inline-flex items-center gap-1 text-red-600 dark:text-red-400 font-mono font-medium text-[11px]">
                                <AlertTriangle className="w-3.5 h-3.5" />
                                تعذر الجلب
                              </span>
                              {s.last_error && (
                                <p className="text-[10px] text-red-500 max-w-xs leading-tight">{s.last_error}</p>
                              )}
                            </div>
                          )}
                        </td>
                        <td className="py-3 px-4 font-mono text-stone-500 dark:text-slate-400">
                          {s.last_success_at ? timeAgo(s.last_success_at) : '—'}
                        </td>
                        <td className="py-3 px-4 font-mono text-stone-700 dark:text-slate-300 font-semibold">
                          {s.stored}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: System Health */}
      {tab === 'system' && runtime && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            <div className="panel p-4 space-y-1">
              <span className="text-[11px] font-mono text-stone-500 dark:text-slate-400">حالة الواجهة (API)</span>
              <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4" />
                <span>متصلة ({health?.rows || 0} خبر)</span>
              </div>
            </div>

            <div className="panel p-4 space-y-1">
              <span className="text-[11px] font-mono text-stone-500 dark:text-slate-400">جدولة التحديث</span>
              <div className="text-lg font-bold text-stone-900 dark:text-stone-100 flex items-center gap-1.5">
                <Timer className="w-4 h-4 text-sky-600" />
                <span>{runtime.scheduler?.thread_alive ? 'نشطة تلقائياً' : 'متوقفة'}</span>
              </div>
            </div>

            <div className="panel p-4 space-y-1">
              <span className="text-[11px] font-mono text-stone-500 dark:text-slate-400">بيئة التشغيل</span>
              <div className="text-lg font-bold text-stone-900 dark:text-stone-100">
                {runtime.on_render ? 'سحابة Render' : 'تشغيل محلي'}
              </div>
            </div>

            <div className="panel p-4 space-y-1">
              <span className="text-[11px] font-mono text-stone-500 dark:text-slate-400">إصدار الكود (Commit)</span>
              <div className="text-sm font-mono text-stone-900 dark:text-stone-100 font-bold truncate">
                {runtime.commit ? runtime.commit.slice(0, 8) : 'غير معلن'}
              </div>
            </div>
          </div>

          <div className="panel p-5 space-y-3">
            <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100 flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-sky-600" />
              <span>مساحة التخزين وقاعدة البيانات (SQLite)</span>
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-stone-50 dark:bg-slate-900/50 border border-stone-200 dark:border-slate-800 space-y-1">
                <span className="text-stone-500 dark:text-slate-400 block">مسار الملف:</span>
                <span className="font-mono text-stone-800 dark:text-stone-200 block truncate" dir="ltr">
                  {runtime.disk?.db_path || '—'}
                </span>
              </div>

              <div className="p-3 rounded-lg bg-stone-50 dark:bg-slate-900/50 border border-stone-200 dark:border-slate-800 space-y-1">
                <span className="text-stone-500 dark:text-slate-400 block">نمط كتابة السجلات:</span>
                <span className="font-mono text-stone-800 dark:text-stone-200 font-bold block">
                  {runtime.sqlite_journal_mode || 'WAL'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: Ingestion Operations */}
      {tab === 'ingest' && (
        <div className="space-y-4">
          <div className="panel p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-bold text-stone-900 dark:text-stone-100">
                  تشغيل استيعاب فوري للمقالات
                </h3>
                <p className="text-xs text-stone-500 dark:text-slate-400 mt-0.5">
                  يسحب أحدث الأخبار من المصادر السعودية المسجلة ويقوم بتصنيفها وفرز الحوادث تلقائياً
                </p>
              </div>

              <div className="flex flex-col sm:items-end gap-1">
                <button
                  onClick={handleRunIngest}
                  disabled={isIngesting || !canTriggerIngest}
                  title={!canTriggerIngest ? 'يتطلب تشغيل السحب صلاحيات مدير المنصة (Admin)' : 'بدء الاستيعاب الآن'}
                  className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
                    isIngesting || !canTriggerIngest
                      ? 'bg-stone-200 dark:bg-slate-800 text-stone-400 dark:text-slate-500 cursor-not-allowed border border-stone-300 dark:border-slate-700'
                      : 'bg-sky-600 hover:bg-sky-700 text-white shadow-sm'
                  }`}
                >
                  <Play className={`w-3.5 h-3.5 fill-current ${isIngesting ? 'animate-pulse' : ''}`} />
                  <span>
                    {isIngesting
                      ? 'جارٍ السحب والفحص…'
                      : !canTriggerIngest
                      ? 'سحب فوري (مدير المنصة فقط)'
                      : 'بدء السحب الآن'}
                  </span>
                </button>
                {!canTriggerIngest && (
                  <span className="text-[10px] text-stone-400 dark:text-slate-500">
                    * مقتصر على حسابات الأدمن
                  </span>
                )}
              </div>
            </div>

            {ingestMsg && (
              <div className="p-3 rounded-lg bg-sky-50 dark:bg-sky-950/40 border border-sky-200 dark:border-sky-800 text-xs text-sky-800 dark:text-sky-300">
                {ingestMsg}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default SourcesAndSystem;
