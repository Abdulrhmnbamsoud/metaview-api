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

type Tab = 'sources' | 'system' | 'ingest';

export const SourcesAndSystem: React.FC = () => {
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
          <div className="panel overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-right text-xs">
                <thead>
                  <tr className="bg-stone-50 dark:bg-slate-900/60 text-stone-500 dark:text-slate-400 border-b border-stone-200 dark:border-slate-800 font-mono text-[11px]">
                    <th className="py-3 px-4">اسم المصدر</th>
                    <th className="py-3 px-4">النوع</th>
                    <th className="py-3 px-4">الحالة الأخيرة</th>
                    <th className="py-3 px-4">آخر جلب ناجح</th>
                    <th className="py-3 px-4">المقالات المخزنة</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100 dark:divide-slate-800/60">
                  {sources.map((s) => {
                    const ok = s.last_status === 'ok';
                    const never = !s.last_status;
                    return (
                      <tr key={s.code} className="hover:bg-stone-50/60 dark:hover:bg-slate-800/40">
                        <td className="py-3 px-4 font-semibold text-stone-900 dark:text-stone-100">
                          <div className="flex items-center gap-1.5">
                            <span>{s.name}</span>
                            <a
                              href={s.url}
                              target="_blank"
                              rel="noreferrer"
                              className="text-stone-400 hover:text-sky-600 p-0.5"
                            >
                              <ExternalLink className="w-3 h-3" />
                            </a>
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <Chip tone={s.official ? 'brand' : 'plain'}>
                            {s.official ? 'رسمي حكومي' : 'صحيفة إعلامية'}
                          </Chip>
                        </td>
                        <td className="py-3 px-4">
                          {never ? (
                            <span className="text-stone-400 text-[11px] font-mono">لم يُجلب</span>
                          ) : ok ? (
                            <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-mono font-medium">
                              <CheckCircle2 className="w-3.5 h-3.5" />
                              يعمل (OK)
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-red-600 dark:text-red-400 font-mono font-medium">
                              <AlertTriangle className="w-3.5 h-3.5" />
                              تعذر الجلب
                            </span>
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
                <span>{runtime.scheduler.thread_alive ? 'نشطة تلقائياً' : 'متوقفة'}</span>
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
                  {runtime.disk.db_path}
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

              <button
                onClick={handleRunIngest}
                disabled={isIngesting}
                className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
                  isIngesting
                    ? 'bg-stone-300 dark:bg-slate-800 text-stone-500 cursor-not-allowed'
                    : 'bg-sky-600 hover:bg-sky-700 text-white shadow-sm'
                }`}
              >
                <Play className={`w-3.5 h-3.5 fill-current ${isIngesting ? 'animate-pulse' : ''}`} />
                <span>{isIngesting ? 'جارٍ السحب والفحص…' : 'بدء السحب الآن'}</span>
              </button>
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
