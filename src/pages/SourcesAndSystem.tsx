// src/pages/SourcesAndSystem.tsx
// الوجهة الرابعة: المصادر والنظام — إدارة مصادر الرصد، تكامل رصد X العام، تشخيص النشر، والتحكم في الاستيعاب
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
  ShieldAlert,
  Search,
  Key,
  Info,
  Sparkles,
  Link2,
  Share2,
} from 'lucide-react';
import { apiService } from '../services/api';
import {
  SaudiSourceInfo,
  SaudiRuntime,
  XPublicCollectorStatus,
  XPublicPostDetail,
  XPublicCollectionReport,
} from '../types/saudi';
import { HealthStatus, IngestRunResult } from '../types';
import { Chip, LoadingBlock, ErrorBlock, fmtDateTime } from '../components/ui';
import { timeAgo } from '../utils/formatDate';
import { useAuth } from '../context/AuthContext';

type Tab = 'sources' | 'x_collector' | 'system' | 'ingest';

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

  // X Collector state
  const [xStatus, setXStatus] = useState<XPublicCollectorStatus | null>(null);
  const [isCollectingX, setIsCollectingX] = useState(false);
  const [xCollectionReport, setXCollectionReport] = useState<XPublicCollectionReport | null>(null);
  const [xCollectionMsg, setXCollectionMsg] = useState<string | null>(null);

  // Single post link analyzer
  const [singlePostUrl, setSinglePostUrl] = useState('');
  const [isAnalyzingPost, setIsAnalyzingPost] = useState(false);
  const [analyzedPost, setAnalyzedPost] = useState<XPublicPostDetail | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Filter state
  const [sourceKindFilter, setSourceKindFilter] = useState<'all' | 'x_social' | 'official' | 'news'>('all');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [srcRes, rtRes, hlthRes, xStatRes] = await Promise.all([
        apiService.getSaudiSources(),
        apiService.getSaudiRuntime(),
        apiService.getHealth(),
        apiService.getXCollectorStatus().catch(() => null),
      ]);
      setSources(srcRes.sources || []);
      setRuntime(rtRes);
      setHealth(hlthRes);
      if (xStatRes) setXStatus(xStatRes);
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

  const handleTriggerXCollection = async () => {
    if (isCollectingX) return;
    setIsCollectingX(true);
    setXCollectionMsg(null);
    try {
      const res = await apiService.triggerXPublicCollection('K_H_AlMuzaini');
      setXCollectionReport(res);
      if (res.status === 'active') {
        setXCollectionMsg(`اكتمل الجلب العام بنجاح: تم جلب ${res.total_fetched} منشوراً، وحفظ ${res.new_stored} جديداً، وتخطي ${res.duplicates_skipped} مكرراً.`);
      } else if (res.status === 'access_restricted') {
        setXCollectionMsg(`الوصول مقيد (Access Restricted): تجاوز معدل الطلبات من قِبل X. لم يتم تجاوز أي حماية.`);
      } else {
        setXCollectionMsg(`حالة المصدر: ${res.status_label || res.status}. ${res.error || ''}`);
      }
      await load();
    } catch (e: any) {
      setXCollectionMsg(`تعذر إكمال الجلب: ${e.message}`);
    } finally {
      setIsCollectingX(false);
    }
  };

  const handleAnalyzeSinglePost = async () => {
    if (!singlePostUrl.trim() || isAnalyzingPost) return;
    setIsAnalyzingPost(true);
    setAnalysisError(null);
    setAnalyzedPost(null);
    try {
      const res = await apiService.analyzeSingleXPost(singlePostUrl.trim());
      setAnalyzedPost(res.post);
    } catch (e: any) {
      setAnalysisError(e.message || 'تعذر تحليل المنشور المحدد');
    } finally {
      setIsAnalyzingPost(false);
    }
  };

  const renderSourceStatus = (s: SaudiSourceInfo) => {
    const st = s.last_status?.toLowerCase();
    if (st === 'active' || st === 'ok') {
      return (
        <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-mono font-medium text-[11px]">
          <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
          <span>Active (نشط)</span>
        </span>
      );
    }
    if (st === 'requires_authentication' || st === 'requires_integration') {
      return (
        <div className="space-y-0.5">
          <span className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-400 font-mono font-medium text-[11px]">
            <Key className="w-3.5 h-3.5 shrink-0" />
            <span>Requires Authentication</span>
          </span>
          <p className="text-[10px] text-stone-500 dark:text-slate-400 max-w-xs leading-tight">
            {s.last_error || 'يتطلب إعداد X API Bearer Token للوصول المصرح.'}
          </p>
        </div>
      );
    }
    if (st === 'access_restricted') {
      return (
        <div className="space-y-0.5">
          <span className="inline-flex items-center gap-1 text-orange-600 dark:text-orange-400 font-mono font-medium text-[11px]">
            <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
            <span>Access Restricted (مقيد)</span>
          </span>
          <p className="text-[10px] text-stone-500 dark:text-slate-400 max-w-xs leading-tight">
            {s.last_error || 'تم بلوغ حدود الطلبات أو تقييد الوصول من X وفق سياسات المنصة.'}
          </p>
        </div>
      );
    }
    if (st === 'not_supported') {
      return (
        <div className="space-y-0.5">
          <span className="inline-flex items-center gap-1 text-stone-500 dark:text-slate-400 font-mono font-medium text-[11px]">
            <Info className="w-3.5 h-3.5 shrink-0" />
            <span>Not Supported (غير مدعوم)</span>
          </span>
          {s.last_error && <p className="text-[10px] text-stone-400 max-w-xs leading-tight">{s.last_error}</p>}
        </div>
      );
    }
    if (st === 'failed') {
      return (
        <div className="space-y-0.5">
          <span className="inline-flex items-center gap-1 text-red-600 dark:text-red-400 font-mono font-medium text-[11px]">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
            <span>Failed (تعذر الاتصال)</span>
          </span>
          {s.last_error && <p className="text-[10px] text-red-500 max-w-xs leading-tight">{s.last_error}</p>}
        </div>
      );
    }
    return <span className="text-stone-400 text-[11px] font-mono">لم يُجلب</span>;
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
          متابعة مصادر الأخبار والبلاغات، تكامل منشورات X العامة، صحة الخادم، والتحكم في الاستيعاب
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-stone-200 dark:border-slate-800 text-xs overflow-x-auto">
        {[
          { key: 'sources', label: `مصادر الرصد (${sources.length})`, icon: Radio },
          { key: 'x_collector', label: 'رصد منشورات X العامة', icon: Share2 },
          { key: 'system', label: 'صحة النظام والتخزين', icon: HeartPulse },
          { key: 'ingest', label: 'مراقبة الاستيعاب الفوري', icon: Activity },
        ].map((t) => {
          const Icon = t.icon;
          const active = tab === t.key;
          return (
            <button
              key={t.key}
              onClick={() => setTab(t.key as Tab)}
              className={`flex items-center gap-2 px-4 py-2.5 font-medium border-b-2 transition-colors whitespace-nowrap ${
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
                    <th className="py-3 px-4">حالة الاتصال المباشر</th>
                    <th className="py-3 px-4">آخر جلب ناجح</th>
                    <th className="py-3 px-4">المقالات المخزنة</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-100 dark:divide-slate-800/60">
                  {filteredSources.map((s) => {
                    const isX = s.kind === 'x_social';
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
                          {renderSourceStatus(s)}
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

      {/* TAB 2: X Public Intelligence Collector */}
      {tab === 'x_collector' && (
        <div className="space-y-6">
          {/* Card 1: Monitored Source Overview */}
          <div className="panel p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-stone-200 dark:border-slate-800 pb-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="w-7 h-7 rounded-lg bg-black text-white flex items-center justify-center font-bold text-sm">
                    𝕏
                  </span>
                  <h2 className="text-base font-bold text-stone-900 dark:text-stone-100">
                    رصد منشورات X العامة — د. خالد بن حمد المزيني (@K_H_AlMuzaini)
                  </h2>
                </div>
                <p className="text-xs text-stone-600 dark:text-slate-400 leading-relaxed">
                  جلب وتصنيف المنشورات التوعوية والتحذيرية العامة للاحتيال المالي دون كسر الحماية أو استخدام cookies مسروقة.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleTriggerXCollection}
                  disabled={isCollectingX}
                  className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
                    isCollectingX
                      ? 'bg-stone-200 dark:bg-slate-800 text-stone-400 dark:text-slate-500 cursor-not-allowed'
                      : 'bg-stone-900 hover:bg-black dark:bg-sky-600 dark:hover:bg-sky-700 text-white shadow-sm'
                  }`}
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isCollectingX ? 'animate-spin' : ''}`} />
                  <span>{isCollectingX ? 'جارٍ الجلب العام…' : 'جلب أحدث المنشورات الآن'}</span>
                </button>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-stone-50 dark:bg-slate-900/50 border border-stone-200 dark:border-slate-800 space-y-1">
                <span className="text-stone-500 dark:text-slate-400 block font-mono text-[11px]">حالة الوصول العامة:</span>
                <span className="font-bold flex items-center gap-1.5 text-stone-800 dark:text-stone-200">
                  {xStatus?.status === 'active' ? (
                    <span className="text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Active (نشط ومتاح)
                    </span>
                  ) : xStatus?.status === 'access_restricted' ? (
                    <span className="text-orange-600 dark:text-orange-400 flex items-center gap-1">
                      <ShieldAlert className="w-3.5 h-3.5" /> Access Restricted (مقيد)
                    </span>
                  ) : (
                    <span className="text-amber-600 dark:text-amber-400 flex items-center gap-1">
                      <Key className="w-3.5 h-3.5" /> {xStatus?.status_label || 'Requires Authentication'}
                    </span>
                  )}
                </span>
              </div>

              <div className="p-3 rounded-lg bg-stone-50 dark:bg-slate-900/50 border border-stone-200 dark:border-slate-800 space-y-1">
                <span className="text-stone-500 dark:text-slate-400 block font-mono text-[11px]">إجمالي المنشورات المخزنة:</span>
                <span className="font-bold text-base text-stone-900 dark:text-stone-100">
                  {xStatus?.total_posts_stored ?? 0} منشوراً
                </span>
              </div>

              <div className="p-3 rounded-lg bg-stone-50 dark:bg-slate-900/50 border border-stone-200 dark:border-slate-800 space-y-1">
                <span className="text-stone-500 dark:text-slate-400 block font-mono text-[11px]">آخر جلب ناجح:</span>
                <span className="font-mono text-stone-800 dark:text-stone-200 block truncate">
                  {xStatus?.last_success_at ? fmtDateTime(xStatus.last_success_at) : '—'}
                </span>
              </div>

              <div className="p-3 rounded-lg bg-stone-50 dark:bg-slate-900/50 border border-stone-200 dark:border-slate-800 space-y-1">
                <span className="text-stone-500 dark:text-slate-400 block font-mono text-[11px]">التكامل الرسمي (X API Token):</span>
                <span className="font-semibold text-stone-700 dark:text-slate-300 block">
                  {xStatus?.official_api_token_configured ? 'مُعد (X_BEARER_TOKEN متوفر)' : 'غير مُعد (يعتمد على القنوات العامة)'}
                </span>
              </div>
            </div>

            {xCollectionMsg && (
              <div className="p-3 rounded-lg bg-sky-50 dark:bg-sky-950/40 border border-sky-200 dark:border-sky-800 text-xs text-sky-800 dark:text-sky-300">
                {xCollectionMsg}
              </div>
            )}

            {/* Collection Report if available */}
            {xCollectionReport && (
              <div className="p-4 rounded-lg bg-stone-50 dark:bg-slate-900/40 border border-stone-200 dark:border-slate-800 space-y-3">
                <div className="flex items-center justify-between text-xs font-semibold text-stone-800 dark:text-stone-200">
                  <span className="flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-sky-600" />
                    نتائج الدورة الأخيرة للجلب العام
                  </span>
                  <span className="font-mono text-[11px] text-stone-500">
                    تم جلب {xCollectionReport.total_fetched} • جديد {xCollectionReport.new_stored} • مكرر مستبعد {xCollectionReport.duplicates_skipped}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center text-xs">
                  <div className="p-2 rounded bg-white dark:bg-slate-800/80 border border-stone-200 dark:border-slate-700">
                    <span className="text-[10px] text-stone-500 block">تحذيرات احتيالية</span>
                    <span className="font-bold text-sky-600 dark:text-sky-400">{xCollectionReport.classified_categories?.fraud_warning || 0}</span>
                  </div>
                  <div className="p-2 rounded bg-white dark:bg-slate-800/80 border border-stone-200 dark:border-slate-700">
                    <span className="text-[10px] text-stone-500 block">حملات توعوية</span>
                    <span className="font-bold text-emerald-600 dark:text-emerald-400">{xCollectionReport.classified_categories?.awareness_campaign || 0}</span>
                  </div>
                  <div className="p-2 rounded bg-white dark:bg-slate-800/80 border border-stone-200 dark:border-slate-700">
                    <span className="text-[10px] text-stone-500 block">بلاغات متداولة</span>
                    <span className="font-bold text-amber-600 dark:text-amber-400">{xCollectionReport.classified_categories?.reported_incident || 0}</span>
                  </div>
                  <div className="p-2 rounded bg-white dark:bg-slate-800/80 border border-stone-200 dark:border-slate-700">
                    <span className="text-[10px] text-stone-500 block">حوادث موثقة</span>
                    <span className="font-bold text-red-600 dark:text-red-400">{xCollectionReport.classified_categories?.confirmed_incident || 0}</span>
                  </div>
                  <div className="p-2 rounded bg-white dark:bg-slate-800/80 border border-stone-200 dark:border-slate-700">
                    <span className="text-[10px] text-stone-500 block">غير مرتبط</span>
                    <span className="font-bold text-stone-500">{xCollectionReport.classified_categories?.unrelated || 0}</span>
                  </div>
                </div>

                <p className="text-[11px] text-stone-500 dark:text-slate-400 font-mono">
                  * ضابط الأمان: لا يتم تحويل منشورات التواصل إلى حوادث جنائية مؤكدة إلا في حال تضمنت قرائن قبض أو إدانة قضائية صريحة.
                </p>
              </div>
            )}
          </div>

          {/* Card 2: Single Post Link Analyzer */}
          <div className="panel p-5 space-y-4">
            <div className="space-y-1">
              <h3 className="text-sm font-bold text-stone-900 dark:text-stone-100 flex items-center gap-2">
                <Link2 className="w-4 h-4 text-sky-600" />
                <span>تحليل وتصنيف رابط منشور محدد من X</span>
              </h3>
              <p className="text-xs text-stone-600 dark:text-slate-400">
                يمكنك إدخال رابط منشور عام لفحصه، استخراج نصه، وتصنيفه تلقائياً دون تجاوز حدود الأمان
              </p>
            </div>

            <div className="flex flex-col sm:flex-row items-center gap-2">
              <input
                type="url"
                value={singlePostUrl}
                onChange={(e) => setSinglePostUrl(e.target.value)}
                placeholder="https://x.com/K_H_AlMuzaini/status/1558848332893753350"
                className="w-full px-3 py-2 text-xs rounded-lg border border-stone-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-stone-900 dark:text-stone-100 focus:outline-none focus:ring-1 focus:ring-sky-500 font-mono"
                dir="ltr"
              />
              <button
                onClick={handleAnalyzeSinglePost}
                disabled={isAnalyzingPost || !singlePostUrl.trim()}
                className={`px-4 py-2 rounded-lg text-xs font-semibold whitespace-nowrap flex items-center gap-1.5 transition-all ${
                  isAnalyzingPost || !singlePostUrl.trim()
                    ? 'bg-stone-200 dark:bg-slate-800 text-stone-400 cursor-not-allowed'
                    : 'bg-sky-600 hover:bg-sky-700 text-white'
                }`}
              >
                <Search className={`w-3.5 h-3.5 ${isAnalyzingPost ? 'animate-spin' : ''}`} />
                <span>{isAnalyzingPost ? 'جارٍ التحليل…' : 'تحليل المنشور'}</span>
              </button>
            </div>

            {analysisError && (
              <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800 text-xs text-red-700 dark:text-red-300">
                {analysisError}
              </div>
            )}

            {analyzedPost && (
              <div className="p-4 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200 dark:border-slate-800 space-y-3 text-xs">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-stone-200 dark:border-slate-800 pb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-stone-900 dark:text-stone-100">{analyzedPost.author_name}</span>
                    <span className="font-mono text-stone-500 text-[11px]" dir="ltr">{analyzedPost.author_handle}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2.5 py-0.5 rounded text-[11px] font-semibold ${
                      analyzedPost.category_key === 'confirmed_incident'
                        ? 'bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300'
                        : analyzedPost.category_key === 'fraud_warning'
                        ? 'bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300'
                        : analyzedPost.category_key === 'awareness_campaign'
                        ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
                        : analyzedPost.category_key === 'reported_incident'
                        ? 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300'
                        : 'bg-stone-200 text-stone-700 dark:bg-slate-800 dark:text-slate-300'
                    }`}>
                      {analyzedPost.category_label}
                    </span>
                    <a
                      href={analyzedPost.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-stone-400 hover:text-sky-600 p-1"
                      title="فتح المنشور في X"
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </div>
                </div>

                <p className="text-stone-800 dark:text-stone-200 leading-relaxed font-sans text-xs">
                  {analyzedPost.text}
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] pt-1">
                  <div className="p-2 rounded bg-white dark:bg-slate-800 border border-stone-200 dark:border-slate-700">
                    <span className="text-stone-500 block">معرّف التغريدة (Tweet ID):</span>
                    <span className="font-mono font-bold text-stone-900 dark:text-stone-100">{analyzedPost.post_id}</span>
                  </div>

                  <div className="p-2 rounded bg-white dark:bg-slate-800 border border-stone-200 dark:border-slate-700">
                    <span className="text-stone-500 block">مؤشرات الاستدلال:</span>
                    <span className="text-stone-800 dark:text-stone-200 font-mono truncate block">
                      {analyzedPost.classification_reasons.join(', ') || '—'}
                    </span>
                  </div>
                </div>

                {analyzedPost.outbound_urls && analyzedPost.outbound_urls.length > 0 && (
                  <div className="space-y-1">
                    <span className="text-[11px] text-stone-500 block">الروابط المستخرجة من المنشور:</span>
                    <div className="flex flex-wrap gap-1">
                      {analyzedPost.outbound_urls.map((u, i) => (
                        <a
                          key={i}
                          href={u}
                          target="_blank"
                          rel="noreferrer"
                          className="text-[11px] font-mono text-sky-600 hover:underline inline-flex items-center gap-1"
                        >
                          <ExternalLink className="w-3 h-3" />
                          <span>{u}</span>
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Card 3: Compliance & Integration Advisory */}
          <div className="panel p-5 bg-gradient-to-r from-stone-50 to-sky-50/20 dark:from-slate-900/60 dark:to-sky-950/20 space-y-2 border border-stone-200 dark:border-slate-800">
            <h4 className="text-xs font-bold text-stone-900 dark:text-stone-100 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 text-sky-600" />
              ضوابط الامتثال والسياسات المعتمدة في MetaView 3
            </h4>
            <p className="text-[11px] text-stone-600 dark:text-slate-400 leading-relaxed">
              تعتمد المنظومة بروتوكول جلب متوافق كلياً مع شروط وسياسات منصة X عبر القنوات العامة الرسمية (Syndication &amp; oEmbed) دون استخدام ملفات ارتباط (Cookies) منسوخة أو تجاوز جدران الحماية أو تعطيل شهادات SSL. في حال تم بلوغ حد الطلبات (429 Rate Limit)، يتحول المصدر تلقائياً إلى حالة <span className="font-semibold text-orange-600 dark:text-orange-400">Access Restricted</span> ويوصى حينها بتعيين متغير البيئة <code className="px-1 py-0.5 bg-stone-200 dark:bg-slate-800 rounded font-mono">X_BEARER_TOKEN</code> للوصول الرسمي عالي السعة.
            </p>
          </div>
        </div>
      )}

      {/* TAB 3: System Health */}
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

      {/* TAB 4: Ingestion Operations */}
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
