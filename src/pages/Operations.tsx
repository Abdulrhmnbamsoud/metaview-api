// src/pages/Operations.tsx
import React, { useState, useEffect } from 'react';
import {
  Activity,
  Play,
  CheckCircle,
  AlertCircle,
  Clock,
  Database,
  Radio,
  Server,
  Terminal,
  ShieldCheck,
  RefreshCw,
  HardDrive,
} from 'lucide-react';
import { apiService } from '../services/api';
import { HealthStatus, IngestRunResult, SourcesResponse } from '../types';
import { formatDate, timeAgo } from '../utils/formatDate';

export const Operations: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [sourcesData, setSourcesData] = useState<SourcesResponse | null>(null);
  const [loading, setLoading] = useState(true);

  // Manual Ingest State
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestLimit, setIngestLimit] = useState(30);
  const [allowWebScrape, setAllowWebScrape] = useState(false);
  const [lastRunResult, setLastRunResult] = useState<IngestRunResult | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setLoading(true);
    try {
      const [h, s] = await Promise.all([
        apiService.getHealth(),
        apiService.getSources(),
      ]);
      setHealth(h);
      setSourcesData(s);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const timer = setInterval(fetchStatus, 20000);
    return () => clearInterval(timer);
  }, []);

  const handleTriggerIngest = async () => {
    if (isIngesting) return;
    setIsIngesting(true);
    setIngestError(null);
    try {
      const result = await apiService.runIngest(ingestLimit, undefined, allowWebScrape);
      setLastRunResult(result);
      await fetchStatus();
    } catch (err: any) {
      setIngestError(err.message || 'فشل تنفيذ عملية الرصد والتحديث');
    } finally {
      setIsIngesting(false);
    }
  };

  const isOnline = health?.status !== 'offline';

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-white/5">
        <div>
          <h1 className="text-2xl font-black text-white royal-gradient-text">
            مركز العمليات ومراقبة تدفق البيانات (Operations Center)
          </h1>
          <p className="text-xs text-gray-400 mt-1">
            متابعة حالة الخوادم، تدفق التغذيات الإخبارية، وإدارة دورات الرصد والتحديث عبر الـ API
          </p>
        </div>

        <button
          onClick={fetchStatus}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-gray-300 text-xs font-semibold border border-white/10 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-[#c5a059] ${loading ? 'animate-spin' : ''}`} />
          <span>تحديث المقاييس</span>
        </button>
      </div>

      {/* System Telemetry Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="sovereign-card rounded-2xl p-5 border border-white/10">
          <span className="text-xs text-gray-400 font-mono block mb-1">حالة الـ API الأساسي</span>
          <div className="flex items-center gap-2 mt-1">
            <span
              className={`w-3 h-3 rounded-full ${
                isOnline ? 'bg-emerald-400 shadow-[0_0_10px_#34d399]' : 'bg-red-500 shadow-[0_0_10px_#ef4444]'
              }`}
            />
            <span className="text-lg font-bold font-mono text-white">
              {isOnline ? 'ONLINE (OPERATIONAL)' : 'DISCONNECTED'}
            </span>
          </div>
          <span className="text-[10px] text-gray-500 font-mono block mt-1">
            URL: {apiService.getBaseUrl()}
          </span>
        </div>

        <div className="sovereign-card rounded-2xl p-5 border border-white/10">
          <span className="text-xs text-gray-400 font-mono block mb-1">المقالات المسجلة بالخادم</span>
          <span className="text-2xl font-black text-white font-mono">
            {health?.rows.toLocaleString() || '—'}
          </span>
          <span className="text-[10px] text-emerald-400 block mt-1 font-mono">
            SQLITE DATABASE CORE
          </span>
        </div>

        <div className="sovereign-card rounded-2xl p-5 border border-white/10">
          <span className="text-xs text-gray-400 font-mono block mb-1">المصادر المربوطة</span>
          <span className="text-2xl font-black text-white font-mono">
            {sourcesData?.count || '—'}
          </span>
          <span className="text-[10px] text-[#e0c58a] block mt-1 font-mono">
            TRUSTED RSS CHANNELS
          </span>
        </div>

        <div className="sovereign-card rounded-2xl p-5 border border-white/10">
          <span className="text-xs text-gray-400 font-mono block mb-1">توقيت نبض الخادم (Heartbeat)</span>
          <span className="text-sm font-bold text-white block mt-1">
            {health?.time ? timeAgo(health.time) : '—'}
          </span>
          <span className="text-[10px] text-gray-500 font-mono block truncate">
            {health?.time || 'Pending...'}
          </span>
        </div>
      </div>

      {/* Manual Execution Control Room */}
      <div className="sovereign-card rounded-3xl p-6 md:p-8 border border-[#c5a059]/20 bg-gradient-to-r from-[#0c0e18] via-[#090b12] to-[#06070a]">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-6 border-b border-white/5">
          <div className="space-y-1">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-[#c5a059]/15 text-[#e0c58a] text-xs font-mono">
              <Terminal className="w-3.5 h-3.5" />
              <span>POST /ingest/run CONTROL TRIGGER</span>
            </div>
            <h2 className="text-xl font-bold text-white">
              تشغيل دورة الرصد وجمع الأخبار يدوياً
            </h2>
            <p className="text-xs text-gray-400 leading-relaxed max-w-2xl">
              إرسال أمر فوري إلى خادم FastAPI لاستدعاء محرك التغذية RSS لكافة القنوات المعتمدة، استخراج العناوين والصور عبر وحدة Image Scraper، وتحديث المستودع مع منع التكرار.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2 bg-[#121520] px-3 py-2 rounded-xl border border-white/10 text-xs">
              <span className="text-gray-400">الحد لكل مصدر:</span>
              <select
                value={ingestLimit}
                onChange={(e) => setIngestLimit(Number(e.target.value))}
                disabled={isIngesting}
                className="bg-transparent text-[#e0c58a] font-bold font-mono outline-none"
              >
                <option value={15}>15 مقال</option>
                <option value={30}>30 مقال</option>
                <option value={50}>50 مقال</option>
              </select>
            </div>

            <button
              onClick={handleTriggerIngest}
              disabled={isIngesting}
              className={`flex items-center gap-2 px-6 py-3 rounded-2xl text-xs font-bold shadow-xl transition-all ${
                isIngesting
                  ? 'bg-[#c5a059]/30 text-gray-400 border border-[#c5a059]/30 cursor-not-allowed'
                  : 'bg-gradient-to-r from-[#c5a059] to-[#dfc377] hover:brightness-110 text-black'
              }`}
            >
              <Play className={`w-4 h-4 fill-current ${isIngesting ? 'animate-spin' : ''}`} />
              <span>{isIngesting ? 'جاري تنفيذ دورة الرصد...' : 'بدء دورة الرصد الآن'}</span>
            </button>
          </div>
        </div>

        {/* Ingest Error Notice */}
        {ingestError && (
          <div className="mt-4 p-4 rounded-xl bg-red-950/40 border border-red-500/30 text-xs text-red-300 flex items-center gap-3">
            <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
            <span>{ingestError}</span>
          </div>
        )}

        {/* Live Execution Results Table */}
        {lastRunResult && (
          <div className="mt-6 space-y-4">
            <div className="flex items-center justify-between text-xs">
              <span className="font-mono text-[#e0c58a] font-bold">
                نتائج آخر دورة رصد مكتملة (Telemetry):
              </span>
              <span className="text-gray-400 font-mono text-[11px]">
                انتهت: {formatDate(lastRunResult.finished_at)}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div className="p-3 rounded-xl bg-[#121520] border border-white/5">
                <span className="text-[10px] text-gray-500 block font-mono">مضافة حديثاً</span>
                <span className="text-lg font-black text-emerald-400 font-mono">
                  +{lastRunResult.inserted_total}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-[#121520] border border-white/5">
                <span className="text-[10px] text-gray-500 block font-mono">استخراج صور ناجح</span>
                <span className="text-lg font-black text-[#e0c58a] font-mono">
                  {lastRunResult.with_images_total || 0}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-[#121520] border border-white/5">
                <span className="text-[10px] text-gray-500 block font-mono">مكررة (تخطي)</span>
                <span className="text-lg font-black text-gray-400 font-mono">
                  {lastRunResult.skipped_dup_total}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-[#121520] border border-white/5">
                <span className="text-[10px] text-gray-500 block font-mono">أخطاء القنوات</span>
                <span className="text-lg font-black text-red-400 font-mono">
                  {lastRunResult.errors_total}
                </span>
              </div>
            </div>

            {/* Channels breakdown */}
            <div className="max-h-60 overflow-y-auto border border-white/5 rounded-2xl bg-[#090b12]">
              <table className="w-full text-right text-xs">
                <thead className="bg-[#121522] text-gray-400 font-mono text-[11px] border-b border-white/5">
                  <tr>
                    <th className="p-3">القناة (Source)</th>
                    <th className="p-3">المسترجع</th>
                    <th className="p-3">المدرج</th>
                    <th className="p-3">الزمن (ث)</th>
                    <th className="p-3">الحالة</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 font-mono text-gray-300">
                  {lastRunResult.sources.map((s, idx) => (
                    <tr key={idx} className="hover:bg-white/5">
                      <td className="p-3 font-bold text-white">{s.source}</td>
                      <td className="p-3 text-gray-400">{s.fetched_entries}</td>
                      <td className="p-3 text-emerald-400 font-bold">+{s.inserted}</td>
                      <td className="p-3 text-gray-400">{s.time_sec}s</td>
                      <td className="p-3">
                        {s.error ? (
                          <span className="text-red-400 text-[10px] truncate block max-w-[150px]">
                            {s.error}
                          </span>
                        ) : (
                          <span className="text-emerald-400 flex items-center gap-1 text-[11px]">
                            <CheckCircle className="w-3 h-3" />
                            <span>سليم</span>
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Sources Health Breakdown */}
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-white">قائمة القنوات الإخبارية وجاهزية التدفق</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {(sourcesData?.sources || []).map((src) => (
            <div
              key={src.code}
              className="sovereign-card rounded-2xl p-4 border border-white/5 flex items-center justify-between"
            >
              <div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span className="font-bold text-white text-sm">{src.name}</span>
                </div>
                <span className="text-[11px] text-gray-500 font-mono block mt-1">
                  الدولة: {src.country} • {src.articles_count} مقال مخزن
                </span>
              </div>
              <span className="px-2.5 py-1 rounded-lg text-[10px] font-mono bg-white/5 text-[#e0c58a] border border-white/10 uppercase">
                {src.code}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
