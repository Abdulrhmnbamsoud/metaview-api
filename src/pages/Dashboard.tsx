// src/pages/Dashboard.tsx
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Database,
  Radio,
  Clock,
  Calendar,
  TrendingUp,
  PieChart as PieIcon,
  ChevronLeft,
  RefreshCw,
  Compass,
  Sparkles,
  Layers,
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { apiService } from '../services/api';
import { Article, StatsSummary, SearchResponse } from '../types';
import { NewsCard } from '../components/NewsCard';
import { ArticleDetailModal } from '../components/ArticleDetailModal';
import { NewsGridSkeleton } from '../components/LoadingSkeleton';
import { ErrorBanner } from '../components/ErrorBanner';
import { formatDate, timeAgo } from '../utils/formatDate';

const GOLD_PALETTE = ['#c5a059', '#e0c58a', '#8e6d2f', '#a8813a', '#dfc377', '#f5efd5', '#6f4e29'];

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<StatsSummary | null>(null);
  const [latestArticles, setLatestArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null);

  const loadDashboardData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [statsData, articlesRes] = await Promise.all([
        apiService.getStats(),
        apiService.getArticles({ page: 1, page_size: 6, sort_by: 'published_at', sort_dir: 'desc' }),
      ]);
      setStats(statsData);
      setLatestArticles(articlesRes.results || []);
    } catch (err: any) {
      console.error('Dashboard load error:', err);
      setError(err.message || 'فشل تحميل بيانات لوحة القيادة');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDashboardData();

    // Listen to global header refresh event
    const handleRefreshEvent = () => loadDashboardData();
    window.addEventListener('metaview:refresh', handleRefreshEvent);
    return () => window.removeEventListener('metaview:refresh', handleRefreshEvent);
  }, [loadDashboardData]);

  // Compute timeline data dynamically from latest 50 articles
  const [timelineData, setTimelineData] = useState<{ time: string; count: number }[]>([]);
  useEffect(() => {
    apiService.getArticles({ page: 1, page_size: 50 }).then((res: SearchResponse) => {
      const counts: Record<string, number> = {};
      (res.results || []).forEach((a: Article) => {
        if (a.published_at) {
          const date = a.published_at.slice(0, 10);
          counts[date] = (counts[date] || 0) + 1;
        }
      });
      const points = Object.entries(counts)
        .map(([time, count]) => ({ time: time.slice(5), count }))
        .sort((a, b) => a.time.localeCompare(b.time));
      setTimelineData(points.length > 0 ? points : [{ time: 'اليوم', count: res.results?.length || 0 }]);
    }).catch(() => {});
  }, []);

  // Top keywords calculated from real headlines
  const trendingWords = useMemo(() => {
    const wordCounts: Record<string, number> = {};
    const stopWords = new Set(['the', 'and', 'to', 'of', 'in', 'a', 'is', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'that', 'this', 'it', 'an', 'be', 'are', 'was', 'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'أن', 'إن', 'قد', 'ما', 'لا', 'تم']);
    latestArticles.forEach((a) => {
      const words = (a.headline || '').split(/\s+/);
      words.forEach((w) => {
        const clean = w.replace(/[^\w\u0600-\u06FF]/g, '').toLowerCase();
        if (clean.length > 3 && !stopWords.has(clean)) {
          wordCounts[clean] = (wordCounts[clean] || 0) + 1;
        }
      });
    });
    return Object.entries(wordCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([word]) => word);
  }, [latestArticles]);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Welcome & Mission Banner */}
      <div className="sovereign-card rounded-3xl p-6 md:p-8 border border-[#c5a059]/20 relative overflow-hidden bg-gradient-to-l from-[#0e111a] via-[#090b12] to-[#06070a]">
        <div className="absolute top-0 left-0 w-96 h-96 bg-[#c5a059]/5 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-[#c5a059]/15 border border-[#c5a059]/30 text-[#e0c58a] text-xs font-mono font-semibold">
              <Sparkles className="w-3.5 h-3.5" />
              <span>GLOBAL INTELLIGENCE & MONITORING OPERATIONS</span>
            </div>
            <h1 className="text-2xl md:text-3xl font-black text-white royal-gradient-text tracking-tight">
              مركز الرصد والتحليل الإخباري العالمي
            </h1>
            <p className="text-sm text-gray-400 max-w-2xl leading-relaxed">
              منظومة رصد استخبارية متقدمة لتدفقات الأخبار العالمية، ترتكز على بيانات حقيقية مباشرة من خوادم MetaView، وتتيح كشف الأنماط، وتتبع المصادر، وتحليل الفروقات الجغرافية والزمنية.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/explorer')}
              className="flex items-center gap-2 px-5 py-3 rounded-2xl bg-[#c5a059] hover:bg-[#dfc377] text-black font-bold text-xs shadow-lg transition-all"
            >
              <Compass className="w-4 h-4" />
              <span>فتح مستكشف الأخبار</span>
            </button>
            <button
              onClick={loadDashboardData}
              disabled={loading}
              className="p-3 rounded-2xl bg-white/5 hover:bg-white/10 text-gray-300 border border-white/10 transition-all"
              title="تحديث البيانات الآن"
            >
              <RefreshCw className={`w-4 h-4 text-[#c5a059] ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      {error && <ErrorBanner message={error} onRetry={loadDashboardData} />}

      {/* Global Overview Cards (4 Key Metrics) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Metric 1: Total Articles */}
        <div className="sovereign-card rounded-2xl p-5 border border-[#c5a059]/15 flex items-center justify-between">
          <div>
            <span className="text-xs text-gray-400 font-mono block mb-1">إجمالي المقالات المتاحة</span>
            <span className="text-2xl font-black text-white font-mono">
              {stats?.total_articles.toLocaleString() || '—'}
            </span>
            <span className="text-[10px] text-emerald-400 block mt-1 font-mono">
              ● REAL DB REPOSITORY
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-[#c5a059]/10 border border-[#c5a059]/20 flex items-center justify-center text-[#c5a059]">
            <Database className="w-6 h-6" />
          </div>
        </div>

        {/* Metric 2: Active Sources */}
        <div className="sovereign-card rounded-2xl p-5 border border-[#c5a059]/15 flex items-center justify-between">
          <div>
            <span className="text-xs text-gray-400 font-mono block mb-1">المصادر الإخبارية النشطة</span>
            <span className="text-2xl font-black text-white font-mono">
              {stats?.total_sources || '—'}
            </span>
            <span className="text-[10px] text-gray-500 block mt-1 font-mono">
              REUTERS, BBC, CNN, DW...
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-[#c5a059]/10 border border-[#c5a059]/20 flex items-center justify-center text-[#c5a059]">
            <Radio className="w-6 h-6" />
          </div>
        </div>

        {/* Metric 3: Last 24 Hours */}
        <div className="sovereign-card rounded-2xl p-5 border border-[#c5a059]/15 flex items-center justify-between">
          <div>
            <span className="text-xs text-gray-400 font-mono block mb-1">أخبار آخر 24 ساعة</span>
            <span className="text-2xl font-black text-white font-mono">
              {stats?.articles_last_24h || '—'}
            </span>
            <span className="text-[10px] text-[#e0c58a] block mt-1 font-mono">
              LIVE CADENCE
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-[#c5a059]/10 border border-[#c5a059]/20 flex items-center justify-center text-[#c5a059]">
            <Clock className="w-6 h-6" />
          </div>
        </div>

        {/* Metric 4: Last Sync */}
        <div className="sovereign-card rounded-2xl p-5 border border-[#c5a059]/15 flex items-center justify-between">
          <div>
            <span className="text-xs text-gray-400 font-mono block mb-1">آخر توقيت لتحديث البيانات</span>
            <span className="text-sm font-bold text-white block mt-1">
              {stats?.last_sync ? timeAgo(stats.last_sync) : 'الآن'}
            </span>
            <span className="text-[10px] text-gray-500 block font-mono truncate max-w-[150px]">
              {stats?.last_sync ? formatDate(stats.last_sync) : 'متزامن'}
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-[#c5a059]/10 border border-[#c5a059]/20 flex items-center justify-center text-[#c5a059]">
            <Calendar className="w-6 h-6" />
          </div>
        </div>
      </div>

      {/* Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Timeline Chart (2 Cols) */}
        <div className="lg:col-span-2 sovereign-card rounded-3xl p-6 border border-[#c5a059]/15 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-[#c5a059]/10 text-[#c5a059]">
                <TrendingUp className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white">حركة نشر الأخبار عبر الزمن (Timeline)</h3>
                <span className="text-[11px] text-gray-500 font-mono">
                  NEWS ACTIVITY & INGESTION CADENCE
                </span>
              </div>
            </div>
            <span className="text-xs font-mono text-gray-400">توزيع يومي</span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timelineData}>
                <defs>
                  <linearGradient id="goldGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#c5a059" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#c5a059" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" stroke="#475569" fontSize={11} tickLine={false} />
                <YAxis stroke="#475569" fontSize={11} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0c0e16',
                    borderColor: 'rgba(197, 160, 89, 0.3)',
                    borderRadius: '12px',
                    color: '#fff',
                    fontSize: '12px',
                    fontFamily: 'Cairo',
                  }}
                  formatter={(val: any) => [`${val} خبر`, 'الأخبار']}
                />
                <Area
                  type="monotone"
                  dataKey="count"
                  stroke="#c5a059"
                  strokeWidth={2.5}
                  fillOpacity={1}
                  fill="url(#goldGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Source Distribution (1 Col) */}
        <div className="sovereign-card rounded-3xl p-6 border border-[#c5a059]/15 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-[#c5a059]/10 text-[#c5a059]">
                <PieIcon className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white">توزيع المصادر (Sources)</h3>
                <span className="text-[11px] text-gray-500 font-mono">COVERAGE SHARE</span>
              </div>
            </div>
          </div>

          <div className="h-48 w-full flex items-center justify-center">
            {stats?.source_distribution && stats.source_distribution.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={stats.source_distribution}
                    dataKey="count"
                    nameKey="source"
                    cx="50%"
                    cy="50%"
                    innerRadius={45}
                    outerRadius={75}
                    paddingAngle={3}
                  >
                    {stats.source_distribution.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={GOLD_PALETTE[index % GOLD_PALETTE.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0c0e16',
                      borderColor: 'rgba(197, 160, 89, 0.3)',
                      borderRadius: '12px',
                      color: '#fff',
                      fontSize: '12px',
                      fontFamily: 'Cairo',
                    }}
                    formatter={(val: any, name: any) => [`${val} خبر`, name]}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-xs text-gray-500 font-mono">جاري حساب التوزيع...</div>
            )}
          </div>

          {/* Source Legend Pills */}
          <div className="flex flex-wrap gap-1.5 pt-3 border-t border-white/5">
            {(stats?.source_distribution || []).slice(0, 5).map((s, idx) => (
              <span
                key={idx}
                className="px-2 py-0.5 rounded-md text-[10px] font-mono bg-white/5 border border-white/10 text-gray-300"
              >
                {s.source}: {s.count}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Trending Topics Bar */}
      {trendingWords.length > 0 && (
        <div className="sovereign-card rounded-2xl p-4 border border-[#c5a059]/15 flex items-center gap-3 overflow-x-auto">
          <span className="text-xs font-mono text-[#c5a059] shrink-0 font-bold flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5" />
            <span>الكلمات الأكثر تكراراً:</span>
          </span>
          <div className="flex items-center gap-2">
            {trendingWords.map((word, i) => (
              <button
                key={i}
                onClick={() => navigate(`/explorer?q=${encodeURIComponent(word)}`)}
                className="px-3 py-1 rounded-xl bg-white/5 hover:bg-[#c5a059]/20 text-gray-300 hover:text-[#e0c58a] text-xs font-medium border border-white/5 transition-all shrink-0"
              >
                #{word}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Latest Intelligence Feed */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[#c5a059] shadow-[0_0_10px_#c5a059]" />
            <h2 className="text-lg font-bold text-white">آخر المستجدات الإخبارية (Latest Feed)</h2>
          </div>
          <button
            onClick={() => navigate('/explorer')}
            className="flex items-center gap-1 text-xs font-semibold text-[#c5a059] hover:text-[#e0c58a] transition-colors"
          >
            <span>استعراض كافة الأخبار</span>
            <ChevronLeft className="w-4 h-4" />
          </button>
        </div>

        {loading ? (
          <NewsGridSkeleton count={6} />
        ) : latestArticles.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {latestArticles.map((article, idx) => (
              <NewsCard
                key={`${article.url}-${idx}`}
                article={article}
                onSelect={(a) => setSelectedArticle(a)}
              />
            ))}
          </div>
        ) : (
          <div className="sovereign-card rounded-2xl p-8 text-center text-gray-400 text-sm">
            لا توجد مقالات متوفرة حالياً. اضغط على زر "تحديث التدفق" في الأعلى لتشغيل عملية الرصد.
          </div>
        )}
      </div>

      {/* Article Detail Modal */}
      <ArticleDetailModal
        article={selectedArticle}
        onClose={() => setSelectedArticle(null)}
      />
    </div>
  );
};
