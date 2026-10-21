// src/pages/Sources.tsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Radio,
  Building2,
  Globe,
  Clock,
  Layers,
  ArrowRight,
  TrendingUp,
  ShieldAlert,
  ChevronLeft,
  ExternalLink,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from 'recharts';
import { apiService } from '../services/api';
import { SourcesResponse, SourceInfo, Article, SearchResponse } from '../types';
import { formatDate, timeAgo } from '../utils/formatDate';

const GOLD_PALETTE = ['#c5a059', '#e0c58a', '#8e6d2f', '#a8813a', '#dfc377', '#f5efd5', '#6f4e29'];

export const Sources: React.FC = () => {
  const navigate = useNavigate();
  const [sourcesData, setSourcesData] = useState<SourcesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedSourceCode, setSelectedSourceCode] = useState<string>('bbc');
  const [sampleArticles, setSampleArticles] = useState<Article[]>([]);

  useEffect(() => {
    setLoading(true);
    apiService
      .getSources()
      .then((data: SourcesResponse) => {
        setSourcesData(data);
        if (data.sources.length > 0) {
          setSelectedSourceCode(data.sources[0].code);
        }
      })
      .finally(() => setLoading(false));
  }, []);

  // Fetch recent sample articles for selected source
  useEffect(() => {
    if (!selectedSourceCode) return;
    apiService
      .getArticles({ source: selectedSourceCode, page: 1, page_size: 4 })
      .then((res: SearchResponse) => {
        setSampleArticles(res.results || []);
      });
  }, [selectedSourceCode]);

  const chartData = (sourcesData?.sources || []).map((s) => ({
    name: s.name,
    code: s.code,
    articles: s.articles_count,
  }));

  const activeSourceInfo = sourcesData?.sources.find((s) => s.code === selectedSourceCode);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-white/5">
        <div>
          <h1 className="text-2xl font-black text-white royal-gradient-text">
            استخبارات المصادر الإخبارية (Source Intelligence)
          </h1>
          <p className="text-xs text-gray-400 mt-1">
            تحليل شبكات وكالات الأنباء والصحف العالمية، وتتبع وتيرة النشر وحصة التغطية
          </p>
        </div>
      </div>

      {/* Methodological Caveat Box */}
      <div className="p-4 rounded-2xl bg-[#0d0f18] border border-[#c5a059]/20 flex items-start gap-3">
        <ShieldAlert className="w-5 h-5 text-[#c5a059] shrink-0 mt-0.5" />
        <div className="text-xs text-gray-300 leading-relaxed">
          <strong className="text-[#e0c58a] block mb-0.5">ضابط الموثوقية (Integrity Principle):</strong>
          إن عدد المقالات المنشورة أو حجم التغطية لا يمثل بأي شكل دليلاً منفرداً على مصداقية أو موضوعية المصدر الإخباري، وإنما يعكس فقط وتيرة النشر الرقمي لخدمات الـ RSS الخاصة به.
        </div>
      </div>

      {/* Coverage Volume Comparison Bar Chart */}
      <div className="sovereign-card rounded-3xl p-6 md:p-8 border border-[#c5a059]/20">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-[#c5a059]/10 text-[#c5a059]">
              <TrendingUp className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">مقارنة حجم التغطية بين المصادر</h3>
              <span className="text-[11px] text-gray-500 font-mono">
                ARTICLE VOLUME COMPARISON
              </span>
            </div>
          </div>
          <span className="text-xs font-mono text-gray-400">عدد المقالات المخزنة</span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 25 }}>
              <XAxis
                dataKey="code"
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                interval={0}
                angle={-25}
                textAnchor="end"
              />
              <YAxis stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0c0e16',
                  borderColor: 'rgba(197, 160, 89, 0.3)',
                  borderRadius: '12px',
                  color: '#fff',
                  fontSize: '12px',
                  fontFamily: 'Cairo',
                }}
                formatter={(val: any) => [`${val} مقال`, 'حجم التغطية']}
              />
              <Bar dataKey="articles" radius={[6, 6, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.code === selectedSourceCode ? '#e0c58a' : '#c5a059'}
                    cursor="pointer"
                    onClick={() => setSelectedSourceCode(entry.code)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sources Grid & Deep Dive Focus */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Source Cards List (2 Cols) */}
        <div className="lg:col-span-2 space-y-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Building2 className="w-4 h-4 text-[#c5a059]" />
            <span>قائمة وكالات الأنباء والصحف المعتمدة</span>
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {(sourcesData?.sources || []).map((src) => {
              const isSelected = selectedSourceCode === src.code;
              return (
                <div
                  key={src.code}
                  onClick={() => setSelectedSourceCode(src.code)}
                  className={`sovereign-card rounded-2xl p-5 border cursor-pointer transition-all ${
                    isSelected
                      ? 'border-[#c5a059] bg-[#121522] shadow-[0_0_20px_rgba(197,160,89,0.15)]'
                      : 'border-white/5 hover:border-[#c5a059]/30'
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-bold text-white text-base">{src.name}</span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-white/5 text-[#e0c58a] border border-white/10 uppercase">
                      {src.code}
                    </span>
                  </div>

                  <div className="space-y-1.5 text-xs text-gray-400">
                    <div className="flex items-center justify-between">
                      <span>بلد المنشأ:</span>
                      <strong className="text-white font-mono">{src.country}</strong>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>إجمالي المقالات:</span>
                      <strong className="text-[#e0c58a] font-mono">{src.articles_count}</strong>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>آخر مقال منشور:</span>
                      <span className="font-mono text-[11px] text-gray-500">
                        {src.latest_published ? timeAgo(src.latest_published) : 'غير محدد'}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Selected Source Deep-Dive Panel (1 Col) */}
        <div className="sovereign-card rounded-3xl p-6 border border-[#c5a059]/20 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-4 border-b border-white/5 mb-4">
              <div>
                <h3 className="text-lg font-black text-white">{activeSourceInfo?.name}</h3>
                <span className="text-[11px] font-mono text-[#c5a059]">
                  {activeSourceInfo?.code.toUpperCase()} INTELLIGENCE STREAM
                </span>
              </div>
              <button
                onClick={() => navigate(`/explorer?source=${activeSourceInfo?.code}`)}
                className="p-2 rounded-xl bg-white/5 hover:bg-[#c5a059]/20 text-[#e0c58a] transition-colors"
                title="فتح كافة أخبار هذا المصدر في المستكشف"
              >
                <ExternalLink className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="p-3 rounded-xl bg-[#0a0c14] border border-white/5 text-xs space-y-1">
                <span className="text-gray-500 block font-mono">رابط تغذية الـ RSS المباشر:</span>
                <span className="font-mono text-[11px] text-gray-300 break-all block">
                  {activeSourceInfo?.feed_url}
                </span>
              </div>

              <div>
                <span className="text-xs font-bold text-gray-300 block mb-2">
                  عينة من أحدث مقالات {activeSourceInfo?.name}:
                </span>
                <div className="space-y-2">
                  {sampleArticles.map((art, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-xl bg-[#0d0f18] border border-white/5 hover:border-[#c5a059]/30 transition-colors"
                    >
                      <a
                        href={art.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs font-semibold text-white hover:text-[#e0c58a] line-clamp-2 leading-snug"
                      >
                        {art.headline}
                      </a>
                      <span className="text-[10px] text-gray-500 font-mono block mt-1">
                        {timeAgo(art.published_at)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="pt-6 border-t border-white/5 mt-6">
            <button
              onClick={() => navigate(`/explorer?source=${activeSourceInfo?.code}`)}
              className="w-full py-3 rounded-xl bg-[#c5a059] hover:bg-[#dfc377] text-black font-bold text-xs flex items-center justify-center gap-2 transition-all shadow-lg"
            >
              <span>استكشاف كافة أخبار {activeSourceInfo?.name}</span>
              <ChevronLeft className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
