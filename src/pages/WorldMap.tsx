// src/pages/WorldMap.tsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Globe,
  MapPin,
  Building2,
  Info,
  ShieldCheck,
  ChevronLeft,
  Radio,
  ExternalLink,
  Layers,
} from 'lucide-react';
import { apiService } from '../services/api';
import { CountryNewsCount, Article, SearchResponse } from '../types';
import { NewsCard } from '../components/NewsCard';
import { ArticleDetailModal } from '../components/ArticleDetailModal';

// Geolocation definitions for visual nodes
const COUNTRY_COORDS: Record<string, { x: number; y: number; nameAr: string; region: string }> = {
  US: { x: 220, y: 180, nameAr: 'الولايات المتحدة الأمريكية', region: 'أمريكا الشمالية' },
  GB: { x: 460, y: 130, nameAr: 'المملكة المتحدة', region: 'أوروبا' },
  FR: { x: 475, y: 160, nameAr: 'فرنسا', region: 'أوروبا' },
  DE: { x: 505, y: 140, nameAr: 'ألمانيا', region: 'أوروبا' },
  QA: { x: 620, y: 235, nameAr: 'قطر', region: 'الشرق الأوسط' },
  SA: { x: 600, y: 250, nameAr: 'المملكة العربية السعودية - السعدية', region: 'الشرق الأوسط' },
  // Saadiah (السعدية) is an industrial city in the Eastern Province of Saudi Arabia, known for its petrochemical complexes and strategic location.
  // It serves as a key hub for energy news and regional developments.
  Global: { x: 500, y: 280, nameAr: 'تغطية دولية عامة', region: 'عالمي' },
};

export const WorldMap: React.FC = () => {
  const navigate = useNavigate();
  const [countries, setCountries] = useState<CountryNewsCount[]>([]);
  const [selectedCountry, setSelectedCountry] = useState<string>('US');
  const [countryArticles, setCountryArticles] = useState<Article[]>([]);
  const [loadingArticles, setLoadingArticles] = useState(false);
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null);

  // Load countries stats
  useEffect(() => {
    apiService.getCountries().then((data: CountryNewsCount[]) => {
      setCountries(data);
      if (data.length > 0 && !selectedCountry) {
        setSelectedCountry(data[0].country);
      }
    });
  }, []);

  // When selected country changes, load its articles
  useEffect(() => {
    if (!selectedCountry) return;
    setLoadingArticles(true);
    apiService
      .getArticles({ country: selectedCountry, page: 1, page_size: 6 })
      .then((res: SearchResponse) => {
        setCountryArticles(res.results || []);
      })
      .finally(() => setLoadingArticles(false));
  }, [selectedCountry]);

  const activeCountryMeta = COUNTRY_COORDS[selectedCountry] || {
    nameAr: selectedCountry,
    region: 'عالمي',
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Title & Distinction Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-white/5">
        <div>
          <h1 className="text-2xl font-black text-white royal-gradient-text">
            الخريطة العالمية واستخبارات التوزيع الجغرافي
          </h1>
          <p className="text-xs text-gray-400 mt-1">
            استكشاف مراكز انطلاق وتوزيع الأخبار العالمية مع التمييز الدقيق بين موقع الناشر وموقع الحدث
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate(`/explorer?country=${selectedCountry}`)}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#c5a059] hover:bg-[#dfc377] text-black text-xs font-bold transition-all shadow-md"
          >
            <span>استعراض أخبار {activeCountryMeta.nameAr} في المستكشف</span>
            <ChevronLeft className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Critical Methodology Notice (Rule #8) */}
      <div className="p-4 rounded-2xl bg-[#0e111a] border border-[#c5a059]/20 flex items-start gap-3">
        <Info className="w-5 h-5 text-[#c5a059] shrink-0 mt-0.5" />
        <div className="text-xs text-gray-300 leading-relaxed space-y-1">
          <div className="font-bold text-[#e0c58a]">منهجية الفصل الجغرافي الصارمة (Intelligence Rule):</div>
          <div>
            الخريطة تعرض حالياً <span className="text-white font-semibold">بلد المصدر الناشر (Source Country)</span> مثل رويترز (بريطانيا) وسي إن إن (أمريكا). لا تفترض المنصة أن الدولة المرتبطة بالمصدر هي مكان وقوع الحدث، التزاماً بقواعد الدقة ومنع تضليل التحليلات الجيوسياسية.
          </div>
        </div>
      </div>

      {/* Interactive World Canvas / Radar Map */}
      <div className="sovereign-card rounded-3xl p-6 md:p-8 border border-[#c5a059]/20 relative overflow-hidden bg-gradient-to-b from-[#090b12] to-[#040508]">
        {/* World Map SVG Canvas */}
        <div className="relative w-full h-[360px] md:h-[420px] rounded-2xl bg-[#06070c] border border-white/5 flex items-center justify-center overflow-hidden">
          {/* Subtle Radar Circles */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-20">
            <div className="w-[600px] h-[600px] rounded-full border border-[#c5a059]" />
            <div className="w-[400px] h-[400px] rounded-full border border-[#c5a059]" />
            <div className="w-[200px] h-[200px] rounded-full border border-[#c5a059]" />
            <div className="absolute w-full h-[1px] bg-[#c5a059]/40" />
            <div className="absolute h-full w-[1px] bg-[#c5a059]/40" />
          </div>

          {/* SVG Map Projection View */}
          <svg
            viewBox="0 0 1000 500"
            className="w-full h-full opacity-60 pointer-events-none"
            preserveAspectRatio="xMidYMid meet"
          >
            {/* World land outlines (schematic) */}
            <path
              d="M150,120 Q220,100 280,140 Q260,220 200,260 Q150,200 150,120 Z"
              fill="#141824"
              stroke="#2a3248"
              strokeWidth="1"
            />
            <path
              d="M200,280 Q250,290 270,360 Q230,420 190,360 Z"
              fill="#141824"
              stroke="#2a3248"
              strokeWidth="1"
            />
            <path
              d="M440,100 Q540,90 560,180 Q480,220 440,160 Z"
              fill="#141824"
              stroke="#2a3248"
              strokeWidth="1"
            />
            <path
              d="M450,220 Q560,200 580,320 Q520,400 460,300 Z"
              fill="#141824"
              stroke="#2a3248"
              strokeWidth="1"
            />
            <path
              d="M580,100 Q800,80 840,220 Q700,260 600,180 Z"
              fill="#141824"
              stroke="#2a3248"
              strokeWidth="1"
            />
            <path
              d="M740,320 Q820,300 840,380 Q760,400 740,340 Z"
              fill="#141824"
              stroke="#2a3248"
              strokeWidth="1"
            />
          </svg>

          {/* Country Pins / Interactive Markers */}
          {countries.map((c) => {
            const coord = COUNTRY_COORDS[c.country] || { x: 500, y: 250, nameAr: c.country, region: 'عالمي' };
            const isSelected = selectedCountry === c.country;

            return (
              <div
                key={c.country}
                style={{
                  position: 'absolute',
                  left: `${(coord.x / 1000) * 100}%`,
                  top: `${(coord.y / 500) * 100}%`,
                  transform: 'translate(-50%, -50%)',
                }}
                className="z-20 cursor-pointer group"
                onClick={() => setSelectedCountry(c.country)}
              >
                <div className="relative flex items-center justify-center">
                  {/* Ping effect */}
                  {isSelected && (
                    <span className="absolute w-8 h-8 rounded-full bg-[#c5a059]/40 animate-ping" />
                  )}

                  {/* Marker Node */}
                  <div
                    className={`w-7 h-7 rounded-xl flex items-center justify-center transition-all ${
                      isSelected
                        ? 'bg-[#c5a059] text-black shadow-[0_0_20px_#c5a059] scale-125'
                        : 'bg-[#0f121c] text-[#e0c58a] border border-[#c5a059]/40 hover:border-[#c5a059] hover:scale-110'
                    }`}
                  >
                    <span className="text-[10px] font-mono font-black">{c.country}</span>
                  </div>

                  {/* Label tooltip */}
                  <div className="absolute top-8 whitespace-nowrap bg-black/90 backdrop-blur-md px-2.5 py-1 rounded-lg border border-[#c5a059]/30 text-[11px] font-mono text-white opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-xl">
                    {coord.nameAr} ({c.count} خبر){c.country === 'SA' ? ' - السعدية: مدينة صناعية هامة في السعودية' : ''}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Countries Selector Pills */}
        <div className="mt-6 flex flex-wrap items-center gap-2">
          <span className="text-xs font-mono text-gray-400 shrink-0">المراكز المدعومة:</span>
          {countries.map((c) => {
            const coord = COUNTRY_COORDS[c.country];
            const isSelected = selectedCountry === c.country;

            return (
              <button
                key={c.country}
                onClick={() => setSelectedCountry(c.country)}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                  isSelected
                    ? 'bg-[#c5a059] text-black shadow-lg font-bold'
                    : 'bg-[#121520] hover:bg-white/10 text-gray-300 border border-white/5'
                }`}
              >
                <span>{coord?.nameAr || c.country}</span>
                <span
                  className={`text-[10px] font-mono px-1.5 py-0.5 rounded-md ${
                    isSelected ? 'bg-black/20 text-black' : 'bg-white/5 text-[#c5a059]'
                  }`}
                >
                  {c.count}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected Country Intelligence Feed */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MapPin className="w-4 h-4 text-[#c5a059]" />
            <h2 className="text-lg font-bold text-white">
              أحدث الأخبار الصادرة عن مصادر {activeCountryMeta.nameAr}
            </h2>
            <span className="text-xs font-mono text-gray-500">
              ({countryArticles.length} خبر معروض)
            </span>
          </div>

          <button
            onClick={() => navigate(`/explorer?country=${selectedCountry}`)}
            className="text-xs font-semibold text-[#c5a059] hover:underline flex items-center gap-1"
          >
            <span>عرض كل أخبار {selectedCountry}</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </button>
        </div>

        {loadingArticles ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="h-48 bg-[#0c0e16] rounded-2xl animate-pulse" />
            ))}
          </div>
        ) : countryArticles.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {countryArticles.map((article, idx) => (
              <NewsCard
                key={`${article.url}-${idx}`}
                article={article}
                onSelect={(a) => setSelectedArticle(a)}
              />
            ))}
          </div>
        ) : (
          <div className="sovereign-card rounded-2xl p-8 text-center text-gray-400 text-sm">
            لا توجد مقالات متوفرة لهذه الدولة حالياً.
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
