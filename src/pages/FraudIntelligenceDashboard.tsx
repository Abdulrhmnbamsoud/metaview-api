// src/pages/FraudIntelligenceDashboard.tsx
// MetaView 3 — Fraud Intelligence Dashboard
// Pixel-Accurate Implementation Matching Approved Reference Design
import React, { useState, useEffect } from 'react';
import {
  Download,
  Calendar,
  MapPin,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  Info,
  ShieldAlert,
  AlertTriangle,
  FileText,
  ExternalLink,
  X,
  CreditCard,
  Briefcase,
  Link as LinkIcon,
  ShoppingCart,
  Coins,
  Landmark,
  Shield,
  Activity,
  Clock,
  CheckCircle2,
  MessageSquare,
  Plus,
  Minus,
} from 'lucide-react';
import { apiService } from '../services/api';
import {
  DashboardViewResponse,
  EnrichedIncident,
  DrilldownResponse,
} from '../types/fraudIntelligence';

export const FraudIntelligenceDashboard: React.FC = () => {
  // Filter States
  const [dateRange, setDateRange] = useState<number>(30);
  const [geoFilter, setGeoFilter] = useState<string>('السعودية');
  const [fraudTypeFilter, setFraudTypeFilter] = useState<string>('جميع الأنواع');
  const [mapMode, setMapMode] = useState<'regions' | 'cities'>('regions');

  // Active Flow Hover State (for Sankey highlighting)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // Data States
  const [data, setData] = useState<DashboardViewResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Drilldown Modal State
  const [drilldownOpen, setDrilldownOpen] = useState(false);
  const [drilldownTitle, setDrilldownTitle] = useState('');
  const [drilldownLoading, setDrilldownLoading] = useState(false);
  const [drilldownData, setDrilldownData] = useState<DrilldownResponse | null>(null);
  const [selectedIncident, setSelectedIncident] = useState<EnrichedIncident | null>(null);

  // Fetch Dashboard Data
  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await apiService.getFraudIntelligenceDashboardView({
          windowDays: dateRange,
          geo: geoFilter === 'السعودية' ? 'SA' : 'all',
          fraudType: fraudTypeFilter !== 'جميع الأنواع' ? fraudTypeFilter : undefined,
        });
        if (isMounted) setData(res);
      } catch (err: any) {
        console.error('Failed to load dashboard view:', err);
        if (isMounted) setError(err?.message || 'تعذر تحميل بيانات ميتا فيو');
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    load();
    return () => {
      isMounted = false;
    };
  }, [dateRange, geoFilter, fraudTypeFilter]);

  // Open Drilldown Modal
  const openDrilldown = async (title: string, params: Record<string, any> = {}) => {
    setDrilldownTitle(title);
    setDrilldownOpen(true);
    setDrilldownLoading(true);
    setSelectedIncident(null);
    try {
      const res = await apiService.getFraudIntelligenceIncidents({
        fraud_type: params.fraud_type,
        channel: params.channel,
        sector: params.sector,
        city: params.city,
      });
      setDrilldownData(res);
    } catch (err) {
      console.error('Failed to load drilldown:', err);
    } finally {
      setDrilldownLoading(false);
    }
  };

  // View specific incident
  const viewSingleIncident = async (incidentId: number) => {
    try {
      setDrilldownLoading(true);
      const inc = await apiService.getFraudIntelligenceSingleIncident(incidentId);
      setSelectedIncident(inc);
      setDrilldownTitle(`تفاصيل الحادثة: ${inc.title}`);
      setDrilldownOpen(true);
    } catch (err) {
      console.error('Failed to load incident detail:', err);
    } finally {
      setDrilldownLoading(false);
    }
  };

  // Export handler
  const handleExport = () => {
    if (!data) return;
    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `metaview-fraud-intelligence-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading && !data) {
    return (
      <div className="p-8 space-y-6 animate-pulse" dir="rtl">
        <div className="h-8 bg-slate-200 rounded-lg w-1/4"></div>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          <div className="lg:col-span-7 h-72 bg-slate-200 rounded-2xl"></div>
          <div className="lg:col-span-5 h-72 bg-slate-200 rounded-2xl"></div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="h-80 bg-slate-200 rounded-2xl"></div>
          <div className="h-80 bg-slate-200 rounded-2xl"></div>
          <div className="h-80 bg-slate-200 rounded-2xl"></div>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="p-8 flex flex-col items-center justify-center min-h-[400px] text-center" dir="rtl">
        <AlertTriangle className="w-12 h-12 text-red-500 mb-3" />
        <h3 className="text-lg font-bold text-[#10213D] mb-1">تعذر تحميل بيانات ميتا فيو</h3>
        <p className="text-sm text-[#64748B] max-w-md mb-4">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-blue-600 text-white rounded-xl text-xs font-semibold hover:bg-blue-700 transition"
        >
          إعادة المحاولة
        </button>
      </div>
    );
  }

  const hero = data?.hero_insight;
  const flow = data?.relationship_flow;
  const patterns = data?.top_patterns || [];
  const mapPoints = data?.map_points || [];
  const changed = data?.what_changed || [];
  const incidents = data?.latest_incidents || [];
  const banking = data?.banking_implications || [];
  const quality = data?.data_quality_stats;

  return (
    <div
      dir="ltr"
      className="min-h-screen bg-[#F7F9FC] text-[#10213D] p-5 md:p-8 space-y-6 font-sans antialiased"
      style={{ fontFamily: "'IBM Plex Sans Arabic', -apple-system, BlinkMacSystemFont, sans-serif" }}
    >
      {/* ------------------------------------------------------------------- */}
      {/* 1. Main Page Header (Left: Title | Right: Filters matching image)   */}
      {/* ------------------------------------------------------------------- */}
      <div dir="ltr" className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Title and Subtitle on the LEFT */}
        <div dir="rtl" className="text-right space-y-1">
          <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight text-[#10213D]">
            ميتا فيو
          </h1>
          <p className="text-xs md:text-sm text-[#64748B]">
            رصد وتحليل اتجاهات الاحتيال المالي وبناء صورة ميتا فيو من مصادر موثوقة
          </p>
        </div>

        {/* Filters & Export controls on the RIGHT (Left-to-Right order matching image) */}
        <div dir="ltr" className="flex flex-wrap items-center gap-2.5">
          {/* 1. Date Range Dropdown */}
          <div className="relative">
            <button
              onClick={() => setDateRange(dateRange === 30 ? 7 : dateRange === 7 ? 90 : 30)}
              className="flex items-center gap-2 px-3.5 py-2 bg-white border border-[#E7ECF3] rounded-xl text-xs font-semibold text-[#10213D] hover:bg-slate-50 transition shadow-xs"
            >
              <Calendar className="w-3.5 h-3.5 text-[#64748B]" />
              <span dir="rtl">آخر {dateRange} يوماً</span>
              <ChevronDown className="w-3.5 h-3.5 text-[#64748B]" />
            </button>
          </div>

          {/* 2. Geographic Dropdown */}
          <div className="relative">
            <button
              onClick={() => setGeoFilter(geoFilter === 'السعودية' ? 'العالم' : 'السعودية')}
              className="flex items-center gap-2 px-3.5 py-2 bg-white border border-[#E7ECF3] rounded-xl text-xs font-semibold text-[#10213D] hover:bg-slate-50 transition shadow-xs"
            >
              <MapPin className="w-3.5 h-3.5 text-[#64748B]" />
              <span>{geoFilter}</span>
              <ChevronDown className="w-3.5 h-3.5 text-[#64748B]" />
            </button>
          </div>

          {/* 3. Fraud Type Dropdown */}
          <div className="relative">
            <select
              value={fraudTypeFilter}
              onChange={(e) => setFraudTypeFilter(e.target.value)}
              className="appearance-none flex items-center gap-2 px-3.5 py-2 pr-8 pl-4 bg-white border border-[#E7ECF3] rounded-xl text-xs font-semibold text-[#10213D] hover:bg-slate-50 transition shadow-xs cursor-pointer outline-none"
            >
              <option value="جميع الأنواع">جميع الأنواع</option>
              <option value="gov_impersonation">انتحال جهات حكومية</option>
              <option value="bank_impersonation">انتحال بنوك ومصارف</option>
              <option value="investment_fraud">احتيال استثماري</option>
              <option value="phishing">تصيد روابط</option>
              <option value="ecommerce_fraud">احتيال تجاري</option>
            </select>
            <ChevronDown className="w-3.5 h-3.5 text-[#64748B] absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          </div>

          {/* 4. Export Button on the FAR RIGHT */}
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-4 py-2 bg-[#102544] hover:bg-[#1a3860] text-white rounded-xl text-xs font-semibold transition shadow-xs"
          >
            <Download className="w-3.5 h-3.5" />
            <span>تصدير</span>
          </button>
        </div>
      </div>

      {/* ------------------------------------------------------------------- */}
      {/* 2. Top Row: (Left: Hero Insight | Right: Geographic Map)            */}
      {/* ------------------------------------------------------------------- */}
      <div dir="ltr" className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        {/* Hero Intelligence Insight (Approx 7 of 12 cols = ~58%) */}
        {hero && (
          <div
            dir="rtl"
            className="lg:col-span-7 relative overflow-hidden bg-white border border-[#E7ECF3] rounded-2xl p-6 md:p-7 shadow-xs flex flex-col justify-between min-h-[310px]"
          >
            {/* Background Riyadh Skyline on the LEFT fading smoothly to pure white on the right */}
            <div
              className="absolute inset-y-0 left-0 w-[45%] bg-cover bg-center pointer-events-none opacity-85 hidden sm:block"
              style={{
                backgroundImage: 'url(/riyadh_skyline.jpg)',
                maskImage: 'linear-gradient(to right, rgba(0,0,0,1) 0%, rgba(0,0,0,0.5) 60%, rgba(0,0,0,0) 100%)',
                WebkitMaskImage: 'linear-gradient(to right, rgba(0,0,0,1) 0%, rgba(0,0,0,0.5) 60%, rgba(0,0,0,0) 100%)',
              }}
            />

            {/* Top Content: Badge, Headline, Summary */}
            <div className="relative z-10 space-y-3.5 max-w-xl text-right">
              {/* Badge */}
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#FEF2F2] border border-[#FEE2E2] text-[#EF4444] text-xs font-bold">
                <TrendingUp className="w-3.5 h-3.5" />
                <span>{hero.trend_badge}</span>
              </div>

              {/* Title & Description */}
              <div className="space-y-2">
                <h2 className="text-xl md:text-2xl font-extrabold text-[#10213D] leading-snug">
                  {hero.headline}
                </h2>
                <p className="text-xs md:text-sm text-[#64748B] leading-relaxed">
                  {hero.summary}
                </p>
              </div>
            </div>

            {/* Bottom Row: Exact matching layout from media_1792925929858.jpg */}
            <div dir="rtl" className="relative z-10 pt-4 flex flex-wrap items-center justify-between gap-4 border-t border-slate-100/80">
              {/* 3 Metrics on the RIGHT */}
              <div className="flex items-center gap-4 sm:gap-6">
                {/* Metric 1: Associated Incidents */}
                <div className="flex items-center gap-2">
                  <div className="p-1.5 rounded-lg bg-[#FEF2F2] text-[#EF4444]">
                    <AlertTriangle className="w-3.5 h-3.5" />
                  </div>
                  <div className="flex items-baseline gap-1">
                    <span className="text-sm font-bold text-[#10213D]">{hero.incidents_count}</span>
                    <span className="text-[11px] text-[#64748B]">حوادث مرتبطة</span>
                  </div>
                </div>

                {/* Metric 2: Sources */}
                <div className="flex items-center gap-2">
                  <div className="p-1.5 rounded-lg bg-[#EFF6FF] text-[#2563EB]">
                    <FileText className="w-3.5 h-3.5" />
                  </div>
                  <div className="flex items-baseline gap-1">
                    <span className="text-sm font-bold text-[#10213D]">{hero.sources_count}</span>
                    <span className="text-[11px] text-[#64748B]">مصادر مستقلة</span>
                  </div>
                </div>

                {/* Metric 3: % Growth */}
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-[#FEF2F2] text-[#EF4444] text-xs font-bold">
                    <TrendingUp className="w-3.5 h-3.5" />
                    <span>{hero.growth_pct}</span>
                  </div>
                  <span className="text-[11px] text-[#64748B]">مقارنة بالشهر السابق</span>
                </div>
              </div>

              {/* Action Button on LEFT */}
              <button
                onClick={() =>
                  openDrilldown(`الحوادث الداعمة للاتجاه: ${hero.headline}`, {
                    fraud_type: 'gov_impersonation',
                  })
                }
                className="flex items-center gap-2 px-4 py-2 bg-[#102544] hover:bg-[#1a3860] text-white rounded-xl text-xs font-bold transition shadow-xs"
              >
                <ChevronRight className="w-4 h-4" />
                <span>عرض الحوادث</span>
              </button>
            </div>
          </div>
        )}

        {/* Geographic Intelligence Map (Approx 5 of 12 cols = ~42%) */}
        <div
          dir="rtl"
          className="lg:col-span-5 bg-white border border-[#E7ECF3] rounded-2xl p-5 md:p-6 shadow-xs flex flex-col justify-between"
        >
          {/* Map Card Header: Controls on Left, Title on Right */}
          <div dir="ltr" className="flex items-center justify-between pb-2 border-b border-[#F1F5F9]">
            {/* Controls on Left */}
            <div dir="rtl" className="flex items-center gap-2">
              <div className="relative">
                <button
                  onClick={() => setGeoFilter(geoFilter === 'السعودية' ? 'العالم' : 'السعودية')}
                  className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-50 border border-[#E7ECF3] rounded-lg text-xs font-semibold text-[#10213D]"
                >
                  <span>{geoFilter}</span>
                  <ChevronDown className="w-3 h-3 text-[#64748B]" />
                </button>
              </div>

              {/* Mode toggles */}
              <div className="flex items-center bg-slate-100 p-0.5 rounded-lg text-xs">
                <button
                  onClick={() => setMapMode('regions')}
                  className={`px-2.5 py-0.5 rounded-md font-medium transition ${
                    mapMode === 'regions'
                      ? 'bg-[#2563EB] text-white shadow-xs'
                      : 'text-[#64748B] hover:text-[#10213D]'
                  }`}
                >
                  المناطق
                </button>
                <button
                  onClick={() => setMapMode('cities')}
                  className={`px-2.5 py-0.5 rounded-md font-medium transition ${
                    mapMode === 'cities'
                      ? 'bg-[#2563EB] text-white shadow-xs'
                      : 'text-[#64748B] hover:text-[#10213D]'
                  }`}
                >
                  المدن
                </button>
              </div>
            </div>

            {/* Title on Right */}
            <div dir="rtl" className="flex items-center gap-1.5 text-sm font-bold text-[#10213D]">
              <span>خريطة الحوادث الموثقة</span>
              <Info className="w-4 h-4 text-[#64748B]" />
            </div>
          </div>

          {/* Interactive Map Canvas */}
          <div className="relative h-56 md:h-64 my-2 flex items-center justify-center overflow-hidden">
            {/* Saudi Map Silhouette SVG */}
            <svg
              viewBox="0 0 500 400"
              className="w-full h-full text-slate-100 select-none drop-shadow-xs"
              style={{ maxHeight: '250px' }}
            >
              <defs>
                <linearGradient id="saudiGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#F8FAFC" />
                  <stop offset="100%" stopColor="#E2E8F0" />
                </linearGradient>
              </defs>
              {/* Generalized Saudi territory polygon */}
              <path
                d="M 90 80 
                   L 140 70 
                   L 200 65 
                   L 270 85 
                   L 340 100 
                   L 390 120 
                   L 440 135 
                   L 460 160 
                   L 420 200 
                   L 435 240 
                   L 400 270 
                   L 360 300 
                   L 310 325 
                   L 250 330 
                   L 210 300 
                   L 180 270 
                   L 140 230 
                   L 110 180 
                   L 80 130 
                   Z"
                fill="url(#saudiGrad)"
                stroke="#CBD5E1"
                strokeWidth="1.5"
                strokeLinejoin="round"
              />
              {/* Regional contours */}
              <path
                d="M 140 70 Q 180 140 250 160 T 360 300"
                fill="none"
                stroke="#E2E8F0"
                strokeWidth="1"
                strokeDasharray="3 3"
              />
              <path
                d="M 110 180 Q 200 200 310 325"
                fill="none"
                stroke="#E2E8F0"
                strokeWidth="1"
                strokeDasharray="3 3"
              />
              <path
                d="M 270 85 Q 280 180 250 330"
                fill="none"
                stroke="#E2E8F0"
                strokeWidth="1"
                strokeDasharray="3 3"
              />
            </svg>

            {/* Bubble Pins placed at exact Saudi cities */}
            {mapPoints.map((pt) => {
              // Exact coordinate mapping matching image
              const coords: Record<string, { top: string; left: string }> = {
                'الرياض': { top: '48%', left: '60%' },
                'جدة': { top: '60%', left: '32%' },
                'الدمام': { top: '38%', left: '74%' },
                'المدينة المنورة': { top: '42%', left: '38%' },
                'مكة': { top: '65%', left: '36%' },
                'تبوك': { top: '22%', left: '26%' },
              };
              const pos = coords[pt.name] || { top: '50%', left: '50%' };

              return (
                <div
                  key={pt.name}
                  style={{ top: pos.top, left: pos.left }}
                  onClick={() =>
                    openDrilldown(`الحوادث الموثقة في: ${pt.name}`, {
                      city: pt.name,
                    })
                  }
                  className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-20"
                >
                  <div className="relative flex items-center justify-center">
                    {/* Outer pulse glow for Riyadh */}
                    {pt.count >= 10 && (
                      <span className="animate-ping absolute inline-flex h-8 w-8 rounded-full bg-red-400 opacity-30"></span>
                    )}

                    {/* Bubble pin */}
                    <div
                      style={{
                        backgroundColor: pt.color,
                        width: `${Math.max(22, Math.min(36, 18 + pt.count * 1.3))}px`,
                        height: `${Math.max(22, Math.min(36, 18 + pt.count * 1.3))}px`,
                      }}
                      className="rounded-full text-white font-extrabold text-xs flex items-center justify-center shadow-md border-2 border-white group-hover:scale-110 transition-transform"
                    >
                      {pt.count}
                    </div>

                    {/* City Name Label */}
                    <span className="absolute top-full mt-1 px-1.5 py-0.5 rounded bg-white/90 shadow-xs border border-slate-200 text-[10px] font-bold text-[#10213D] whitespace-nowrap opacity-90 group-hover:opacity-100">
                      {pt.name}
                    </span>
                  </div>
                </div>
              );
            })}

            {/* Map Zoom Controls on bottom-left of map canvas */}
            <div className="absolute bottom-2 left-2 flex flex-col gap-1 bg-white border border-[#E7ECF3] rounded-lg shadow-xs p-0.5 z-10">
              <button className="p-1 text-[#64748B] hover:text-[#10213D] hover:bg-slate-50 rounded">
                <Plus className="w-3.5 h-3.5" />
              </button>
              <button className="p-1 text-[#64748B] hover:text-[#10213D] hover:bg-slate-50 rounded">
                <Minus className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Legend on the right side */}
            <div className="absolute top-2 right-2 bg-white/95 backdrop-blur-xs border border-[#E7ECF3] rounded-xl p-2.5 shadow-xs space-y-1 text-[10px] z-10 text-right">
              <div className="font-bold text-[#10213D] pb-0.5">عدد الحوادث</div>
              <div className="flex items-center gap-1.5 justify-end">
                <span className="text-[#64748B]">+10</span>
                <span className="w-2.5 h-2.5 rounded-full bg-[#EF4444]" />
              </div>
              <div className="flex items-center gap-1.5 justify-end">
                <span className="text-[#64748B]">6 - 10</span>
                <span className="w-2.5 h-2.5 rounded-full bg-[#F97316]" />
              </div>
              <div className="flex items-center gap-1.5 justify-end">
                <span className="text-[#64748B]">3 - 5</span>
                <span className="w-2.5 h-2.5 rounded-full bg-[#FBBF24]" />
              </div>
              <div className="flex items-center gap-1.5 justify-end">
                <span className="text-[#64748B]">1 - 2</span>
                <span className="w-2.5 h-2.5 rounded-full bg-[#3B82F6]" />
              </div>
            </div>
          </div>

          {/* Disclaimer at bottom */}
          <div className="pt-2 text-[10.5px] text-[#64748B] text-center border-t border-[#F1F5F9]">
            تعكس المواقع الحوادث المرصودة في مصادرنا، وليست إحصائية رسمية لمعدلات الاحتيال.
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------------- */}
      {/* 3. Middle Row: 3 Columns (Left: Patterns | Mid: Flow | Right: Change)*/}
      {/* ------------------------------------------------------------------- */}
      <div dir="ltr" className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        {/* Column 1 (Left): أبرز الأنماط والاتجاهات (Approx 4 of 12 cols = ~33%) */}
        <div
          dir="rtl"
          className="lg:col-span-4 bg-white border border-[#E7ECF3] rounded-2xl p-5 md:p-6 shadow-xs space-y-4"
        >
          <div className="flex items-center justify-between text-sm font-bold text-[#10213D]">
            <div className="flex items-center gap-1.5">
              <span>أبرز الأنماط والاتجاهات</span>
              <Info className="w-4 h-4 text-[#64748B]" />
            </div>
          </div>

          <div className="space-y-3">
            {patterns.map((p, idx) => (
              <div
                key={p.key}
                onClick={() =>
                  openDrilldown(`الحوادث المصنفة كـ: ${p.name_ar}`, { fraud_type: p.key })
                }
                className="group cursor-pointer p-3 rounded-xl bg-white hover:bg-[#F8FAFC] border border-[#E7ECF3] hover:border-[#CBD5E1] transition-all flex items-center justify-between"
              >
                {/* Right: Icon + Name + Meta */}
                <div className="flex items-center gap-3">
                  <div
                    className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 ${
                      idx === 0
                        ? 'bg-[#FEF2F2] text-[#EF4444]'
                        : idx === 1
                        ? 'bg-[#EFF6FF] text-[#2563EB]'
                        : idx === 2
                        ? 'bg-[#FFF7ED] text-[#F97316]'
                        : idx === 3
                        ? 'bg-[#ECFDF5] text-[#10B981]'
                        : 'bg-[#FDF2F8] text-[#DB2777]'
                    }`}
                  >
                    {idx === 0 && <AlertTriangle className="w-4 h-4" />}
                    {idx === 1 && <MessageSquare className="w-4 h-4" />}
                    {idx === 2 && <LinkIcon className="w-4 h-4" />}
                    {idx === 3 && <ShoppingCart className="w-4 h-4" />}
                    {idx === 4 && <Briefcase className="w-4 h-4" />}
                  </div>

                  <div>
                    <div className="font-bold text-xs md:text-sm text-[#10213D]">
                      {p.name_ar}
                    </div>
                    <div className="text-[11px] text-[#64748B]">
                      {p.incidents_count} حوادث • {p.sources_count} مصادر
                    </div>
                  </div>
                </div>

                {/* Left: Change % + Mini Sparkline + Chevron */}
                <div className="flex items-center gap-2">
                  <span
                    className={`text-xs font-bold flex items-center gap-0.5 ${
                      p.direction === 'up' ? 'text-[#EF4444]' : 'text-[#10B981]'
                    }`}
                  >
                    {p.direction === 'up' ? '↑' : '↓'}
                    {p.change_pct > 0 ? `+${p.change_pct}%` : `${p.change_pct}%`}
                  </span>

                  {/* Sparkline */}
                  <div className="flex items-end gap-0.5 h-4 w-7">
                    {p.sparkline.map((val, sIdx) => (
                      <div
                        key={sIdx}
                        style={{ height: `${(val / 7) * 16}px` }}
                        className={`w-1 rounded-t-xs ${
                          p.direction === 'up' ? 'bg-[#EF4444]' : 'bg-[#10B981]'
                        }`}
                      />
                    ))}
                  </div>

                  <ChevronLeft className="w-4 h-4 text-[#CBD5E1] group-hover:text-[#64748B] group-hover:translate-x-[-2px] transition-transform" />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Column 2 (Middle): كيف تحدث عمليات الاحتيال؟ (Sankey Flow Diagram in RTL Order) (~42%) */}
        {flow && (
          <div
            dir="rtl"
            className="lg:col-span-5 bg-white border border-[#E7ECF3] rounded-2xl p-5 md:p-6 shadow-xs space-y-3 flex flex-col justify-between"
          >
            <h3 className="text-sm font-bold text-[#10213D] text-center">
              كيف تحدث عمليات الاحتيال؟
            </h3>

            {/* 3 Columns Layout: Right = Channels, Center = Typologies, Left = Sectors (RTL Flow) */}
            <div dir="rtl" className="relative pt-1 min-h-[270px]">
              {/* SVG Alluvial Ribbon Overlay connecting Col 1 (Right) -> Col 2 (Mid) -> Col 3 (Left) */}
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none z-0"
                preserveAspectRatio="none"
                viewBox="0 0 300 240"
              >
                <defs>
                  <linearGradient id="flowGradRed" x1="1" y1="0" x2="0" y2="0">
                    <stop offset="0%" stopColor="#EF4444" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="#DC2626" stopOpacity="0.25" />
                  </linearGradient>
                  <linearGradient id="flowGradBlue" x1="1" y1="0" x2="0" y2="0">
                    <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="#2563EB" stopOpacity="0.25" />
                  </linearGradient>
                  <linearGradient id="flowGradAmber" x1="1" y1="0" x2="0" y2="0">
                    <stop offset="0%" stopColor="#F59E0B" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="#D97706" stopOpacity="0.25" />
                  </linearGradient>
                  <linearGradient id="flowGradEmerald" x1="1" y1="0" x2="0" y2="0">
                    <stop offset="0%" stopColor="#10B981" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="#059669" stopOpacity="0.25" />
                  </linearGradient>
                  <linearGradient id="flowGradPurple" x1="1" y1="0" x2="0" y2="0">
                    <stop offset="0%" stopColor="#8B5CF6" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="#7C3AED" stopOpacity="0.25" />
                  </linearGradient>
                </defs>

                {/* Flow paths: Channels (Right: x ~ 215) to Types (Mid: x ~ 185) */}
                <path
                  d="M 215 24 C 200 24, 200 24, 185 24 L 185 42 C 200 42, 200 42, 215 42 Z"
                  fill="url(#flowGradRed)"
                />
                <path
                  d="M 215 30 C 200 30, 200 62, 185 62 L 185 76 C 200 76, 200 42, 215 42 Z"
                  fill="url(#flowGradAmber)"
                />
                <path
                  d="M 215 62 C 200 62, 200 28, 185 28 L 185 36 C 200 36, 200 72, 215 72 Z"
                  fill="url(#flowGradRed)"
                />
                <path
                  d="M 215 100 C 200 100, 200 100, 185 100 L 185 116 C 200 116, 200 116, 215 116 Z"
                  fill="url(#flowGradBlue)"
                />
                <path
                  d="M 215 138 C 200 138, 200 138, 185 138 L 185 154 C 200 154, 200 154, 215 154 Z"
                  fill="url(#flowGradEmerald)"
                />
                <path
                  d="M 215 176 C 200 176, 200 68, 185 68 L 185 80 C 200 80, 200 190, 215 190 Z"
                  fill="url(#flowGradAmber)"
                />
                <path
                  d="M 215 184 C 200 184, 200 176, 185 176 L 185 190 C 200 190, 200 196, 215 196 Z"
                  fill="url(#flowGradPurple)"
                />

                {/* Flow paths: Types (Mid: x ~ 115) to Sectors (Left: x ~ 85) */}
                <path
                  d="M 115 24 C 100 24, 100 24, 85 24 L 85 42 C 100 42, 100 42, 115 42 Z"
                  fill="url(#flowGradRed)"
                />
                <path
                  d="M 115 30 C 100 30, 100 62, 85 62 L 85 76 C 100 76, 100 42, 115 42 Z"
                  fill="url(#flowGradRed)"
                />
                <path
                  d="M 115 62 C 100 62, 100 28, 85 28 L 85 38 C 100 38, 100 74, 115 74 Z"
                  fill="url(#flowGradAmber)"
                />
                <path
                  d="M 115 70 C 100 70, 100 176, 85 176 L 85 190 C 100 190, 100 80, 115 80 Z"
                  fill="url(#flowGradAmber)"
                />
                <path
                  d="M 115 100 C 100 100, 100 100, 85 100 L 85 116 C 100 116, 100 116, 115 116 Z"
                  fill="url(#flowGradBlue)"
                />
                <path
                  d="M 115 138 C 100 138, 100 138, 85 138 L 85 154 C 100 154, 100 154, 115 154 Z"
                  fill="url(#flowGradEmerald)"
                />
                <path
                  d="M 115 176 C 100 176, 100 36, 85 36 L 85 44 C 100 44, 100 186, 115 186 Z"
                  fill="url(#flowGradPurple)"
                />
              </svg>

              {/* 3 Interactive Node Columns in Arabic Flow: Channels -> Types -> Sectors */}
              <div className="relative z-10 grid grid-cols-3 gap-3 text-xs">
                {/* Column 1 (RIGHT): القنوات المستخدمة */}
                <div className="space-y-2">
                  <div className="font-bold text-[11px] text-[#64748B] pb-1 text-center">
                    القنوات المستخدمة
                  </div>
                  <div className="space-y-1.5">
                    {flow.channels.map((c) => (
                      <div
                        key={c.id}
                        onMouseEnter={() => setHoveredNode(c.id)}
                        onMouseLeave={() => setHoveredNode(null)}
                        onClick={() =>
                          openDrilldown(`الحوادث المنفذة عبر: ${c.name_ar}`, {
                            channel: c.id !== 'other_channel' ? c.id : undefined,
                          })
                        }
                        className="p-1.5 rounded-lg bg-white hover:bg-slate-50 border border-slate-200/80 shadow-xs flex items-center justify-between cursor-pointer transition text-[11px]"
                      >
                        <div className="flex items-center gap-1.5 truncate">
                          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: c.color }} />
                          <span className="font-medium text-[#10213D] truncate">{c.name_ar}</span>
                        </div>
                        <span className="font-bold text-[#64748B] text-[10px]">{c.percentage}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Column 2 (MIDDLE): نوع الاحتيال */}
                <div className="space-y-2">
                  <div className="font-bold text-[11px] text-[#64748B] pb-1 text-center">
                    نوع الاحتيال
                  </div>
                  <div className="space-y-1.5">
                    {flow.typologies.map((t) => (
                      <div
                        key={t.id}
                        onMouseEnter={() => setHoveredNode(t.id)}
                        onMouseLeave={() => setHoveredNode(null)}
                        onClick={() =>
                          openDrilldown(`حوادث نوع: ${t.name_ar}`, {
                            fraud_type: t.id !== 'other_fraud' ? t.id : undefined,
                          })
                        }
                        className="p-1.5 rounded-lg bg-white hover:bg-slate-50 border border-slate-200/80 shadow-xs flex items-center justify-between cursor-pointer transition text-[11px]"
                      >
                        <div className="flex items-center gap-1.5 truncate">
                          <span className="w-1.5 h-3.5 rounded-xs shrink-0" style={{ backgroundColor: t.color }} />
                          <span className="font-medium text-[#10213D] truncate">{t.name_ar}</span>
                        </div>
                        <span className="font-bold text-[#64748B] text-[10px]">{t.percentage}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Column 3 (LEFT): الجهات أو القطاعات المستهدفة */}
                <div className="space-y-2">
                  <div className="font-bold text-[11px] text-[#64748B] pb-1 text-center">
                    الجهات أو القطاعات المستهدفة
                  </div>
                  <div className="space-y-1.5">
                    {flow.sectors.map((s) => (
                      <div
                        key={s.id}
                        onMouseEnter={() => setHoveredNode(s.id)}
                        onMouseLeave={() => setHoveredNode(null)}
                        onClick={() =>
                          openDrilldown(`الحوادث المستهدفة لقطاع: ${s.name_ar}`, {
                            sector: s.id !== 'other_sector' ? s.id : undefined,
                          })
                        }
                        className="p-1.5 rounded-lg bg-white hover:bg-slate-50 border border-slate-200/80 shadow-xs flex items-center justify-between cursor-pointer transition text-[11px]"
                      >
                        <div className="flex items-center gap-1.5 truncate">
                          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: s.color }} />
                          <span className="font-medium text-[#10213D] truncate">{s.name_ar}</span>
                        </div>
                        <span className="font-bold text-[#64748B] text-[10px]">{s.percentage}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Column 3 (Right): ما الذي تغيّر؟ (~25%) */}
        <div
          dir="rtl"
          className="lg:col-span-3 bg-white border border-[#E7ECF3] rounded-2xl p-5 md:p-6 shadow-xs space-y-4"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-[#10213D]">ما الذي تغيّر؟</h3>
            <span className="text-[11px] text-[#64748B]">مقارنة بالفترة السابقة</span>
          </div>

          <div className="space-y-3">
            {changed.map((c) => (
              <div
                key={c.id}
                onClick={() => openDrilldown(`تحول ميتا فيو: ${c.title}`)}
                className="group cursor-pointer p-2.5 rounded-xl bg-white hover:bg-[#F8FAFC] border border-[#E7ECF3] flex items-center justify-between transition text-xs"
              >
                {/* Right: Icon box + Texts */}
                <div className="flex items-center gap-3">
                  {/* Left: Direction Arrow */}
                  <div className="pl-1">
                    <span
                      className={`font-bold text-sm ${
                        c.direction === 'up' ? 'text-[#EF4444]' : 'text-[#10B981]'
                      }`}
                    >
                      {c.direction === 'up' ? '↑' : '↓'}
                    </span>
                  </div>

                  {/* Texts */}
                  <div className="truncate">
                    <div className="font-bold text-xs text-[#10213D] truncate">{c.title}</div>
                    <div className="text-[11px] text-[#64748B]">{c.detail}</div>
                  </div>
                </div>

                {/* Far Right: Icon box */}
                <div className="p-2 rounded-lg bg-[#EFF6FF] text-[#2563EB] shrink-0">
                  {c.id === 1 && <CreditCard className="w-3.5 h-3.5 text-[#2563EB]" />}
                  {c.id === 2 && <Briefcase className="w-3.5 h-3.5 text-[#EF4444]" />}
                  {c.id === 3 && <LinkIcon className="w-3.5 h-3.5 text-[#8B5CF6]" />}
                  {c.id === 4 && <ShoppingCart className="w-3.5 h-3.5 text-[#10B981]" />}
                  {c.id === 5 && <Coins className="w-3.5 h-3.5 text-[#F59E0B]" />}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------------- */}
      {/* 4. Bottom Row: 3 Columns (Left: Latest | Mid: Banking | Right: Qual)*/}
      {/* ------------------------------------------------------------------- */}
      <div dir="ltr" className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        {/* Column 1 (Left): أحدث الحوادث (Approx 5 of 12 cols = ~42%) */}
        <div
          dir="rtl"
          className="lg:col-span-5 bg-white border border-[#E7ECF3] rounded-2xl p-5 md:p-6 shadow-xs space-y-4"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-[#10213D]">أحدث الحوادث</h3>
            <button
              onClick={() => openDrilldown('جميع الحوادث المرصودة في المنصة')}
              className="text-xs font-semibold text-[#2563EB] hover:underline flex items-center gap-1"
            >
              <span>عرض الكل</span>
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Table matching reference order: Right = التاريخ, Left = Chevron */}
          <div className="overflow-x-auto">
            <table className="w-full text-right text-xs">
              <thead>
                <tr className="border-b border-[#F1F5F9] text-[#64748B] font-medium text-[11px]">
                  <th className="pb-2 text-right">التاريخ</th>
                  <th className="pb-2 text-right">العنوان</th>
                  <th className="pb-2 text-right">النوع</th>
                  <th className="pb-2 text-right">الموقع</th>
                  <th className="pb-2 text-right">القناة</th>
                  <th className="pb-2 text-right">الحالة</th>
                  <th className="pb-2 w-5 text-left"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#F8FAFC]">
                {incidents.map((inc) => (
                  <tr
                    key={inc.id}
                    onClick={() => viewSingleIncident(inc.id)}
                    className="cursor-pointer hover:bg-[#F8FAFC] transition group"
                  >
                    <td className="py-2.5 text-[#64748B] text-[11px] text-right whitespace-nowrap">
                      {inc.date}
                    </td>
                    <td className="py-2.5 text-[#10213D] font-bold max-w-[150px] truncate text-right">
                      {inc.title}
                    </td>
                    <td className="py-2.5 text-[#64748B] text-right whitespace-nowrap">
                      {inc.type}
                    </td>
                    <td className="py-2.5 text-[#10213D] font-medium text-right whitespace-nowrap">
                      {inc.location}
                    </td>
                    <td className="py-2.5 text-[#64748B] text-right whitespace-nowrap">
                      {inc.channel}
                    </td>
                    <td className="py-2.5 text-right whitespace-nowrap">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          inc.status === 'مؤكدة'
                            ? 'bg-[#ECFDF5] text-[#10B981]'
                            : 'bg-[#FFFBEB] text-[#D97706]'
                        }`}
                      >
                        {inc.status}
                      </span>
                    </td>
                    <td className="py-2.5 text-[#CBD5E1] group-hover:text-[#2563EB] text-left">
                      <ChevronRight className="w-3.5 h-3.5" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Column 2 (Middle): ماذا يعني ذلك للبنوك؟ (Approx 4 of 12 cols = ~33%) */}
        <div
          dir="rtl"
          className="lg:col-span-4 bg-white border border-[#E7ECF3] rounded-2xl p-5 md:p-6 shadow-xs space-y-4"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-[#10213D]">ماذا يعني ذلك للبنوك؟</h3>
            <button
              onClick={() => openDrilldown('المخاطر والفرضيات المصرفية المرتبطة بالحوادث')}
              className="text-xs font-semibold text-[#2563EB] hover:underline flex items-center gap-1"
            >
              <span>عرض الكل</span>
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="space-y-3">
            {banking.map((b) => (
              <div
                key={b.id}
                onClick={() => openDrilldown(`التحليل المصرفي: ${b.title}`)}
                className="group cursor-pointer p-2.5 rounded-xl bg-white hover:bg-[#F8FAFC] border border-[#E7ECF3] flex items-start gap-3 transition text-xs"
              >
                {/* Icon on Right */}
                <div
                  className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5 ${
                    b.id === 1
                      ? 'bg-[#FEF2F2] text-[#EF4444]'
                      : b.id === 2
                      ? 'bg-[#EFF6FF] text-[#2563EB]'
                      : b.id === 3
                      ? 'bg-[#F1F5F9] text-[#64748B]'
                      : 'bg-[#EFF6FF] text-[#2563EB]'
                  }`}
                >
                  {b.id === 1 && <AlertTriangle className="w-3.5 h-3.5" />}
                  {b.id === 2 && <Landmark className="w-3.5 h-3.5" />}
                  {b.id === 3 && <Activity className="w-3.5 h-3.5" />}
                  {b.id === 4 && <Shield className="w-3.5 h-3.5" />}
                </div>

                {/* Details on Left */}
                <div className="flex-1 text-right">
                  <div className="font-bold text-xs text-[#10213D] leading-snug">{b.title}</div>
                  <div className="text-[11px] text-[#64748B] leading-relaxed mt-0.5">
                    {b.detail}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Column 3 (Right): جودة البيانات والمصادر (Approx 3 of 12 cols = ~25%) */}
        {quality && (
          <div
            dir="rtl"
            className="lg:col-span-3 bg-white border border-[#E7ECF3] rounded-2xl p-5 md:p-6 shadow-xs space-y-4 flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between pb-2">
                <h3 className="text-sm font-bold text-[#10213D]">جودة البيانات والمصادر</h3>
                <button
                  onClick={() => openDrilldown('تدقيق جودة البيانات والمصادر')}
                  className="text-xs font-semibold text-[#2563EB] hover:underline flex items-center gap-1"
                >
                  <span>عرض الكل</span>
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* 2x2 Metric Tiles in RTL order: Right column = 47 & 31, Left column = 52 & 24 */}
              <div dir="rtl" className="grid grid-cols-2 gap-2.5 pt-1">
                {/* Right Column: Tile 47 and Tile 31 */}
                <div className="space-y-2.5">
                  {/* Tile 47 */}
                  <div className="p-2.5 rounded-xl bg-white border border-[#E7ECF3] space-y-1">
                    <div className="flex items-center justify-between">
                      <AlertTriangle className="w-4 h-4 text-[#EF4444]" />
                      <span className="text-[10px] font-bold text-[#10B981] bg-[#ECFDF5] px-1.5 py-0.2 rounded">
                        {quality.incidents_change_pct}
                      </span>
                    </div>
                    <div className="text-lg font-extrabold text-[#10213D]">
                      {quality.incidents_extracted}
                    </div>
                    <div className="text-[10.5px] text-[#64748B]">حادثة مستخرجة</div>
                  </div>

                  {/* Tile 31 */}
                  <div className="p-2.5 rounded-xl bg-white border border-[#E7ECF3] space-y-1">
                    <div className="flex items-center justify-between">
                      <Clock className="w-4 h-4 text-[#F59E0B]" />
                      <span className="text-[10px] font-bold text-[#EF4444] bg-[#FEF2F2] px-1.5 py-0.2 rounded">
                        {quality.pending_change_pct}
                      </span>
                    </div>
                    <div className="text-lg font-extrabold text-[#10213D]">
                      {quality.pending_sources}
                    </div>
                    <div className="text-[10.5px] text-[#64748B]">مصدراً في الانتظار</div>
                  </div>
                </div>

                {/* Left Column: Tile 52 and Tile 24 */}
                <div className="space-y-2.5">
                  {/* Tile 52 */}
                  <div className="p-2.5 rounded-xl bg-white border border-[#E7ECF3] space-y-1">
                    <div className="flex items-center justify-between">
                      <FileText className="w-4 h-4 text-[#2563EB]" />
                      <span className="text-[10px] font-bold text-[#10B981] bg-[#ECFDF5] px-1.5 py-0.2 rounded">
                        {quality.articles_change_pct}
                      </span>
                    </div>
                    <div className="text-lg font-extrabold text-[#10213D]">
                      {quality.articles_analyzed}
                    </div>
                    <div className="text-[10.5px] text-[#64748B]">مقالاً محللاً</div>
                  </div>

                  {/* Tile 24 */}
                  <div className="p-2.5 rounded-xl bg-white border border-[#E7ECF3] space-y-1">
                    <div className="flex items-center justify-between">
                      <CheckCircle2 className="w-4 h-4 text-[#10B981]" />
                      <span className="text-[10px] font-bold text-[#10B981] bg-[#ECFDF5] px-1.5 py-0.2 rounded">
                        {quality.sources_change_pct}
                      </span>
                    </div>
                    <div className="text-lg font-extrabold text-[#10213D]">
                      {quality.active_sources}
                    </div>
                    <div className="text-[10.5px] text-[#64748B]">مصدراً نشطاً</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Progress Bar Section */}
            <div className="pt-3 border-t border-[#F1F5F9] space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-[#10213D]">
                  {quality.fields_completion_pct}%
                </span>
                <span className="text-[11px] text-[#64748B]">
                  اكتمال استخراج الحقول الرئيسية
                </span>
              </div>
              <div className="w-full bg-[#E2E8F0] h-2 rounded-full overflow-hidden">
                <div
                  className="bg-[#10B981] h-full rounded-full transition-all duration-500"
                  style={{ width: `${quality.fields_completion_pct}%` }}
                />
              </div>
              <div className="text-[10px] text-[#94A3B8] leading-tight">
                {quality.missing_fields_note ||
                  'بعض الحقول (الموقع، القيمة المالية) غير متوفرة في عدد من الحوادث'}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ------------------------------------------------------------------- */}
      {/* 5. Interactive Drill-Down Evidence Modal                            */}
      {/* ------------------------------------------------------------------- */}
      {drilldownOpen && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div
            dir="rtl"
            className="bg-white rounded-2xl max-w-3xl w-full max-h-[85vh] shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150"
          >
            {/* Modal Header */}
            <div className="p-5 border-b border-[#E7ECF3] flex items-center justify-between bg-slate-50/50">
              <div className="flex items-center gap-2">
                <ShieldAlert className="w-5 h-5 text-blue-600" />
                <h3 className="font-bold text-sm md:text-base text-[#10213D]">
                  {drilldownTitle}
                </h3>
              </div>
              <button
                onClick={() => setDrilldownOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-5 overflow-y-auto space-y-4">
              {drilldownLoading ? (
                <div className="p-8 text-center text-sm text-[#64748B] animate-pulse">
                  جاري جلب الأدلة والمصادر المستقلة من سجل الحوادث...
                </div>
              ) : selectedIncident ? (
                // Detailed single incident view
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-[#ECFDF5] text-[#10B981]">
                      حادثة موثقة
                    </span>
                    <span className="text-xs text-[#64748B]">
                      {selectedIncident.published_at?.slice(0, 10)}
                    </span>
                  </div>

                  <h4 className="text-base font-bold text-[#10213D]">
                    {selectedIncident.title}
                  </h4>

                  <p className="text-xs text-[#64748B] leading-relaxed">
                    {selectedIncident.summary}
                  </p>

                  {/* Metadata Chips */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2">
                    <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                      <div className="text-[10px] text-[#64748B]">نوع الاحتيال</div>
                      <div className="font-bold text-xs text-[#10213D] mt-0.5">
                        {selectedIncident.fraud_type_ar || selectedIncident.fraud_type}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                      <div className="text-[10px] text-[#64748B]">قناة الهجوم</div>
                      <div className="font-bold text-xs text-[#10213D] mt-0.5">
                        {selectedIncident.channel_ar || selectedIncident.channel || 'غير محددة'}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                      <div className="text-[10px] text-[#64748B]">الموقع الموثق</div>
                      <div className="font-bold text-xs text-[#10213D] mt-0.5">
                        {selectedIncident.incident_city || selectedIncident.incident_country || 'المملكة العربية السعودية'}
                      </div>
                    </div>
                    <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                      <div className="text-[10px] text-[#64748B]">الجهة المستهدفة</div>
                      <div className="font-bold text-xs text-[#10213D] mt-0.5">
                        {selectedIncident.targeted_sector_ar || selectedIncident.targeted_sector || 'الأفراد'}
                      </div>
                    </div>
                  </div>

                  {/* Verbatim Evidence Quote */}
                  {selectedIncident.evidence_text && (
                    <div className="p-3.5 rounded-xl bg-[#F8FAFC] border-r-4 border-blue-500 text-xs text-[#334155] space-y-1">
                      <div className="font-bold text-[#10213D]">النص المقتبس من المصدر الأصلي:</div>
                      <blockquote className="italic">
                        "{selectedIncident.evidence_text}"
                      </blockquote>
                    </div>
                  )}

                  {/* Source Links */}
                  {selectedIncident.articles && selectedIncident.articles.length > 0 && (
                    <div className="space-y-1.5 pt-2">
                      <div className="font-bold text-xs text-[#10213D]">المصادر المرجعية المستقلة:</div>
                      <div className="space-y-1">
                        {selectedIncident.articles.map((art, uIdx) => (
                          <a
                            key={uIdx}
                            href={art.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1.5 text-xs text-blue-600 hover:underline truncate"
                          >
                            <ExternalLink className="w-3.5 h-3.5 shrink-0" />
                            <span className="truncate">{art.headline || art.url}</span>
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : drilldownData && drilldownData.items && drilldownData.items.length > 0 ? (
                // List of incidents matching filter
                <div className="space-y-2.5">
                  <div className="text-xs text-[#64748B]">
                    تم العثور على {drilldownData.total} حادثة مرتبطة بهذا التحليل:
                  </div>
                  {drilldownData.items.map((inc) => (
                    <div
                      key={inc.id}
                      onClick={() => setSelectedIncident(inc)}
                      className="p-3 rounded-xl border border-[#E7ECF3] hover:border-blue-400 hover:bg-slate-50/70 transition cursor-pointer flex items-center justify-between"
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-[#ECFDF5] text-[#10B981]">
                            مؤكدة
                          </span>
                          <h5 className="font-bold text-xs text-[#10213D]">
                            {inc.title}
                          </h5>
                        </div>
                        <div className="text-[11px] text-[#64748B]">
                          {inc.fraud_type_ar || inc.fraud_type} • {inc.incident_city || 'السعودية'} • {inc.channel_ar || inc.channel || 'غير محددة'}
                        </div>
                      </div>
                      <ChevronLeft className="w-4 h-4 text-slate-400" />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-8 text-center text-xs text-[#64748B]">
                  لا تتوفر حوادث إضافية مسجلة لهذه الفئة حالياً.
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-[#E7ECF3] bg-slate-50/50 flex justify-end">
              <button
                onClick={() => {
                  if (selectedIncident && drilldownData?.items?.length) {
                    setSelectedIncident(null);
                  } else {
                    setDrilldownOpen(false);
                  }
                }}
                className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-[#10213D] rounded-xl text-xs font-semibold transition"
              >
                {selectedIncident && drilldownData?.items?.length ? 'العودة للقائمة' : 'إغلاق'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
