// src/pages/FraudIntelligenceDashboard.tsx
// MetaView 3 — Fraud Intelligence Dashboard
// Pixel-Accurate Implementation Matching Approved Reference Design
import React, { useState, useEffect, useCallback, useMemo } from 'react';
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
  RotateCcw,
  RefreshCw,
  Globe,
  Crosshair,
} from 'lucide-react';
import { apiService } from '../services/api';
import {
  DashboardViewResponse,
  EnrichedIncident,
  DrilldownResponse,
  MapPoint,
  WorldHotspot,
  TopCountryItem,
} from '../types/fraudIntelligence';
import { timeAgo } from '../utils/formatDate';
import { fmtDateTime } from '../components/ui';
import { useRefresh } from '../context/RefreshContext';
import { WORLD_BASE_PATHS, WORLD_HIGHLIGHT_PATHS } from '../data/worldMapPaths';

const MINOR_DOTS = [
  { id: 'ca_dot', name: 'كندا', country: 'CA', x: 17.8, y: 27.5 },
  { id: 'br_dot', name: 'البرازيل', country: 'BR', x: 30.8, y: 55.6 },
  { id: 'af_dot', name: 'وسط وغرب أفريقيا', country: 'AFR', x: 44.6, y: 56.0 },
  { id: 'cn_dot', name: 'شرق آسيا', country: 'CN', x: 62.5, y: 35.0 },
  { id: 'ea_dot', name: 'شرق آسيا واليابان', country: 'JP', x: 68.2, y: 34.0 },
];

const WorldMapVector: React.FC = () => (
  <svg
    viewBox="0 0 1000 500"
    className="w-full h-full select-none"
    preserveAspectRatio="xMidYMid meet"
  >
    <defs>
      <linearGradient id="oceanGlow" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#FAFCFF" />
        <stop offset="100%" stopColor="#F1F5F9" />
      </linearGradient>
    </defs>

    {/* Background Base */}
    <rect width="1000" height="500" fill="url(#oceanGlow)" />

    {/* Realistic World Base Continents */}
    <path
      d={WORLD_BASE_PATHS}
      fill="#F1F5F9"
      stroke="#D8E2ED"
      strokeWidth="0.8"
      strokeLinejoin="round"
    />

    {/* Highlighted Key Countries (matching screenshot) */}
    <path
      d={WORLD_HIGHLIGHT_PATHS}
      fill="#DBEAFE"
      stroke="#93C5FD"
      strokeWidth="1.1"
      strokeLinejoin="round"
    />
  </svg>
);

export const FraudIntelligenceDashboard: React.FC = () => {
  // Centralized Global Filter States (Single Source of Truth)
  const [datePreset, setDatePreset] = useState<7 | 30 | 90 | 'custom'>(30);
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');
  const [dateBasis, setDateBasis] = useState<'first_reported_at' | 'occurred_at'>('first_reported_at');
  const [dateMenuOpen, setDateMenuOpen] = useState<boolean>(false);

  const [geoFilter, setGeoFilter] = useState<string>('السعودية');
  const [fraudTypeFilter, setFraudTypeFilter] = useState<string>('all');
  const [mapScope, setMapScope] = useState<'world' | 'regions' | 'cities'>('world');
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  const [selectedCityPoint, setSelectedCityPoint] = useState<MapPoint | null>(null);
  const [cityModalOpen, setCityModalOpen] = useState(false);

  // View presentation states for Adaptive UI (Map vs List, Sankey vs Mechanisms)
  const [mapViewType, setMapViewType] = useState<'map' | 'list'>('map');
  const [flowViewType, setFlowViewType] = useState<'sankey' | 'mechanisms'>('sankey');

  // Active Flow Hover State (for Sankey highlighting)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // Data States
  const [data, setData] = useState<DashboardViewResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const worldHotspots: WorldHotspot[] = useMemo(() => {
    return data?.world_hotspots || [];
  }, [data?.world_hotspots]);

  const topCountriesList: TopCountryItem[] = useMemo(() => {
    return data?.top_countries || [];
  }, [data?.top_countries]);

  // Race condition protection ref
  const reqSequenceRef = React.useRef<number>(0);

  // Drilldown Modal State
  const [drilldownOpen, setDrilldownOpen] = useState(false);
  const [drilldownTitle, setDrilldownTitle] = useState('');
  const [drilldownLoading, setDrilldownLoading] = useState(false);
  const [drilldownData, setDrilldownData] = useState<DrilldownResponse | null>(null);
  const [selectedIncident, setSelectedIncident] = useState<EnrichedIncident | null>(null);

  // Reset all filters to default
  const handleResetFilters = () => {
    setDatePreset(30);
    setDateFrom('');
    setDateTo('');
    setGeoFilter('السعودية');
    setFraudTypeFilter('all');
    setDateMenuOpen(false);
  };

  const isFiltered = datePreset !== 30 || geoFilter !== 'السعودية' || fraudTypeFilter !== 'all' || Boolean(dateFrom);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(new Date());

  const {
    isRefreshing,
    lastUIRefresh,
    lastSourceFetch,
    lastDataProcessing,
    triggerGlobalRefresh,
    updateTelemetry,
  } = useRefresh();

  // Fetch Dashboard Data Centrally (supports silent background updates & forceRefresh cache-busting)
  const loadDashboardData = useCallback(async (isSilent = false, isForce = false) => {
    const reqId = ++reqSequenceRef.current;
    try {
      if (!isSilent) setLoading(true);
      setError(null);

      const geoParam = geoFilter === 'السعودية' ? 'SA' : (geoFilter === 'العالم' ? 'ALL' : geoFilter);

      const res = await apiService.getFraudIntelligenceDashboardView({
        windowDays: typeof datePreset === 'number' ? datePreset : 30,
        geo: geoParam,
        country: geoParam,
        mapMode: mapScope === 'world' ? 'regions' : mapScope,
        fraudType: fraudTypeFilter !== 'all' ? fraudTypeFilter : undefined,
        dateFrom: datePreset === 'custom' && dateFrom ? dateFrom : undefined,
        dateTo: datePreset === 'custom' && dateTo ? dateTo : undefined,
        dateBasis: dateBasis,
        forceRefresh: isForce,
      });

      // Drop stale responses if a newer request was dispatched
      if (reqId === reqSequenceRef.current) {
        setData(res);
        setLastUpdated(new Date());
        if (res.telemetry) {
          updateTelemetry(res.telemetry);
        }
      }
    } catch (err: any) {
      if (reqId === reqSequenceRef.current) {
        console.error('Failed to load dashboard view:', err);
        if (!isSilent) setError(err?.message || 'تعذر تحميل بيانات ميتا فيو');
      }
    } finally {
      if (reqId === reqSequenceRef.current && !isSilent) {
        setLoading(false);
      }
    }
  }, [datePreset, dateFrom, dateTo, dateBasis, geoFilter, mapScope, fraudTypeFilter, updateTelemetry]);

  // Initial and reactive load
  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  // Auto-refresh every 60 seconds when document is visible so newest updates appear automatically
  useEffect(() => {
    const timer = setInterval(() => {
      if (document.visibilityState === 'visible') {
        loadDashboardData(true, false);
      }
    }, 60000);

    const handleRefresh = (e?: Event) => {
      const isForce = (e as CustomEvent)?.detail?.forceRefresh ?? false;
      loadDashboardData(false, isForce);
    };
    window.addEventListener('metaview:refresh', handleRefresh);

    return () => {
      clearInterval(timer);
      window.removeEventListener('metaview:refresh', handleRefresh);
    };
  }, [loadDashboardData]);

  // Open Drilldown Modal
  const openDrilldown = async (title: string, params: Record<string, any> = {}) => {
    setDrilldownTitle(title);
    setDrilldownOpen(true);
    setDrilldownLoading(true);
    setSelectedIncident(null);
    try {
      const activeCountry = geoFilter === 'السعودية' ? 'SA' : (geoFilter !== 'العالم' ? geoFilter : undefined);
      const activeFraudType = fraudTypeFilter !== 'all' ? fraudTypeFilter : undefined;
      const res = await apiService.getFraudIntelligenceIncidents({
        fraud_type: params.fraud_type !== undefined ? params.fraud_type : activeFraudType,
        channel: params.channel,
        sector: params.sector,
        city: params.city,
        country: params.country !== undefined ? params.country : activeCountry,
        incident_ids: params.incident_ids,
        window_days: !params.incident_ids && datePreset !== 'custom' ? Number(datePreset) : undefined,
        date_from: !params.incident_ids && datePreset === 'custom' && dateFrom ? dateFrom : undefined,
        date_to: !params.incident_ids && datePreset === 'custom' && dateTo ? dateTo : undefined,
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
            MetaView 3
          </h1>
          <p className="text-xs md:text-sm text-[#64748B]">
            رصد وتحليل اتجاهات الاحتيال المالي وبناء صورة تحليلية شاملة من مصادر موثوقة
          </p>
        </div>

        {/* Filters & Export controls on the RIGHT (Left-to-Right order matching image) */}
        <div dir="ltr" className="grid grid-cols-2 sm:flex sm:flex-wrap items-center gap-2 sm:gap-2.5 w-full md:w-auto">
          {/* 1. Date Range Dropdown with Popover */}
          <div className="col-span-2 sm:col-span-1 relative">
            <button
              onClick={() => setDateMenuOpen(!dateMenuOpen)}
              className="w-full sm:w-auto flex items-center justify-between sm:justify-start gap-2 px-3.5 py-2 bg-white border border-[#E7ECF3] rounded-xl text-xs font-semibold text-[#10213D] hover:bg-slate-50 transition shadow-xs"
            >
              <Calendar className="w-3.5 h-3.5 text-[#64748B]" />
              <span dir="rtl">
                {datePreset === 'custom'
                  ? `مخصص (${dateFrom || '...'} إلى ${dateTo || '...'})`
                  : `آخر ${datePreset} يوماً`}
              </span>
              <ChevronDown className="w-3.5 h-3.5 text-[#64748B]" />
            </button>

            {dateMenuOpen && (
              <div
                dir="rtl"
                className="absolute right-0 top-full mt-1.5 w-64 max-w-[calc(100vw-2rem)] bg-white border border-[#E7ECF3] rounded-2xl shadow-xl p-3 z-50 space-y-2 text-xs"
              >
                <div className="text-[11px] font-bold text-[#64748B] pb-1 border-b border-slate-100">
                  نطاق التحليل الزمني
                </div>
                <div className="space-y-1">
                  {[7, 30, 90].map((days) => (
                    <button
                      key={days}
                      onClick={() => {
                        setDatePreset(days as 7 | 30 | 90);
                        setDateFrom('');
                        setDateTo('');
                        setDateMenuOpen(false);
                      }}
                      className={`w-full text-right px-3 py-1.5 rounded-lg font-medium transition ${
                        datePreset === days
                          ? 'bg-blue-50 text-blue-600 font-bold'
                          : 'hover:bg-slate-50 text-slate-700'
                      }`}
                    >
                      آخر {days} يوماً
                    </button>
                  ))}
                  <button
                    onClick={() => setDatePreset('custom')}
                    className={`w-full text-right px-3 py-1.5 rounded-lg font-medium transition ${
                      datePreset === 'custom'
                        ? 'bg-blue-50 text-blue-600 font-bold'
                        : 'hover:bg-slate-50 text-slate-700'
                    }`}
                  >
                    نطاق زمني مخصص...
                  </button>
                </div>

                {datePreset === 'custom' && (
                  <div className="pt-2 border-t border-slate-100 space-y-2">
                    <div>
                      <label className="text-[10px] text-[#64748B] block mb-0.5">من تاريخ:</label>
                      <input
                        type="date"
                        value={dateFrom}
                        onChange={(e) => setDateFrom(e.target.value)}
                        className="w-full px-2 py-1 border border-slate-200 rounded-lg text-xs outline-none"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] text-[#64748B] block mb-0.5">إلى تاريخ:</label>
                      <input
                        type="date"
                        value={dateTo}
                        onChange={(e) => setDateTo(e.target.value)}
                        className="w-full px-2 py-1 border border-slate-200 rounded-lg text-xs outline-none"
                      />
                    </div>
                    <button
                      onClick={() => setDateMenuOpen(false)}
                      className="w-full py-1.5 bg-[#102544] text-white rounded-lg text-xs font-semibold mt-1"
                    >
                      تطبيق النطاق
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 2. Geographic Dropdown */}
          <div className="relative">
            <select
              value={geoFilter}
              onChange={(e) => setGeoFilter(e.target.value)}
              className="w-full sm:w-auto appearance-none flex items-center gap-2 px-3.5 py-2 pr-8 pl-4 bg-white border border-[#E7ECF3] rounded-xl text-xs font-semibold text-[#10213D] hover:bg-slate-50 transition shadow-xs cursor-pointer outline-none"
            >
              <option value="السعودية">السعودية</option>
              <option value="العالم">جميع الدول</option>
              <option value="AE">الإمارات</option>
              <option value="EG">مصر</option>
              <option value="US">الولايات المتحدة</option>
              <option value="GB">بريطانيا</option>
            </select>
            <MapPin className="w-3.5 h-3.5 text-[#64748B] absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          </div>

          {/* 3. Canonical Fraud Type Dropdown */}
          <div className="relative">
            <select
              value={fraudTypeFilter}
              onChange={(e) => setFraudTypeFilter(e.target.value)}
              className="w-full sm:w-auto appearance-none flex items-center gap-2 px-3.5 py-2 pr-8 pl-4 bg-white border border-[#E7ECF3] rounded-xl text-xs font-semibold text-[#10213D] hover:bg-slate-50 transition shadow-xs cursor-pointer outline-none"
            >
              <option value="all">جميع الأنواع</option>
              <option value="gov_impersonation">انتحال جهات حكومية</option>
              <option value="bank_impersonation">انتحال بنوك ومصارف</option>
              <option value="investment_fraud">الاحتيال الاستثماري</option>
              <option value="phishing">روابط التصيد الاحتيالي</option>
              <option value="ecommerce_fraud">احتيال البيع والشراء</option>
              <option value="job_scam">الاحتيال الوظيفي</option>
              <option value="romance_scam">احتيال عاطفي وابتزاز</option>
              <option value="other">أنواع أخرى</option>
            </select>
            <ChevronDown className="w-3.5 h-3.5 text-[#64748B] absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          </div>

          {/* 4. Actions: Freshness, Refresh, Reset & Export */}
          <div className="col-span-2 sm:col-span-1 flex items-center gap-2">
            {/* Live freshness badge with 3 distinct timestamps */}
            <div
              className="hidden lg:flex items-center gap-2 px-3 py-2 bg-white border border-[#E7ECF3] rounded-xl text-[11px] font-medium text-[#64748B] shadow-xs cursor-default"
              title={`- آخر محاولة جلب من المصادر: ${lastSourceFetch ? fmtDateTime(lastSourceFetch) : 'غير متوفر'}\n- آخر معالجة للبيانات: ${lastDataProcessing ? fmtDateTime(lastDataProcessing) : 'غير متوفر'}\n- آخر طلب للواجهة: ${lastUIRefresh ? lastUIRefresh.toLocaleTimeString('ar-SA') : 'الآن'}`}
            >
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <div className="flex items-center gap-1.5 font-mono">
                <span>المصادر: {lastSourceFetch ? timeAgo(lastSourceFetch) : 'محدثة'}</span>
                <span className="text-slate-300">|</span>
                <span>المعالجة: {lastDataProcessing ? timeAgo(lastDataProcessing) : 'جاهزة'}</span>
              </div>
            </div>

            {/* Refresh Button */}
            <button
              onClick={() => triggerGlobalRefresh()}
              disabled={loading || isRefreshing}
              title={`تحديث موحد للبيانات (آخر فحص للواجهة: ${lastUIRefresh ? lastUIRefresh.toLocaleTimeString('ar-SA') : 'الآن'})`}
              className="flex items-center justify-center p-2 bg-white hover:bg-slate-50 border border-[#E7ECF3] text-[#10213D] rounded-xl text-xs font-semibold transition shadow-xs cursor-pointer disabled:opacity-60"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading || isRefreshing ? 'animate-spin text-blue-600' : ''}`} />
            </button>

            {/* Reset Filters button if modified */}
            {isFiltered && (
              <button
                onClick={handleResetFilters}
                title="إعادة ضبط الفلاتر إلى النطاق الافتراضي"
                className="flex-1 sm:flex-initial flex items-center justify-center gap-1.5 px-3 py-2 bg-slate-100 hover:bg-slate-200 text-[#64748B] hover:text-[#10213D] rounded-xl text-xs font-semibold transition cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>إعادة ضبط</span>
              </button>
            )}

            {/* Export Button */}
            <button
              onClick={handleExport}
              className="flex-1 sm:flex-initial flex items-center justify-center gap-2 px-4 py-2 bg-[#102544] hover:bg-[#1a3860] text-white rounded-xl text-xs font-semibold transition shadow-xs cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>تصدير</span>
            </button>
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------------- */}
      {/* 2. Top Row: (Left: Hero Insight | Right: Geographic Map)            */}
      {/* ------------------------------------------------------------------- */}
      {/* Insufficient Evidence State Alert (Phase 4 Adaptive UI) */}
      {data?.data_state === 'insufficient_evidence' && (
        <div dir="rtl" className="p-4 bg-amber-50 border border-amber-200 rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-amber-900 animate-in fade-in duration-200">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0" />
            <div>
              <span className="font-bold block text-sm text-[#10213D]">حالة محدودية الأدلة — لا توجد حوادث مسجلة بالمعايير الحالية</span>
              <span className="text-[#64748B] text-xs">
                لم يتم رصد حوادث احتيال موثقة تطابق النطاق المحدد ({geoFilter} • {datePreset === 'custom' ? `${dateFrom} إلى ${dateTo}` : `آخر ${datePreset} يوماً`} • {fraudTypeFilter !== 'all' ? fraudTypeFilter : 'جميع الأنواع'}). لا يتم توليد أنماط أو استنتاجات تقديرية دون أدلة كافية.
              </span>
            </div>
          </div>
          <button
            onClick={handleResetFilters}
            className="shrink-0 px-3.5 py-1.5 bg-[#102544] text-white rounded-xl text-xs font-semibold hover:bg-[#1a3860] transition"
          >
            إعادة ضبط الفلاتر
          </button>
        </div>
      )}

      {/* X Social Intelligence Integration Banner */}
      <div
        dir="rtl"
        className="bg-white border border-[#E7ECF3] rounded-2xl p-3.5 px-4.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-xs"
      >
        <div className="flex items-center gap-3">
          <span className="w-8 h-8 rounded-xl bg-black text-white flex items-center justify-center font-black text-sm shrink-0 shadow-xs">
            𝕏
          </span>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-xs sm:text-sm text-[#10213D]">
                رصد المنشورات العامة والتحذيرات المجتمعية (منصة 𝕏)
              </span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-sky-50 text-sky-700 border border-sky-200">
                14 تحذيراً وتوعية نشطة
              </span>
            </div>
            <p className="text-[11px] text-[#64748B] mt-0.5">
              متابعة حساب د. خالد المزيني (@K_H_AlMuzaini) لتصنيف التهديدات دون اختلاق وقائع وبشفافية كاملة
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <a
            href="/explore?source=x_k_h_almuzaini"
            className="px-3 py-1.5 rounded-xl bg-stone-900 hover:bg-black text-white text-xs font-semibold flex items-center gap-1.5 transition shadow-xs"
          >
            <span>استعراض المنشورات في Explore</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
          <a
            href="/system"
            className="px-3 py-1.5 rounded-xl border border-stone-200 hover:bg-slate-50 text-stone-700 text-xs font-semibold transition"
          >
            <span>إعدادات المصدر</span>
          </a>
        </div>
      </div>

      {/* ------------------------------------------------------------------- */}
      {/* 2. Top Row: Hero Intelligence Insight (Full-Width Executive Banner) */}
      {/* ------------------------------------------------------------------- */}
      {hero && (
        <div
          dir="rtl"
          className="relative overflow-hidden bg-white border border-[#E7ECF3] rounded-3xl p-6 md:p-7 shadow-xs flex flex-col justify-between"
        >
          {/* Background Riyadh Skyline: Only shown when Saudi is selected and insight is not a generic neutral summary */}
          {(geoFilter === 'السعودية' || geoFilter === 'SA') && !hero.is_neutral && (
            <div
              className="absolute inset-y-0 left-0 w-[45%] bg-cover bg-center pointer-events-none opacity-85 hidden sm:block"
              style={{
                backgroundImage: 'url(/riyadh_skyline.jpg)',
                maskImage: 'linear-gradient(to right, rgba(0,0,0,1) 0%, rgba(0,0,0,0.5) 60%, rgba(0,0,0,0) 100%)',
                WebkitMaskImage: 'linear-gradient(to right, rgba(0,0,0,1) 0%, rgba(0,0,0,0.5) 60%, rgba(0,0,0,0) 100%)',
              }}
            />
          )}

          {/* Top Content: Badge, Headline, Summary */}
          <div className="relative z-10 space-y-3.5 max-w-4xl text-right">
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

          {/* Bottom Row: 3 Metrics on the RIGHT, Action Button on the LEFT */}
          <div dir="rtl" className="relative z-10 pt-4 mt-2 flex flex-wrap items-center justify-between gap-4 border-t border-slate-100/80">
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
                  <span>{hero.growth_note || hero.growth_pct}</span>
                </div>
                <span className="text-[11px] text-[#64748B]">مقارنة بالشهر السابق</span>
              </div>
            </div>

            {/* Action Button on LEFT */}
            <button
              onClick={() =>
                openDrilldown(`الحوادث الداعمة للاتجاه: ${hero.headline}`, {
                  fraud_type: hero.fraud_type !== 'all' ? hero.fraud_type : undefined,
                  incident_ids: hero.incident_ids && hero.incident_ids.length > 0 ? hero.incident_ids.join(',') : undefined,
                  country: geoFilter === 'السعودية' ? 'SA' : undefined,
                })
              }
              className="flex items-center gap-2 px-4 py-2 bg-[#102544] hover:bg-[#1a3860] text-white rounded-xl text-xs font-bold transition shadow-xs cursor-pointer"
            >
              <ChevronRight className="w-4 h-4" />
              <span>عرض الحوادث</span>
            </button>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------- */}
      {/* 3. Documented Incidents Map (Full-width matching user's exact design) */}
      {/* ------------------------------------------------------------------- */}
      <div
        dir="rtl"
        className="w-full bg-white border border-[#E7ECF3] rounded-3xl p-6 md:p-8 shadow-xs relative overflow-hidden space-y-4"
      >
        {/* Header: Controls on Left, Title & Subtitle on Right */}
        <div dir="rtl" className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-[#F1F5F9]">
          {/* Right: Title & Subtitle */}
          <div className="space-y-1 text-right">
            <div className="flex items-center gap-2">
              <h3 className="text-base md:text-lg font-extrabold text-[#10213D]">
                خريطة الحوادث الموثقة
              </h3>
              {data?.total_scope_incidents !== undefined && (
                <span
                  role="button"
                  tabIndex={0}
                  onClick={() => openDrilldown('جميع حوادث النطاق الموثقة', {
                    country: geoFilter === 'السعودية' ? 'SA' : undefined,
                    incident_ids: data?.incident_ids && data.incident_ids.length > 0 ? data.incident_ids.join(',') : undefined,
                  })}
                  className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-blue-50 text-blue-700 border border-blue-200/60 hover:bg-blue-100 transition cursor-pointer"
                  title="اضغط لعرض كافة الحوادث المندرجة ضمن الإجمالي"
                >
                  {data.total_scope_incidents} حادثة موثقة
                </span>
              )}
              <div className="relative group">
                <Info className="w-4 h-4 text-[#64748B] hover:text-[#10213D] cursor-pointer transition" />
                <div className="hidden group-hover:block absolute right-0 top-full mt-1.5 w-64 p-2.5 bg-slate-900/95 text-white text-[11px] rounded-xl shadow-xl z-50 pointer-events-none leading-relaxed">
                  توضح التوزيع الجغرافي للحوادث المرصودة في مصادرنا، وليست إحصائية رسمية لمعدلات الاحتيال.
                </div>
              </div>
            </div>
            <p className="text-xs text-[#64748B]">
              توضح التوزيع الجغرافي للحوادث المرصودة في مصادرنا، وليست إحصائية رسمية لمعدلات الاحتيال.
              {mapScope !== 'world' && Boolean(data?.unknown_location_count) ? ` (يشمل الإجمالي ${data?.unknown_location_count} حادثة داخل المملكة غير محددة المدينة بدقة).` : ''}
            </p>
          </div>

          {/* Left: Segmented Pill Toggle [المدن] [المناطق] [العالم] */}
          <div dir="ltr" className="flex items-center bg-[#F8FAFC] border border-[#E2E8F0] p-1 rounded-xl self-start sm:self-auto shadow-2xs">
            <button
              onClick={() => setMapScope('world')}
              className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-bold transition cursor-pointer ${
                mapScope === 'world'
                  ? 'bg-[#0D1F3C] text-white shadow-xs'
                  : 'text-[#64748B] hover:text-[#10213D]'
              }`}
            >
              <Globe className="w-3.5 h-3.5" />
              <span>العالم</span>
            </button>
            <button
              onClick={() => setMapScope('regions')}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold transition cursor-pointer ${
                mapScope === 'regions'
                  ? 'bg-[#0D1F3C] text-white shadow-xs'
                  : 'text-[#64748B] hover:text-[#10213D]'
              }`}
            >
              المناطق
            </button>
            <button
              onClick={() => setMapScope('cities')}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold transition cursor-pointer ${
                mapScope === 'cities'
                  ? 'bg-[#0D1F3C] text-white shadow-xs'
                  : 'text-[#64748B] hover:text-[#10213D]'
              }`}
            >
              المدن
            </button>
          </div>
        </div>

        {/* Map Canvas with Left Controls & Right Floating Panel */}
        <div className="relative w-full h-[420px] md:h-[480px] lg:h-[500px] rounded-2xl bg-[#FAFCFF] border border-[#F1F5F9] flex items-center justify-center overflow-hidden select-none">
          {/* Zoom/Pan Scaled Layer */}
          <div
            className="relative w-full h-full transition-transform duration-300 ease-out origin-center flex items-center justify-center"
            style={{ transform: `scale(${zoomLevel})` }}
          >
            {mapScope === 'world' ? (
              <>
                {/* SVG World Map */}
                <WorldMapVector />

                {/* Ocean Names in Arabic matching screenshot */}
                <div className="absolute top-[60%] left-[8%] text-[#94A3B8] text-xs md:text-sm font-semibold pointer-events-none select-none">
                  المحيط الهادئ
                </div>
                <div className="absolute top-[72%] left-[33%] text-[#94A3B8] text-xs md:text-sm font-semibold pointer-events-none select-none">
                  المحيط الأطلسي
                </div>
                <div className="absolute top-[74%] left-[53%] text-[#94A3B8] text-xs md:text-sm font-semibold pointer-events-none select-none">
                  المحيط الهندي
                </div>

                {/* Data Hotspot Pins dynamically matching active query */}
                {worldHotspots.map((h) => {
                  const onClickHandler = () =>
                    openDrilldown(`الحوادث المسجلة في: ${h.name}`, {
                      country: h.country,
                      incident_ids: h.incident_ids && h.incident_ids.length > 0 ? h.incident_ids.join(',') : undefined,
                    });

                  if (h.type === 'red_major') {
                    return (
                      <div
                        key={h.id}
                        style={{ left: `${h.x}%`, top: `${h.y}%` }}
                        onClick={onClickHandler}
                        className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-20"
                      >
                        <div className="relative flex items-center justify-center">
                          {/* Pulsing Halos */}
                          <span className="animate-ping absolute inline-flex h-12 w-12 rounded-full bg-red-400 opacity-25" />
                          <span className="absolute inline-flex h-14 w-14 rounded-full bg-red-500/20" />
                          <div className="relative w-8 h-8 rounded-full bg-[#EF4444] text-white font-black text-xs flex items-center justify-center shadow-lg border-2 border-white group-hover:scale-110 transition-transform">
                            {h.label}
                          </div>
                          {/* Tooltip */}
                          <div className="hidden group-hover:block absolute bottom-full mb-2 right-1/2 translate-x-1/2 p-2 bg-slate-900/95 text-white rounded-lg shadow-lg text-[10px] whitespace-nowrap z-50 pointer-events-none text-right">
                            <div className="font-bold text-red-300">{h.name} ({h.count} حوادث موثقة)</div>
                            <div className="text-[9px] text-slate-300 mt-0.5">اضغط لعرض تفاصيل الحوادث والأدلة ←</div>
                          </div>
                        </div>
                      </div>
                    );
                  }

                  if (h.type === 'blue_major') {
                    return (
                      <div
                        key={h.id}
                        style={{ left: `${h.x}%`, top: `${h.y}%` }}
                        onClick={onClickHandler}
                        className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-20"
                      >
                        <div className="relative flex items-center justify-center">
                          <span className="absolute inline-flex h-16 w-16 rounded-full bg-blue-500/20" />
                          <div className="relative w-8 h-8 rounded-full bg-[#3B82F6] text-white font-black text-xs flex items-center justify-center shadow-lg border-2 border-white group-hover:scale-110 transition-transform">
                            {h.label}
                          </div>
                          <div className="hidden group-hover:block absolute bottom-full mb-2 right-1/2 translate-x-1/2 p-2 bg-slate-900/95 text-white rounded-lg shadow-lg text-[10px] whitespace-nowrap z-50 pointer-events-none text-right">
                            <div className="font-bold text-sky-300">{h.name} ({h.count} حوادث موثقة)</div>
                            <div className="text-[9px] text-slate-300 mt-0.5">اضغط لعرض تفاصيل الحوادث والأدلة ←</div>
                          </div>
                        </div>
                      </div>
                    );
                  }

                  if (h.type === 'orange_square') {
                    return (
                      <div
                        key={h.id}
                        style={{ left: `${h.x}%`, top: `${h.y}%` }}
                        onClick={onClickHandler}
                        className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-20"
                      >
                        <div className="relative flex items-center justify-center">
                          <span className="absolute inline-flex h-10 w-10 rounded-full bg-amber-500/20" />
                          <div className="relative w-6 h-6 rounded-md rotate-45 bg-[#F97316] text-white font-black text-xs flex items-center justify-center shadow-md border-2 border-white group-hover:scale-110 transition-transform">
                            <span className="-rotate-45">{h.label}</span>
                          </div>
                          <div className="hidden group-hover:block absolute bottom-full mb-2 right-1/2 translate-x-1/2 p-2 bg-slate-900/95 text-white rounded-lg shadow-lg text-[10px] whitespace-nowrap z-50 pointer-events-none text-right">
                            <div className="font-bold text-amber-300">{h.name} ({h.count} حوادث موثقة)</div>
                            <div className="text-[9px] text-slate-300 mt-0.5">اضغط لعرض تفاصيل الحوادث والأدلة ←</div>
                          </div>
                        </div>
                      </div>
                    );
                  }

                  if (h.type === 'yellow_circle') {
                    return (
                      <div
                        key={h.id}
                        style={{ left: `${h.x}%`, top: `${h.y}%` }}
                        onClick={onClickHandler}
                        className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-20"
                      >
                        <div className="relative flex items-center justify-center">
                          <span className="absolute inline-flex h-9 w-9 rounded-full bg-amber-400/20" />
                          <div className="relative w-6 h-6 rounded-full bg-[#FBBF24] text-white font-black text-xs flex items-center justify-center shadow-md border-2 border-white group-hover:scale-110 transition-transform">
                            {h.label}
                          </div>
                          <div className="hidden group-hover:block absolute bottom-full mb-2 right-1/2 translate-x-1/2 p-2 bg-slate-900/95 text-white rounded-lg shadow-lg text-[10px] whitespace-nowrap z-50 pointer-events-none text-right">
                            <div className="font-bold text-amber-300">{h.name} ({h.count} حوادث موثقة)</div>
                            <div className="text-[9px] text-slate-300 mt-0.5">اضغط لعرض تفاصيل الحوادث والأدلة ←</div>
                          </div>
                        </div>
                      </div>
                    );
                  }

                  // Default blue_circle (India, Europe East, Australia)
                  return (
                    <div
                      key={h.id}
                      style={{ left: `${h.x}%`, top: `${h.y}%` }}
                      onClick={onClickHandler}
                      className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-20"
                    >
                      <div className="relative flex items-center justify-center">
                        <span className="absolute inline-flex h-9 w-9 rounded-full bg-blue-500/20" />
                        <div className="relative w-6 h-6 rounded-full bg-[#3B82F6] text-white font-black text-xs flex items-center justify-center shadow-md border-2 border-white group-hover:scale-110 transition-transform">
                          {h.label}
                        </div>
                        <div className="hidden group-hover:block absolute bottom-full mb-2 right-1/2 translate-x-1/2 p-2 bg-slate-900/95 text-white rounded-lg shadow-lg text-[10px] whitespace-nowrap z-50 pointer-events-none text-right">
                          <div className="font-bold text-sky-300">{h.name} ({h.count} حوادث موثقة)</div>
                          <div className="text-[9px] text-slate-300 mt-0.5">اضغط لعرض تفاصيل الحوادث والأدلة ←</div>
                        </div>
                      </div>
                    </div>
                  );
                })}

                {/* Minor Incident Dots */}
                {MINOR_DOTS.map((d) => (
                  <div
                    key={d.id}
                    style={{ left: `${d.x}%`, top: `${d.y}%` }}
                    onClick={() => openDrilldown(`الحوادث المسجلة في: ${d.name}`, { country: d.country })}
                    className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-10"
                  >
                    <div className="relative flex items-center justify-center">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#3B82F6] border border-white shadow-xs group-hover:scale-125 transition-transform" />
                      <div className="hidden group-hover:block absolute bottom-full mb-1.5 right-1/2 translate-x-1/2 p-1.5 bg-slate-900/95 text-white rounded-md text-[9px] whitespace-nowrap z-50 pointer-events-none text-right">
                        {d.name}
                      </div>
                    </div>
                  </div>
                ))}
              </>
            ) : (
              /* Saudi Regional/City Map View */
              <div className="relative w-full h-full flex items-center justify-center">
                <svg
                  viewBox="0 0 500 400"
                  className="w-full h-full text-slate-100 select-none drop-shadow-xs"
                  style={{ maxHeight: '350px' }}
                >
                  <defs>
                    <linearGradient id="saudiGradFull" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#F8FAFC" />
                      <stop offset="100%" stopColor="#E2E8F0" />
                    </linearGradient>
                  </defs>
                  <path
                    d="M 90 80 L 140 70 L 200 65 L 270 85 L 340 100 L 390 120 L 440 135 L 460 160 L 420 200 L 435 240 L 400 270 L 360 300 L 310 325 L 250 330 L 210 300 L 180 270 L 140 230 L 110 180 L 80 130 Z"
                    fill="url(#saudiGradFull)"
                    stroke="#CBD5E1"
                    strokeWidth="1.5"
                    strokeLinejoin="round"
                  />
                  <path d="M 140 70 Q 180 140 250 160 T 360 300" fill="none" stroke="#E2E8F0" strokeWidth="1" strokeDasharray="3 3" />
                  <path d="M 110 180 Q 200 200 310 325" fill="none" stroke="#E2E8F0" strokeWidth="1" strokeDasharray="3 3" />
                  <path d="M 270 85 Q 280 180 250 330" fill="none" stroke="#E2E8F0" strokeWidth="1" strokeDasharray="3 3" />
                </svg>

                {/* Saudi Pinpoints */}
                {mapPoints.map((pt) => (
                  <div
                    key={pt.id || pt.name}
                    style={{ top: `${pt.y}%`, left: `${pt.x}%` }}
                    onClick={() => {
                      setSelectedCityPoint(pt);
                      setCityModalOpen(true);
                    }}
                    className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-20"
                  >
                    <div className="relative flex items-center justify-center">
                      {pt.count >= 6 && (
                        <span className="animate-ping absolute inline-flex h-8 w-8 rounded-full bg-red-400 opacity-30" />
                      )}
                      <div
                        style={{
                          backgroundColor: pt.color,
                          width: `${Math.max(22, Math.min(36, 18 + pt.count * 1.5))}px`,
                          height: `${Math.max(22, Math.min(36, 18 + pt.count * 1.5))}px`,
                        }}
                        className="rounded-full text-white font-extrabold text-xs flex items-center justify-center shadow-md border-2 border-white group-hover:scale-110 transition-transform"
                      >
                        {pt.count}
                      </div>
                      <span className="absolute top-full mt-1 px-1.5 py-0.5 rounded bg-white/95 shadow-xs border border-slate-200 text-[10px] font-bold text-[#10213D] whitespace-nowrap opacity-90 group-hover:opacity-100 group-hover:scale-105 transition">
                        {pt.name}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Left Floating Controls (+, -, Crosshair) */}
          <div className="absolute top-6 left-6 flex flex-col gap-1 bg-white/95 backdrop-blur-xs border border-[#E7ECF3] rounded-xl shadow-xs p-1 z-30">
            <button
              onClick={() => setZoomLevel((z) => Math.min(2.5, z + 0.25))}
              title="تكبير الخريطة"
              className="p-1.5 text-[#64748B] hover:text-[#10213D] hover:bg-slate-100 rounded-lg transition cursor-pointer"
            >
              <Plus className="w-4 h-4" />
            </button>
            <button
              onClick={() => setZoomLevel((z) => Math.max(0.75, z - 0.25))}
              title="تصغير الخريطة"
              className="p-1.5 text-[#64748B] hover:text-[#10213D] hover:bg-slate-100 rounded-lg transition cursor-pointer"
            >
              <Minus className="w-4 h-4" />
            </button>
            <div className="h-px bg-slate-200 my-0.5" />
            <button
              onClick={() => setZoomLevel(1)}
              title="إعادة ضبط العرض"
              className="p-1.5 text-[#64748B] hover:text-[#10213D] hover:bg-slate-100 rounded-lg transition cursor-pointer"
            >
              <Crosshair className="w-4 h-4" />
            </button>
          </div>

          {/* Right Floating Analytical Panel matching reference image */}
          <div className="absolute top-4 right-4 bg-white/95 backdrop-blur-md border border-[#E7ECF3] rounded-2xl p-4 shadow-md w-56 md:w-60 z-30 text-right">
            {/* Dropdown Button */}
            <div className="relative mb-3">
              <button className="w-full flex items-center justify-between px-3 py-1.5 bg-slate-50 hover:bg-slate-100 border border-[#E7ECF3] rounded-xl text-xs font-semibold text-[#10213D] transition cursor-pointer">
                <ChevronDown className="w-3.5 h-3.5 text-[#64748B]" />
                <span>عدد الحوادث</span>
              </button>
            </div>
            {/* Color / Incident Count Legend */}
            <div className="space-y-1.5 text-xs pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2 justify-end">
                <span dir="ltr" className="text-xs font-medium text-[#64748B] font-mono">10+</span>
                <span className="w-3 h-3 rounded-full bg-[#EF4444]" />
              </div>
              <div className="flex items-center gap-2 justify-end">
                <span dir="ltr" className="text-xs font-medium text-[#64748B] font-mono">6 - 10</span>
                <span className="w-3 h-3 rounded-full bg-[#F97316]" />
              </div>
              <div className="flex items-center gap-2 justify-end">
                <span dir="ltr" className="text-xs font-medium text-[#64748B] font-mono">3 - 5</span>
                <span className="w-3 h-3 rounded-full bg-[#FBBF24]" />
              </div>
              <div className="flex items-center gap-2 justify-end">
                <span dir="ltr" className="text-xs font-medium text-[#64748B] font-mono">1 - 2</span>
                <span className="w-3 h-3 rounded-full bg-[#3B82F6]" />
              </div>
            </div>

            {/* Section Heading: أبرز الدول (حسب الحوادث) */}
            <div className="mt-3">
              <div className="text-[11px] font-bold text-[#64748B] mb-2 text-right">
                {mapScope === 'world'
                  ? 'أبرز الدول (حسب الحوادث)'
                  : mapScope === 'regions'
                  ? 'أبرز المناطق (حسب الحوادث)'
                  : 'أبرز المدن (حسب الحوادث)'}
              </div>
              <div className="space-y-1.5">
                {mapScope === 'world' ? (
                  topCountriesList.map((c) => (
                    <div
                      key={c.code}
                      id={`btn-country-${c.code.toLowerCase()}`}
                      role="button"
                      tabIndex={0}
                      onClick={() =>
                        openDrilldown(`الحوادث المسجلة في: ${c.name}`, {
                          country: c.code,
                          incident_ids: c.incident_ids && c.incident_ids.length > 0 ? c.incident_ids.join(',') : undefined,
                        })
                      }
                      className="flex items-center justify-between p-1.5 rounded-lg hover:bg-slate-50 cursor-pointer transition select-none"
                    >
                      <span className="text-xs font-extrabold text-[#10213D] font-mono">{c.count}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-[#10213D]">{c.name}</span>
                        <span className="text-base leading-none">{c.flag}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <>
                    {mapPoints.slice(0, 5).map((pt) => (
                      <div
                        key={pt.name}
                        role="button"
                        tabIndex={0}
                        onClick={() => {
                          setSelectedCityPoint(pt);
                          setCityModalOpen(true);
                        }}
                        className="flex items-center justify-between p-1.5 rounded-lg hover:bg-slate-50 cursor-pointer transition select-none"
                      >
                        <span className="text-xs font-extrabold text-[#10213D] font-mono">{pt.count}</span>
                        <span className="text-xs font-semibold text-[#10213D]">{pt.name}</span>
                      </div>
                    ))}
                    {Boolean(data?.unknown_location_count && data.unknown_location_count > 0) && (
                      <div
                        role="button"
                        tabIndex={0}
                        onClick={() =>
                          openDrilldown('حوادث نطاق المملكة (دون تحديد مدينة بدقة)', {
                            country: 'SA',
                            incident_ids: data?.unknown_location_incident_ids ? data.unknown_location_incident_ids.join(',') : undefined,
                          })
                        }
                        className="flex items-center justify-between p-1.5 rounded-lg hover:bg-slate-50 cursor-pointer transition select-none border-t border-slate-100 mt-1 pt-1.5"
                        title="حوادث مثبتة داخل المملكة لكن دون تعيين مدينة محددة"
                      >
                        <span className="text-xs font-extrabold text-[#64748B] font-mono">{data?.unknown_location_count}</span>
                        <span className="text-[11px] font-medium text-[#64748B]">عموم المملكة (دون تحديد مدينة)</span>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* Bottom Action: عرض جميع الدول */}
            <button
              id="btn-view-all-countries"
              onClick={() =>
                openDrilldown(mapScope === 'world' ? 'جميع الحوادث المرصودة عالمياً' : 'جميع حوادث النطاق الموثقة', {
                  country: geoFilter === 'السعودية' ? 'SA' : undefined,
                  incident_ids: data?.incident_ids && data.incident_ids.length > 0 ? data.incident_ids.join(',') : undefined,
                })
              }
              className="w-full mt-3 py-1.5 px-3 bg-slate-50 hover:bg-slate-100 border border-slate-200/60 rounded-xl text-xs font-semibold text-[#10213D] flex items-center justify-between transition cursor-pointer"
            >
              <ChevronLeft className="w-3.5 h-3.5 text-[#64748B]" />
              <span>{mapScope === 'world' ? 'عرض جميع الدول' : 'عرض جميع حوادث النطاق'}</span>
            </button>
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
            <span className="text-[11px] text-[#64748B]">مقارنة بفترة سابقة مماثلة</span>
          </div>

          <div className="space-y-3">
            {patterns.map((p, idx) => (
              <div
                key={p.key}
                onClick={() =>
                  openDrilldown(`الحوادث المصنفة كـ: ${p.name_ar}`, {
                    fraud_type: p.key,
                    incident_ids: p.incident_ids && p.incident_ids.length > 0 ? p.incident_ids.join(',') : undefined,
                    country: geoFilter === 'السعودية' ? 'SA' : undefined,
                  })
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
                <div
                  className="flex items-center gap-2"
                  title={p.calculation_tooltip || 'معدل التغير المحسوب بمقارنة الفترة الحالية بالسابقة المماثلة'}
                >
                  <span
                    className={`text-xs font-bold flex items-center gap-0.5 ${
                      p.direction === 'up' ? 'text-[#EF4444]' : 'text-[#10B981]'
                    }`}
                  >
                    {p.direction === 'up' ? '↑' : '↓'}
                    {p.change_label ? p.change_label : (p.change_pct && p.change_pct > 0 ? `+${p.change_pct}%` : `${p.change_pct}%`)}
                  </span>

                  {/* Sparkline */}
                  <div className="flex items-end gap-0.5 h-4 w-7">
                    {p.sparkline.map((val, sIdx) => (
                      <div
                        key={sIdx}
                        style={{ height: `${Math.max(2, (val / 7) * 16)}px` }}
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

        {/* Column 2 (Middle): كيف تحدث عمليات الاحتيال؟ (Dual View: Sankey vs Observed Mechanisms) (~42%) */}
        {flow && (() => {
          const isNodeConnected = (nodeId: string) => {
            if (!hoveredNode) return false;
            if (hoveredNode === nodeId) return true;
            const isDirect = flow.flows.some(
              (f) =>
                (f.from === hoveredNode && f.to === nodeId) ||
                (f.to === hoveredNode && f.from === nodeId)
            );
            if (isDirect) return true;
            // 2-hop check: channel -> typology -> sector
            const intermediateTypes = flow.flows
              .filter((f) => f.from === hoveredNode)
              .map((f) => f.to);
            if (intermediateTypes.includes(nodeId)) return true;
            const is2Hop = flow.flows.some(
              (f) => intermediateTypes.includes(f.from) && f.to === nodeId
            );
            if (is2Hop) return true;
            const incomingTypes = flow.flows
              .filter((f) => f.to === hoveredNode)
              .map((f) => f.from);
            if (incomingTypes.includes(nodeId)) return true;
            const is2HopReverse = flow.flows.some(
              (f) => incomingTypes.includes(f.to) && f.from === nodeId
            );
            return is2HopReverse;
          };

          const isFlowHighlighted = (from: string, to: string) => {
            if (!hoveredNode) return false;
            if (hoveredNode === from || hoveredNode === to) return true;
            const fromChannel = flow.channels.some((c) => c.id === hoveredNode);
            if (fromChannel) {
              const intermediateTypes = flow.flows
                .filter((f) => f.from === hoveredNode)
                .map((f) => f.to);
              if (intermediateTypes.includes(from)) return true;
            }
            const toSector = flow.sectors.some((s) => s.id === hoveredNode);
            if (toSector) {
              const intermediateTypes = flow.flows
                .filter((f) => f.to === hoveredNode)
                .map((f) => f.from);
              if (intermediateTypes.includes(to)) return true;
            }
            return false;
          };

          return (
            <div
              dir="rtl"
              className="lg:col-span-5 bg-white border border-[#E7ECF3] rounded-2xl p-5 md:p-6 shadow-xs space-y-3 flex flex-col justify-between"
            >
              {/* Header with Title matching reference screenshot */}
              <div className="flex items-center justify-center pb-2 border-b border-[#F1F5F9]">
                <h3 className="text-sm md:text-base font-bold text-[#10213D]">
                  كيف تحدث عمليات الاحتيال؟
                </h3>
              </div>

              {flowViewType === 'sankey' ? (
                /* 3 Columns Layout: Right = Channels, Center = Typologies, Left = Sectors (RTL Flow) */
                <div dir="rtl" className="relative pt-1 min-h-[270px]">
                  {/* Dynamic SVG Alluvial Ribbon Overlay connecting Col 1 (Right) -> Col 2 (Mid) -> Col 3 (Left) */}
                  <svg
                    className="absolute inset-0 w-full h-full pointer-events-none z-0"
                    preserveAspectRatio="none"
                    viewBox="0 0 300 240"
                  >
                    {flow.flows.map((f, idx) => {
                      const cIdx = flow.channels.findIndex((c) => c.id === f.from);
                      const tIdx = flow.typologies.findIndex((t) => t.id === f.to);
                      const tIdx2 = flow.typologies.findIndex((t) => t.id === f.from);
                      const sIdx = flow.sectors.findIndex((s) => s.id === f.to);

                      let x1 = 0, y1 = 0, x2 = 0, y2 = 0;
                      let strokeColor = '#3B82F6';

                      if (cIdx >= 0 && tIdx >= 0) {
                        x1 = 205;
                        y1 = 26 + cIdx * 34.5;
                        x2 = 185;
                        y2 = 26 + tIdx * 34.5;
                        strokeColor = flow.channels[cIdx]?.color || '#3B82F6';
                      } else if (tIdx2 >= 0 && sIdx >= 0) {
                        x1 = 115;
                        y1 = 26 + tIdx2 * 34.5;
                        x2 = 95;
                        y2 = 26 + sIdx * 34.5;
                        strokeColor = flow.typologies[tIdx2]?.color || '#EF4444';
                      } else {
                        return null;
                      }

                      const highlighted = isFlowHighlighted(f.from, f.to);
                      const dimmed = hoveredNode !== null && !highlighted;

                      return (
                        <path
                          key={`flow-${f.from}-${f.to}-${idx}`}
                          d={`M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`}
                          fill="none"
                          stroke={strokeColor}
                          strokeWidth={highlighted ? Math.max(3, f.pct / 3) : Math.max(1.2, f.pct / 6)}
                          strokeOpacity={highlighted ? 0.95 : dimmed ? 0.05 : 0.35}
                          strokeLinecap="round"
                          className="transition-all duration-300"
                        />
                      );
                    })}
                  </svg>

                  {/* 3 Interactive Node Columns in Arabic Flow: Channels -> Types -> Sectors */}
                  <div className="relative z-10 grid grid-cols-3 gap-3 text-xs">
                    {/* Column 1 (RIGHT): القنوات المستخدمة */}
                    <div className="space-y-2">
                      <div className="font-bold text-[11px] text-[#64748B] pb-1 text-center">
                        القنوات المستخدمة
                      </div>
                      <div className="space-y-1.5">
                        {flow.channels.map((c) => {
                          const isHovered = hoveredNode === c.id;
                          const isConnected = isNodeConnected(c.id);
                          const isDimmed = hoveredNode !== null && !isHovered && !isConnected;

                          return (
                            <div
                              key={c.id}
                              title={c.pct_basis || `${c.count} حوادث من أصل ${data?.total_scope_incidents}`}
                              onMouseEnter={() => setHoveredNode(c.id)}
                              onMouseLeave={() => setHoveredNode(null)}
                              onClick={() =>
                                openDrilldown(`الحوادث المنفذة عبر: ${c.name_ar}`, {
                                  channel: c.id,
                                  country: geoFilter === 'السعودية' ? 'SA' : undefined,
                                })
                              }
                              className={`p-1.5 rounded-lg border transition-all duration-200 flex items-center justify-between cursor-pointer text-[11px] ${
                                isHovered
                                  ? 'ring-2 ring-blue-500 bg-blue-50/90 border-blue-400 font-bold scale-[1.02] shadow-sm z-20'
                                  : isConnected
                                  ? 'ring-1 ring-blue-300 bg-slate-50 border-blue-300 font-semibold z-10'
                                  : isDimmed
                                  ? 'opacity-40 bg-white/70 border-slate-200'
                                  : 'bg-white hover:bg-slate-50 border-slate-200/80 shadow-xs'
                              }`}
                            >
                              <div className="flex items-center gap-1.5 truncate">
                                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: c.color }} />
                                <span className="font-medium text-[#10213D] truncate">{c.name_ar}</span>
                              </div>
                              <span className="font-bold text-[#64748B] text-[10px]">{c.percentage}%</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Column 2 (MIDDLE): نوع الاحتيال */}
                    <div className="space-y-2">
                      <div className="font-bold text-[11px] text-[#64748B] pb-1 text-center">
                        نوع الاحتيال
                      </div>
                      <div className="space-y-1.5">
                        {flow.typologies.map((t) => {
                          const isHovered = hoveredNode === t.id;
                          const isConnected = isNodeConnected(t.id);
                          const isDimmed = hoveredNode !== null && !isHovered && !isConnected;

                          return (
                            <div
                              key={t.id}
                              title={t.pct_basis || `${t.count} حوادث من أصل ${data?.total_scope_incidents}`}
                              onMouseEnter={() => setHoveredNode(t.id)}
                              onMouseLeave={() => setHoveredNode(null)}
                              onClick={() =>
                                openDrilldown(`حوادث نوع: ${t.name_ar}`, {
                                  fraud_type: t.id,
                                  country: geoFilter === 'السعودية' ? 'SA' : undefined,
                                })
                              }
                              className={`p-1.5 rounded-lg border transition-all duration-200 flex items-center justify-between cursor-pointer text-[11px] ${
                                isHovered
                                  ? 'ring-2 ring-blue-500 bg-blue-50/90 border-blue-400 font-bold scale-[1.02] shadow-sm z-20'
                                  : isConnected
                                  ? 'ring-1 ring-blue-300 bg-slate-50 border-blue-300 font-semibold z-10'
                                  : isDimmed
                                  ? 'opacity-40 bg-white/70 border-slate-200'
                                  : 'bg-white hover:bg-slate-50 border-slate-200/80 shadow-xs'
                              }`}
                            >
                              <div className="flex items-center gap-1.5 truncate">
                                <span className="w-1.5 h-3.5 rounded-xs shrink-0" style={{ backgroundColor: t.color }} />
                                <span className="font-medium text-[#10213D] truncate">{t.name_ar}</span>
                              </div>
                              <span className="font-bold text-[#64748B] text-[10px]">{t.percentage}%</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Column 3 (LEFT): الجهات أو القطاعات المستهدفة */}
                    <div className="space-y-2">
                      <div className="font-bold text-[11px] text-[#64748B] pb-1 text-center">
                        الجهات أو القطاعات المستهدفة
                      </div>
                      <div className="space-y-1.5">
                        {flow.sectors.map((s) => {
                          const isHovered = hoveredNode === s.id;
                          const isConnected = isNodeConnected(s.id);
                          const isDimmed = hoveredNode !== null && !isHovered && !isConnected;

                          return (
                            <div
                              key={s.id}
                              title={s.pct_basis || `${s.count} حوادث من أصل ${data?.total_scope_incidents}`}
                              onMouseEnter={() => setHoveredNode(s.id)}
                              onMouseLeave={() => setHoveredNode(null)}
                              onClick={() =>
                                openDrilldown(`الحوادث المستهدفة لقطاع: ${s.name_ar}`, {
                                  sector: s.id,
                                  country: geoFilter === 'السعودية' ? 'SA' : undefined,
                                })
                              }
                              className={`p-1.5 rounded-lg border transition-all duration-200 flex items-center justify-between cursor-pointer text-[11px] ${
                                isHovered
                                  ? 'ring-2 ring-blue-500 bg-blue-50/90 border-blue-400 font-bold scale-[1.02] shadow-sm z-20'
                                  : isConnected
                                  ? 'ring-1 ring-blue-300 bg-slate-50 border-blue-300 font-semibold z-10'
                                  : isDimmed
                                  ? 'opacity-40 bg-white/70 border-slate-200'
                                  : 'bg-white hover:bg-slate-50 border-slate-200/80 shadow-xs'
                              }`}
                            >
                              <div className="flex items-center gap-1.5 truncate">
                                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: s.color }} />
                                <span className="font-medium text-[#10213D] truncate">{s.name_ar}</span>
                              </div>
                              <span className="font-bold text-[#64748B] text-[10px]">{s.percentage}%</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* Phase 6 Alternative View: Observed Fraud Mechanisms (Direct Incident Evidence) */
                <div className="space-y-2 max-h-[300px] overflow-y-auto p-1">
                  {flow.observed_mechanisms && flow.observed_mechanisms.length > 0 ? (
                    flow.observed_mechanisms.map((m) => (
                      <div
                        key={m.incident_id}
                        className="p-3 rounded-xl border border-slate-200/80 bg-slate-50/60 hover:bg-white hover:border-blue-300 transition space-y-2 text-xs"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="px-2 py-0.5 rounded-md bg-blue-100 text-blue-800 text-[10px] font-bold">
                              القناة: {m.channel_ar}
                            </span>
                            <span className="px-2 py-0.5 rounded-md bg-red-100 text-red-800 text-[10px] font-bold">
                              النمط: {m.fraud_type_ar}
                            </span>
                            <span className="px-2 py-0.5 rounded-md bg-amber-100 text-amber-900 text-[10px] font-bold">
                              القطاع: {m.sector_ar}
                            </span>
                          </div>
                          <span className="text-[10px] font-semibold text-slate-500">
                            {m.incident_ref}
                          </span>
                        </div>

                        <div className="font-bold text-[#10213D]">
                          {m.title}
                        </div>

                        {m.evidence_text && (
                          <div className="text-[11px] text-slate-600 bg-white p-2 rounded-lg border-r-2 border-blue-400 italic">
                            "{m.evidence_text}"
                          </div>
                        )}

                        <div className="flex justify-end pt-1">
                          <button
                            onClick={() => viewSingleIncident(m.incident_id)}
                            className="px-2.5 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg text-[11px] font-semibold transition"
                          >
                            عرض الحادثة والأدلة المرجعية ←
                          </button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="p-6 text-center text-xs text-[#64748B]">
                      لا توجد آليات احتيال مسجلة تطابق هذه المعايير.
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })()}

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

      {/* ------------------------------------------------------------------- */}
      {/* 11. City / Region Intelligence Inspector Modal (Section 5 Spec)      */}
      {/* ------------------------------------------------------------------- */}
      {cityModalOpen && selectedCityPoint && (
        <div
          dir="rtl"
          className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4"
        >
          <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[85vh] shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
            {/* Modal Header */}
            <div className="p-5 border-b border-[#E7ECF3] flex items-center justify-between bg-slate-50/70">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center text-blue-600">
                  <MapPin className="w-5 h-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-extrabold text-base text-[#10213D]">
                      ميتا فيو الموقع: {selectedCityPoint.name}
                    </h3>
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-[#ECFDF5] text-[#10B981]">
                      موثقة بالأدلة
                    </span>
                  </div>
                  <p className="text-xs text-[#64748B] mt-0.5">
                    {selectedCityPoint.region} • {geoFilter === 'العالم' ? 'رصد دولي' : 'المملكة العربية السعودية'}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setCityModalOpen(false)}
                className="p-2 hover:bg-slate-200/60 rounded-xl text-slate-400 hover:text-slate-600 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-5 overflow-y-auto space-y-4">
              {/* 3 Metric Badges */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="p-3 rounded-xl bg-[#F8FAFC] border border-[#E7ECF3] text-right space-y-1">
                  <span className="text-[11px] text-[#64748B]">عدد الحوادث الموثقة</span>
                  <div className="text-xl font-black text-[#10213D]">{selectedCityPoint.count}</div>
                </div>

                <div className="p-3 rounded-xl bg-[#F8FAFC] border border-[#E7ECF3] text-right space-y-1">
                  <span className="text-[11px] text-[#64748B]">أكثر أنواع الاحتيال</span>
                  <div className="text-xs font-bold text-[#10213D] truncate">
                    {selectedCityPoint.top_fraud_types && selectedCityPoint.top_fraud_types.length > 0
                      ? selectedCityPoint.top_fraud_types[0].type_ar
                      : 'غير محدد'}
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[#F8FAFC] border border-[#E7ECF3] text-right space-y-1">
                  <span className="text-[11px] text-[#64748B]">القنوات المرتبطة</span>
                  <div className="text-xs font-bold text-[#10213D] truncate">
                    {selectedCityPoint.top_channels && selectedCityPoint.top_channels.length > 0
                      ? selectedCityPoint.top_channels[0].channel_ar
                      : 'قنوات متعددة'}
                  </div>
                </div>
              </div>

              {/* Documented Incidents List */}
              <div className="space-y-2 pt-2">
                <div className="font-bold text-xs text-[#10213D]">قائمة الحوادث الموثقة في هذا الموقع:</div>
                {selectedCityPoint.incidents && selectedCityPoint.incidents.length > 0 ? (
                  selectedCityPoint.incidents.map((inc: any) => (
                    <div
                      key={inc.id}
                      className="p-3.5 rounded-xl border border-[#E7ECF3] bg-white hover:border-blue-400 hover:bg-slate-50/50 transition space-y-2"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-1">
                          <h4 className="font-bold text-xs text-[#10213D] leading-snug">
                            {inc.title}
                          </h4>
                          <div className="flex flex-wrap items-center gap-2 text-[10.5px] text-[#64748B]">
                            <span>{inc.occurred_at ? inc.occurred_at.slice(0, 10) : 'تاريخ موثق'}</span>
                            <span>•</span>
                            <span className="text-blue-600 font-medium">{inc.fraud_type_ar || 'احتيال مالي'}</span>
                            <span>•</span>
                            <span>{inc.channel_ar || 'قناة الرصد'}</span>
                          </div>
                        </div>
                        <button
                          onClick={() => {
                            setCityModalOpen(false);
                            viewSingleIncident(inc.id);
                          }}
                          className="shrink-0 px-2.5 py-1 rounded-lg bg-blue-50 hover:bg-blue-100 text-blue-700 text-[11px] font-semibold transition"
                        >
                          عرض الأدلة والمصدر ←
                        </button>
                      </div>
                      {inc.evidence_text && (
                        <div className="text-[11px] text-slate-600 bg-slate-50 p-2 rounded-lg border-r-2 border-blue-400 italic">
                          "{inc.evidence_text}"
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="p-4 text-center text-xs text-[#64748B] bg-slate-50 rounded-xl">
                    الحوادث المرتبطة مسجلة وموثقة في سجل المنصة. اضغط أدناه لعرض الحوادث في محرك الاستكشاف.
                  </div>
                )}
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-[#E7ECF3] bg-slate-50/50 flex justify-between items-center">
              <button
                onClick={() => {
                  setCityModalOpen(false);
                  openDrilldown(`الحوادث الموثقة في: ${selectedCityPoint.name}`, {
                    city: selectedCityPoint.name,
                    incident_ids: selectedCityPoint.incident_ids ? selectedCityPoint.incident_ids.join(',') : undefined,
                  });
                }}
                className="px-3.5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl text-xs font-semibold transition"
              >
                فتح السجل الكامل للحوادث ←
              </button>
              <button
                onClick={() => setCityModalOpen(false)}
                className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-[#10213D] rounded-xl text-xs font-semibold transition"
              >
                إغلاق
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
