// src/components/Layout.tsx
import React, { useState, useEffect } from 'react';
import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Compass,
  MapPin,
  Activity,
  Database,
  Radio,
  Sparkles,
  RefreshCw,
  Search,
  Menu,
  X,
  Server,
  ShieldCheck,
  Zap,
  ShieldAlert,
  FolderSearch,
  Languages,
  Network,
  ListChecks,
  HeartPulse,
  Globe2,
  Gauge,
} from 'lucide-react';
import { apiService, getArabicOnly, setArabicOnly } from '../services/api';
import { HealthStatus } from '../types';
import { SaudiOverview, SaudiRuntime } from '../types/saudi';
import { timeAgo } from '../utils/formatDate';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [health, setHealth] = useState<HealthStatus>({
    status: 'checking',
    service: 'metaview',
    rows: 0,
    time: '',
  });
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [headerSearch, setHeaderSearch] = useState('');
  const [arabicOnly, setArabicOnlyState] = useState<boolean>(getArabicOnly());
  const toggleArabicOnly = () => {
    const next = !arabicOnly;
    setArabicOnly(next);
    setArabicOnlyState(next);
  };
  const navigate = useNavigate();
  const location = useLocation();

  const [saudiOv, setSaudiOv] = useState<SaudiOverview | null>(null);
  const [runtime, setRuntime] = useState<SaudiRuntime | null>(null);

  const fetchHealth = async () => {
    const res = await apiService.getHealth();
    setHealth(res);
    apiService.getSaudiOverview().then(setSaudiOv).catch(() => {});
    apiService.getSaudiRuntime().then(setRuntime).catch(() => {});
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 25000);
    return () => clearInterval(interval);
  }, []);

  const handleGlobalRefresh = async () => {
    if (isRefreshing) return;
    setIsRefreshing(true);
    try {
      await apiService.runIngest(15);
      await fetchHealth();
      // Dispatch custom event to notify current active page to revalidate data
      window.dispatchEvent(new CustomEvent('metaview:refresh'));
    } catch (e) {
      console.error('Refresh error:', e);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleHeaderSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (headerSearch.trim()) {
      navigate(`/explorer?q=${encodeURIComponent(headerSearch.trim())}`);
    }
  };

  type NavItem = { to: string; label: string; icon: React.ComponentType<{ className?: string }>; exact?: boolean };
  const navSections: { title: string; items: NavItem[] }[] = [
    {
      title: 'الاستخبارات (Intelligence)',
      items: [
        { to: '/', label: 'النظرة التنفيذية', icon: LayoutDashboard, exact: true },
        { to: '/saudi-fraud', label: 'الاستخبارات السعودية', icon: ShieldAlert },
        { to: '/global', label: 'الاستخبارات العالمية', icon: Globe2 },
        { to: '/incident-registry', label: 'سجل الحوادث', icon: FolderSearch },
        { to: '/patterns', label: 'أنماط الاحتيال', icon: Network },
      ],
    },
    {
      title: 'الكشف (Detection)',
      items: [{ to: '/rules', label: 'قواعد الكشف', icon: ListChecks }],
    },
    {
      title: 'العمليات (Operations)',
      items: [
        { to: '/sources', label: 'مصادر البيانات', icon: Radio },
        { to: '/operations', label: 'مراقبة الاستيعاب', icon: Activity },
        { to: '/system-health', label: 'صحة النظام', icon: HeartPulse },
      ],
    },
    {
      title: 'أدوات التحليل (Tools)',
      items: [
        { to: '/explorer', label: 'مستكشف الأخبار', icon: Compass },
        { to: '/map', label: 'الخريطة العالمية', icon: MapPin },
        { to: '/repository', label: 'مستودع الأخبار', icon: Database },
        { to: '/ai-layer', label: 'طبقة الذكاء الاصطناعي', icon: Sparkles },
      ],
    },
  ];

  const isOnline = health.status !== 'offline' && health.status !== 'checking';

  return (
    <div className="min-h-screen bg-[#0A0E1A] text-[#f8fafc] flex flex-col font-sans">
      {/* Top Header */}
      <header className="sticky top-0 z-40 bg-[#0A0E1A]/95 border-b border-[#4cd7f6]/15 backdrop-blur-2xl h-16 px-4 md:px-6 flex items-center justify-between shadow-2xl">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2.5 rounded-xl bg-white/5 hover:bg-[#4cd7f6]/15 text-[#acedff] transition-colors"
            title="تبديل القائمة الجانبية"
          >
            {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>

          {/* Logo & Brand */}
          <div
            onClick={() => navigate('/')}
            className="flex items-center gap-3 cursor-pointer group select-none"
          >
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-[#4cd7f6] to-[#06b6d4] flex items-center justify-center shadow-[0_0_20px_rgba(76,215,246,0.3)] group-hover:scale-105 transition-transform">
              <Zap className="w-5 h-5 text-[#003640] fill-current" />
            </div>
            <div className="flex flex-col">
              <span className="text-xl font-black tracking-tight text-white royal-gradient-text uppercase">
                MetaView 3
              </span>
              <span className="text-[9px] font-mono tracking-widest text-[#4cd7f6] uppercase -mt-0.5">
                Fraud Intelligence
              </span>
            </div>
          </div>
        </div>

        {/* Global Search Bar */}
        <div className="flex-1 max-w-xl mx-6 hidden md:block">
          <form onSubmit={handleHeaderSearchSubmit} className="relative">
            <input
              type="text"
              value={headerSearch}
              onChange={(e) => setHeaderSearch(e.target.value)}
              placeholder="البحث الفوري في كافة الأخبار والمصادر العالمية..."
              className="w-full bg-[#0F172A] border border-[#4cd7f6]/20 focus:border-[#4cd7f6]/60 focus:bg-[#121522] py-2.5 pr-11 pl-4 rounded-xl text-xs text-white placeholder-gray-500 outline-none transition-all shadow-inner"
            />
            <button
              type="submit"
              className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-gray-500 hover:text-[#4cd7f6] transition-colors"
            >
              <Search className="w-4 h-4" />
            </button>
          </form>
        </div>

        {/* Right Tools & Status */}
        <div className="flex items-center gap-3">
          {/* Arabic-only language filter */}
          <button
            id="arabic-only-toggle"
            onClick={toggleArabicOnly}
            aria-pressed={arabicOnly}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold border transition-all ${
              arabicOnly
                ? 'bg-[#4cd7f6]/25 border-[#4cd7f6]/60 text-[#acedff]'
                : 'bg-white/5 hover:bg-[#4cd7f6]/15 border-white/10 hover:border-[#4cd7f6]/30 text-gray-200'
            }`}
            title="عرض الأخبار المنشورة باللغة العربية فقط في كل الصفحات"
          >
            <Languages className="w-3.5 h-3.5 text-[#4cd7f6]" />
            <span>{arabicOnly ? 'عربي فقط: مفعّل' : 'عربي فقط'}</span>
          </button>

          {/* Refresh Action Trigger */}
          <button
            onClick={handleGlobalRefresh}
            disabled={isRefreshing}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold border transition-all ${
              isRefreshing
                ? 'bg-[#4cd7f6]/20 border-[#4cd7f6]/40 text-[#acedff] cursor-not-allowed'
                : 'bg-white/5 hover:bg-[#4cd7f6]/15 border-white/10 hover:border-[#4cd7f6]/30 text-gray-200'
            }`}
            title="تحديث تدفق الأخبار وسحب المقالات الجديدة عبر /ingest/run"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-[#4cd7f6] ${isRefreshing ? 'animate-spin' : ''}`} />
            <span className="hidden sm:inline">
              {isRefreshing ? 'جاري الرصد...' : 'تحديث التدفق (Sync)'}
            </span>
          </button>

          {/* Live ingestion pill (real data: last Saudi fetch + sources OK) */}
          {saudiOv && (
            <div
              className="hidden xl:flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-low border border-edge-subtle"
              title="آخر جلب لمصادر الاحتيال السعودية"
            >
              <span className={`w-2 h-2 rounded-full ${saudiOv.ingest_running ? 'bg-threat-medium animate-pulse' : 'bg-threat-low'}`} />
              <span className="text-[11px] font-mono text-ink">
                {saudiOv.ingest_running ? 'جلب جارٍ' : `آخر جلب: ${timeAgo(saudiOv.last_fetch_at)}`}
              </span>
              <span className="text-edge-strong">|</span>
              <span className="text-[11px] font-mono text-brand">
                {saudiOv.sources_ok}/{saudiOv.sources_total} مصدر
              </span>
            </div>
          )}

          {/* System Health Pill */}
          <div
            onClick={() => navigate('/operations')}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[#0F172A] border border-white/10 hover:border-[#4cd7f6]/40 cursor-pointer transition-colors"
            title="حالة الاتصال بالنظام وقاعدة البيانات"
          >
            <div className="relative flex items-center justify-center">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  isOnline ? 'bg-emerald-400' : 'bg-red-500'
                }`}
              />
              {isOnline && (
                <span className="absolute w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping opacity-75" />
              )}
            </div>
            <div className="flex flex-col text-right">
              <span className="text-[10px] font-mono text-gray-400 leading-tight">
                {isOnline ? 'LIVE API' : 'OFFLINE'}
              </span>
              <span className="text-[11px] font-mono font-bold text-white leading-tight">
                {health.rows.toLocaleString()} خبر
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Workspace Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Collapsible Sidebar */}
        <aside
          className={`${
            sidebarOpen ? 'w-64' : 'w-20'
          } shrink-0 bg-surface-lowest border-l border-edge-subtle transition-all duration-300 flex flex-col justify-between hidden sm:flex z-30`}
        >
          {/* Navigation Sections */}
          <div className="p-3 space-y-5 overflow-y-auto">
            {navSections.map((section) => (
              <div key={section.title} className="space-y-1">
                <div className="px-3 pb-1 label-mono">{sidebarOpen ? section.title : '•••'}</div>
                {section.items.map((link) => {
                  const Icon = link.icon;
                  const isActive = link.exact ? location.pathname === link.to : location.pathname.startsWith(link.to);
                  return (
                    <NavLink
                      key={link.to}
                      to={link.to}
                      className={`flex items-center gap-3 px-3 py-2 rounded-lg text-[13px] transition-colors ${
                        isActive
                          ? 'bg-surface-high text-brand border-r-2 border-brand font-medium'
                          : 'text-ink-variant hover:text-ink hover:bg-surface-container'
                      }`}
                      title={link.label}
                    >
                      <Icon className={`w-[18px] h-[18px] shrink-0 ${isActive ? 'text-brand' : 'text-ink-outline'}`} />
                      {sidebarOpen && <span className="truncate">{link.label}</span>}
                    </NavLink>
                  );
                })}
              </div>
            ))}
          </div>

          {/* Sidebar footer: real deployment state */}
          <div className="p-3 border-t border-edge-subtle">
            {sidebarOpen ? (
              <div className="p-2.5 rounded-lg bg-surface-low border border-edge-subtle space-y-1.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-threat-low animate-pulse' : 'bg-threat-critical'}`} />
                    <span className="text-[11px] font-mono text-ink">{isOnline ? 'API ONLINE' : 'API OFFLINE'}</span>
                  </div>
                  <Gauge className="w-3.5 h-3.5 text-ink-outline" />
                </div>
                <div className="text-[10px] font-mono text-ink-outline">
                  {runtime?.commit ? `commit ${runtime.commit.slice(0, 7)}` : runtime ? 'commit: غير مُعلن (محلي)' : '—'}
                </div>
              </div>
            ) : (
              <div className="flex justify-center">
                <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-threat-low' : 'bg-threat-critical'}`} />
              </div>
            )}
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto bg-surface p-4 md:p-6">
          <div key={arabicOnly ? 'ar' : 'all'} className="max-w-[1500px] mx-auto">{children}</div>
        </main>
      </div>
    </div>
  );
};
