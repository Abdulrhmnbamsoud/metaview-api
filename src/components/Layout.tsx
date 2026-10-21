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
} from 'lucide-react';
import { apiService } from '../services/api';
import { HealthStatus } from '../types';

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
  const navigate = useNavigate();
  const location = useLocation();

  const fetchHealth = async () => {
    const res = await apiService.getHealth();
    setHealth(res);
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

  const navLinks = [
    { to: '/', label: 'لوحة القيادة العالمية', icon: LayoutDashboard, exact: true },
    { to: '/explorer', label: 'مستكشف الأخبار', icon: Compass },
    { to: '/map', label: 'الخريطة العالمية', icon: MapPin },
    { to: '/operations', label: 'مركز العمليات', icon: Activity },
    { to: '/repository', label: 'مستودع الأخبار', icon: Database },
    { to: '/sources', label: 'تحليل المصادر', icon: Radio },
    { to: '/ai-layer', label: 'طبقة الذكاء الاصطناعي', icon: Sparkles },
  ];

  const isOnline = health.status !== 'offline' && health.status !== 'checking';

  return (
    <div className="min-h-screen bg-[#050508] text-[#f8fafc] flex flex-col font-sans">
      {/* Top Header */}
      <header className="sticky top-0 z-40 bg-[#08090f]/95 border-b border-[#c5a059]/15 backdrop-blur-2xl h-20 px-4 md:px-8 flex items-center justify-between shadow-2xl">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2.5 rounded-xl bg-white/5 hover:bg-[#c5a059]/15 text-[#e0c58a] transition-colors"
            title="تبديل القائمة الجانبية"
          >
            {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>

          {/* Logo & Brand */}
          <div
            onClick={() => navigate('/')}
            className="flex items-center gap-3 cursor-pointer group select-none"
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#c5a059] to-[#8e6d2f] flex items-center justify-center shadow-[0_0_20px_rgba(197,160,89,0.35)] group-hover:scale-105 transition-transform">
              <Zap className="w-6 h-6 text-black fill-current" />
            </div>
            <div className="flex flex-col">
              <span className="text-xl font-black tracking-tight text-white royal-gradient-text uppercase">
                MetaView 3
              </span>
              <span className="text-[9px] font-mono tracking-widest text-[#c5a059] uppercase -mt-0.5">
                Global News Intelligence
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
              className="w-full bg-[#0d0f18] border border-[#c5a059]/20 focus:border-[#c5a059]/60 focus:bg-[#121522] py-2.5 pr-11 pl-4 rounded-xl text-xs text-white placeholder-gray-500 outline-none transition-all shadow-inner"
            />
            <button
              type="submit"
              className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-gray-500 hover:text-[#c5a059] transition-colors"
            >
              <Search className="w-4 h-4" />
            </button>
          </form>
        </div>

        {/* Right Tools & Status */}
        <div className="flex items-center gap-3">
          {/* Refresh Action Trigger */}
          <button
            onClick={handleGlobalRefresh}
            disabled={isRefreshing}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold border transition-all ${
              isRefreshing
                ? 'bg-[#c5a059]/20 border-[#c5a059]/40 text-[#e0c58a] cursor-not-allowed'
                : 'bg-white/5 hover:bg-[#c5a059]/15 border-white/10 hover:border-[#c5a059]/30 text-gray-200'
            }`}
            title="تحديث تدفق الأخبار وسحب المقالات الجديدة عبر /ingest/run"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-[#c5a059] ${isRefreshing ? 'animate-spin' : ''}`} />
            <span className="hidden sm:inline">
              {isRefreshing ? 'جاري الرصد...' : 'تحديث التدفق (Sync)'}
            </span>
          </button>

          {/* System Health Pill */}
          <div
            onClick={() => navigate('/operations')}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[#0d0f18] border border-white/10 hover:border-[#c5a059]/40 cursor-pointer transition-colors"
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
          } shrink-0 bg-[#07080d] border-l border-[#c5a059]/10 transition-all duration-300 flex flex-col justify-between hidden sm:flex z-30`}
        >
          {/* Navigation Links */}
          <div className="p-3 space-y-1.5">
            <div className="px-3 py-2 text-[10px] font-mono text-gray-500 tracking-wider uppercase">
              {sidebarOpen ? 'وحدات المنصة (Modules)' : '•••'}
            </div>

            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = link.exact
                ? location.pathname === link.to
                : location.pathname.startsWith(link.to);

              return (
                <NavLink
                  key={link.to}
                  to={link.to}
                  className={`flex items-center gap-3 px-3.5 py-3 rounded-xl text-xs font-semibold transition-all ${
                    isActive
                      ? 'bg-gradient-to-r from-[#c5a059]/20 to-transparent text-[#e0c58a] border-r-2 border-[#c5a059] shadow-sm'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                  title={link.label}
                >
                  <Icon className={`w-5 h-5 shrink-0 ${isActive ? 'text-[#c5a059]' : 'text-gray-400'}`} />
                  {sidebarOpen && <span className="truncate">{link.label}</span>}
                </NavLink>
              );
            })}
          </div>

          {/* Sidebar Footer Info */}
          <div className="p-4 border-t border-white/5">
            {sidebarOpen ? (
              <div className="p-3 rounded-xl bg-[#0c0e16] border border-white/5 text-[11px] space-y-1">
                <div className="flex items-center justify-between text-gray-400 font-mono">
                  <span>SYSTEM</span>
                  <span className="text-emerald-400 font-bold">READY</span>
                </div>
                <div className="text-[10px] text-gray-500 leading-tight">
                  MetaView v3.0 Intelligence Core
                </div>
              </div>
            ) : (
              <div className="flex justify-center text-gray-600 font-mono text-[10px]">v3.0</div>
            )}
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto bg-[#050508] p-4 md:p-8">
          <div className="max-w-7xl mx-auto">{children}</div>
        </main>
      </div>
    </div>
  );
};
