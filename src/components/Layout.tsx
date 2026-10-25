// src/components/Layout.tsx
// Linear / Notion inspired information-first layout matching the exact design mockup
import React, { useState, useEffect } from 'react';
import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import {
  Search,
  Compass,
  ListChecks,
  Server,
  Home,
  Sun,
  Moon,
  ShieldAlert,
  AlertTriangle,
  Network,
  BellRing,
  CheckCircle2,
  Menu,
  X,
  User,
  Activity,
  Briefcase,
  Users,
  LogOut,
  ChevronDown,
  Shield,
  Check,
} from 'lucide-react';
import { apiService, getArabicOnly, setArabicOnly } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { UserRole } from '../types/auth';
import { HealthStatus } from '../types';
import { SaudiOverview } from '../types/saudi';
import { getStoredTheme, applyTheme, Theme } from '../utils/theme';
import { fmtDateTime } from './ui';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>(getStoredTheme());
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [headerSearch, setHeaderSearch] = useState('');
  const [health, setHealth] = useState<HealthStatus>({
    status: 'checking',
    service: 'metaview',
    rows: 0,
    time: '',
  });
  const [overview, setOverview] = useState<SaudiOverview | null>(null);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  const { user, role, can, logout, switchPersona } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const toggleTheme = () => {
    const nextTheme: Theme = theme === 'dark' ? 'light' : 'dark';
    setTheme(nextTheme);
    applyTheme(nextTheme);
  };

  const fetchStatus = async () => {
    try {
      const [h, ov] = await Promise.all([apiService.getHealth(), apiService.getSaudiOverview()]);
      setHealth(h);
      setOverview(ov);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleHeaderSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (headerSearch.trim()) {
      navigate(`/explore?q=${encodeURIComponent(headerSearch.trim())}`);
    }
  };

  const isOnline = health.status !== 'offline' && health.status !== 'checking';

  // Dynamic Navigation based on RBAC Least Privilege
  const primaryNav = [
    { to: '/', label: 'الرئيسية', icon: Home, exact: true },
    { to: '/explore', label: 'الاستكشاف', icon: Compass },
    { to: '/rules', label: 'قواعد الكشف', icon: ListChecks },
    ...(can('rule.create') || can('incident.annotate') || role === 'analyst' || role === 'admin'
      ? [{ to: '/workspace', label: 'مساحة العمل', icon: Briefcase }]
      : []),
    { to: '/system', label: 'المصادر والنظام', icon: Server },
    ...(can('team.manage') || role === 'admin'
      ? [{ to: '/team', label: 'إدارة الفريق', icon: Users }]
      : []),
  ];

  // Focus areas (مناطق التركيز)
  const focusAreas = [
    { to: '/explore?type=incident', label: 'الحوادث', icon: ShieldAlert },
    { to: '/explore?type=warning', label: 'التحذيرات', icon: AlertTriangle },
    { to: '/explore?type=pattern', label: 'الأنماط', icon: Network },
    { to: '/explore?type=awareness', label: 'التوعية', icon: BellRing },
  ];

  return (
    <div className="min-h-screen bg-stone-100/70 dark:bg-[#070b14] text-stone-900 dark:text-stone-100 flex flex-col font-sans antialiased transition-colors duration-150">
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar (Matching Screenshot) */}
        <aside className="w-56 shrink-0 bg-white dark:bg-[#0f172a] border-l border-stone-200/80 dark:border-slate-800 hidden md:flex flex-col justify-between z-30 select-none">
          <div className="p-4 space-y-6">
            {/* Logo */}
            <div
              onClick={() => navigate('/')}
              className="flex items-center gap-2.5 cursor-pointer group"
            >
              <div className="w-8 h-8 rounded-lg bg-blue-600 text-white flex items-center justify-center shadow-sm">
                {/* 4-wave stylized icon */}
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 12c2-4 4-4 6 0s4 4 6 0 4-4 4-4" />
                </svg>
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-bold tracking-tight text-stone-900 dark:text-white leading-none">
                  MetaView
                </span>
                <span className="text-[10px] font-sans text-stone-400 dark:text-slate-400 mt-1">
                  ميتا فيو
                </span>
              </div>
            </div>

            {/* Primary Navigation */}
            <nav className="space-y-1">
              {primaryNav.map((item) => {
                const Icon = item.icon;
                const isActive = item.exact
                  ? location.pathname === item.to
                  : location.pathname.startsWith(item.to);

                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    className={`flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-semibold transition-all ${
                      isActive
                        ? 'bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 shadow-xs'
                        : 'text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-stone-100 hover:bg-stone-50 dark:hover:bg-slate-800/40'
                    }`}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? 'text-blue-600 dark:text-blue-400' : 'text-stone-400 dark:text-slate-500'}`} />
                    <span>{item.label}</span>
                  </NavLink>
                );
              })}
            </nav>

            {/* Focus Areas (مناطق التركيز) */}
            <div className="space-y-1.5 pt-2">
              <div className="px-3 text-[11px] font-medium text-stone-400 dark:text-slate-500">
                مناطق التركيز
              </div>
              {focusAreas.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    className="flex items-center gap-3 px-3 py-1.5 rounded-lg text-xs text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-stone-100 hover:bg-stone-50 dark:hover:bg-slate-800/40 transition-colors"
                  >
                    <Icon className="w-3.5 h-3.5 text-stone-400 dark:text-slate-500" />
                    <span>{item.label}</span>
                  </NavLink>
                );
              })}
            </div>
          </div>

          {/* Bottom Status Widget */}
          <div className="p-4 border-t border-stone-100 dark:border-slate-800/80 bg-stone-50/50 dark:bg-slate-900/30">
            <div className="p-2.5 rounded-xl bg-white dark:bg-slate-800/60 border border-stone-200/70 dark:border-slate-700/60 space-y-1">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
                <span className="text-xs font-semibold text-stone-800 dark:text-stone-200">
                  {isOnline ? 'النظام يعمل' : 'غير متصل'}
                </span>
              </div>
              <div className="text-[10px] text-stone-400 dark:text-slate-400 font-mono">
                آخر سحب للبيانات:
              </div>
              <div className="text-[10px] text-stone-600 dark:text-slate-300 font-mono font-medium">
                {overview?.last_fetch_at ? fmtDateTime(overview.last_fetch_at) : '13:18، 23 أكتوبر'}
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
          {/* Top Header */}
          <header className="sticky top-0 z-40 bg-white/95 dark:bg-[#0f172a]/95 border-b border-stone-200/80 dark:border-slate-800 backdrop-blur-md h-15 px-4 sm:px-6 flex items-center justify-between gap-4">
            {/* Search Input with ⌘ K */}
            <form onSubmit={handleHeaderSearch} className="flex-1 max-w-md relative hidden sm:block">
              <input
                type="text"
                value={headerSearch}
                onChange={(e) => setHeaderSearch(e.target.value)}
                placeholder="ابحث عن حادثة احتيال، أسلوب، جهة، أو مؤشر خطر..."
                className="w-full bg-stone-100/80 dark:bg-slate-800/60 border border-transparent focus:border-blue-500 dark:focus:border-blue-400 focus:bg-white dark:focus:bg-slate-900 py-2 pr-9 pl-12 rounded-xl text-xs text-stone-900 dark:text-stone-100 placeholder:text-stone-400 outline-none transition-all"
              />
              <Search className="w-4 h-4 text-stone-400 absolute top-1/2 -translate-y-1/2 right-3 pointer-events-none" />
              <kbd className="absolute top-1/2 -translate-y-1/2 left-2.5 px-1.5 py-0.5 rounded bg-white dark:bg-slate-700 border border-stone-200 dark:border-slate-600 text-[10px] font-mono text-stone-400">
                ⌘ K
              </kbd>
            </form>

            {/* Mobile menu trigger */}
            <div className="flex items-center gap-2 md:hidden">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="p-1.5 rounded-lg text-stone-600 dark:text-slate-400 hover:bg-stone-100"
              >
                {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
              <span className="font-bold text-sm">MetaView 3</span>
            </div>

            {/* Right Tools & Badges */}
            <div className="flex items-center gap-3">
              {/* Saudi Sector Badge */}
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/60 text-emerald-700 dark:text-emerald-300 text-xs font-semibold">
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                <span>السعودية</span>
              </div>

              {/* Last update */}
              <div className="hidden lg:flex items-center gap-1.5 text-xs text-stone-500 dark:text-slate-400 font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span>آخر تحديث: {overview?.last_fetch_at ? fmtDateTime(overview.last_fetch_at) : '13:18، 23 أكتوبر'}</span>
              </div>

              {/* Theme toggle */}
              <button
                onClick={toggleTheme}
                className="p-2 rounded-xl text-stone-500 hover:text-stone-900 dark:text-slate-400 dark:hover:text-stone-100 hover:bg-stone-100 dark:hover:bg-slate-800 transition-colors"
                title={theme === 'dark' ? 'الوضع النهاري' : 'الوضع الليلي'}
              >
                {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-stone-600" />}
              </button>

              {/* User Profile & Persona Switcher */}
              <div className="relative">
                {user ? (
                  <button
                    onClick={() => setUserMenuOpen(!userMenuOpen)}
                    className="flex items-center gap-2 p-1.5 pr-2.5 rounded-xl border border-stone-200 dark:border-slate-700 hover:bg-stone-50 dark:hover:bg-slate-800 transition-colors"
                  >
                    <div className="w-6 h-6 rounded-lg bg-blue-600 text-white flex items-center justify-center text-[10px] font-bold">
                      {user.full_name?.charAt(0) || 'U'}
                    </div>
                    <div className="hidden sm:flex flex-col text-right leading-tight">
                      <span className="text-xs font-bold text-stone-900 dark:text-white">
                        {user.full_name}
                      </span>
                      <span className="text-[10px] text-blue-600 dark:text-blue-400 font-medium">
                        {role === 'admin' ? 'مدير المنصة' : role === 'analyst' ? 'محلل ميتا فيو' : 'مستعرض'}
                      </span>
                    </div>
                    <ChevronDown className="w-3.5 h-3.5 text-stone-400" />
                  </button>
                ) : (
                  <button
                    onClick={() => navigate('/login')}
                    className="px-3 py-1.5 rounded-xl bg-blue-600 text-white text-xs font-semibold shadow-xs hover:bg-blue-700"
                  >
                    تسجيل الدخول
                  </button>
                )}

                {/* Dropdown Menu */}
                {userMenuOpen && user && (
                  <div className="absolute left-0 mt-2 w-64 bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200 dark:border-slate-800 shadow-xl p-2 space-y-2 z-50 text-right">
                    <div className="p-2 border-b border-stone-100 dark:border-slate-800">
                      <div className="text-xs font-bold text-stone-900 dark:text-white">{user.full_name}</div>
                      <div className="text-[11px] text-stone-400 font-mono" dir="ltr">{user.email}</div>
                    </div>

                    <div className="space-y-1">
                      <div className="text-[10px] font-semibold text-stone-400 px-2 py-0.5">
                        تبديل الدور للتجربة (Evaluation Switcher):
                      </div>
                      <button
                        onClick={() => {
                          switchPersona('admin');
                          setUserMenuOpen(false);
                        }}
                        className={`w-full text-right px-2.5 py-1.5 rounded-xl text-xs flex items-center justify-between transition-colors ${
                          role === 'admin'
                            ? 'bg-purple-50 dark:bg-purple-950/50 text-purple-700 dark:text-purple-300 font-bold'
                            : 'hover:bg-stone-50 dark:hover:bg-slate-800 text-stone-700 dark:text-slate-300'
                        }`}
                      >
                        <span>مدير المنصة (ADMIN)</span>
                        {role === 'admin' && <Check className="w-3.5 h-3.5" />}
                      </button>

                      <button
                        onClick={() => {
                          switchPersona('analyst');
                          setUserMenuOpen(false);
                        }}
                        className={`w-full text-right px-2.5 py-1.5 rounded-xl text-xs flex items-center justify-between transition-colors ${
                          role === 'analyst'
                            ? 'bg-blue-50 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300 font-bold'
                            : 'hover:bg-stone-50 dark:hover:bg-slate-800 text-stone-700 dark:text-slate-300'
                        }`}
                      >
                        <span>محلل ميتا فيو (ANALYST)</span>
                        {role === 'analyst' && <Check className="w-3.5 h-3.5" />}
                      </button>

                      <button
                        onClick={() => {
                          switchPersona('viewer');
                          setUserMenuOpen(false);
                        }}
                        className={`w-full text-right px-2.5 py-1.5 rounded-xl text-xs flex items-center justify-between transition-colors ${
                          role === 'viewer'
                            ? 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 font-bold'
                            : 'hover:bg-stone-50 dark:hover:bg-slate-800 text-stone-700 dark:text-slate-300'
                        }`}
                      >
                        <span>مستعرض مصرح (VIEWER)</span>
                        {role === 'viewer' && <Check className="w-3.5 h-3.5" />}
                      </button>
                    </div>

                    <div className="pt-1 border-t border-stone-100 dark:border-slate-800">
                      <button
                        onClick={() => {
                          logout();
                          setUserMenuOpen(false);
                          navigate('/login');
                        }}
                        className="w-full text-right px-2.5 py-1.5 rounded-xl text-xs text-red-600 hover:bg-red-50 dark:hover:bg-red-950/30 flex items-center gap-2 transition-colors"
                      >
                        <LogOut className="w-3.5 h-3.5" />
                        <span>تسجيل الخروج</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </header>

          {/* Mobile Drawer */}
          {mobileMenuOpen && (
            <div className="md:hidden bg-white dark:bg-[#0f172a] border-b border-stone-200 dark:border-slate-800 p-4 space-y-2">
              {primaryNav.map((link) => (
                <NavLink
                  key={link.to}
                  to={link.to}
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-semibold text-stone-700 dark:text-slate-200"
                >
                  <link.icon className="w-4 h-4 text-blue-600" />
                  <span>{link.label}</span>
                </NavLink>
              ))}
              <div className="pt-2 border-t border-stone-100 dark:border-slate-800">
                <div className="text-[11px] text-stone-400 px-3 pb-1">مناطق التركيز</div>
                {focusAreas.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    onClick={() => setMobileMenuOpen(false)}
                    className="flex items-center gap-3 px-3 py-1.5 rounded-lg text-xs text-stone-600 dark:text-slate-300"
                  >
                    <item.icon className="w-3.5 h-3.5" />
                    <span>{item.label}</span>
                  </NavLink>
                ))}
              </div>
            </div>
          )}

          {/* Page Workspace Body */}
          <main className="flex-1 p-4 lg:p-6 overflow-x-hidden">
            {children}
          </main>
        </div>
      </div>
    </div>
  );
};

export default Layout;
