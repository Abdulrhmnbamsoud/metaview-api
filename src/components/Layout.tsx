// src/components/Layout.tsx
// Information-first minimalist layout with 4 destinations, light-mode first, and dark-mode toggle
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
  Languages,
  Menu,
  X,
  Zap,
} from 'lucide-react';
import { apiService, getArabicOnly, setArabicOnly } from '../services/api';
import { HealthStatus } from '../types';
import { getStoredTheme, applyTheme, Theme } from '../utils/theme';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>(getStoredTheme());
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [arabicOnly, setArabicOnlyState] = useState<boolean>(getArabicOnly());
  const [health, setHealth] = useState<HealthStatus>({
    status: 'checking',
    service: 'metaview',
    rows: 0,
    time: '',
  });

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

  const toggleArabicOnly = () => {
    const next = !arabicOnly;
    setArabicOnly(next);
    setArabicOnlyState(next);
  };

  const fetchHealth = async () => {
    const res = await apiService.getHealth();
    setHealth(res);
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // 4 Main Destinations as specified
  const mainNav = [
    { to: '/', label: 'الرئيسية', icon: Home, exact: true },
    { to: '/explore', label: 'الاستكشاف الموحد', icon: Compass },
    { to: '/rules', label: 'قواعد الكشف', icon: ListChecks },
    { to: '/system', label: 'المصادر والنظام', icon: Server },
  ];

  const isOnline = health.status !== 'offline' && health.status !== 'checking';

  return (
    <div className="min-h-screen bg-stone-50 dark:bg-[#090d16] text-stone-900 dark:text-stone-100 flex flex-col font-sans transition-colors duration-150">
      {/* Top Header */}
      <header className="sticky top-0 z-40 bg-white/90 dark:bg-[#111726]/90 border-b border-stone-200 dark:border-slate-800/80 backdrop-blur-md h-15 px-4 sm:px-6 flex items-center justify-between">
        <div className="flex items-center gap-6">
          {/* Logo / Brand */}
          <div
            onClick={() => navigate('/')}
            className="flex items-center gap-2.5 cursor-pointer select-none group"
          >
            <div className="w-8 h-8 rounded-lg bg-stone-900 dark:bg-stone-100 text-white dark:text-stone-900 flex items-center justify-center font-bold text-xs">
              <Zap className="w-4 h-4 fill-current text-sky-400 dark:text-sky-600" />
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-bold tracking-tight text-stone-900 dark:text-stone-100">
                MetaView 3
              </span>
              <span className="text-[10px] font-mono text-stone-500 dark:text-slate-400 -mt-0.5">
                استخبارات الاحتيال
              </span>
            </div>
          </div>

          {/* 4 Primary Navigation Links (Desktop) */}
          <nav className="hidden md:flex items-center gap-1">
            {mainNav.map((link) => {
              const Icon = link.icon;
              const isActive = link.exact
                ? location.pathname === link.to
                : location.pathname.startsWith(link.to);

              return (
                <NavLink
                  key={link.to}
                  to={link.to}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    isActive
                      ? 'bg-stone-100 dark:bg-slate-800 text-stone-900 dark:text-stone-100 font-semibold'
                      : 'text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-stone-100 hover:bg-stone-50 dark:hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-sky-600 dark:text-sky-400' : 'text-stone-400 dark:text-slate-500'}`} />
                  <span>{link.label}</span>
                </NavLink>
              );
            })}
          </nav>
        </div>

        {/* Right Tools & Controls */}
        <div className="flex items-center gap-2">
          {/* Arabic-only filter toggle */}
          <button
            onClick={toggleArabicOnly}
            className={`px-2.5 py-1 rounded-md text-[11px] font-medium border transition-colors flex items-center gap-1.5 ${
              arabicOnly
                ? 'bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border-sky-300 dark:border-sky-800'
                : 'bg-white dark:bg-slate-800 text-stone-600 dark:text-slate-400 border-stone-200 dark:border-slate-700 hover:bg-stone-50'
            }`}
            title="عرض الأخبار والمحتوى العربي فقط"
          >
            <Languages className="w-3 h-3 text-sky-600 dark:text-sky-400" />
            <span className="hidden sm:inline">{arabicOnly ? 'عربي فقط: مفعّل' : 'عربي فقط'}</span>
          </button>

          {/* Theme Toggle (Light / Dark) */}
          <button
            onClick={toggleTheme}
            className="p-1.5 rounded-md text-stone-500 dark:text-slate-400 hover:text-stone-900 dark:hover:text-stone-100 hover:bg-stone-100 dark:hover:bg-slate-800 border border-transparent transition-colors"
            title={theme === 'dark' ? 'التحويل للوضع النهاري (Light)' : 'التحويل للوضع الليلي (Dark)'}
          >
            {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-stone-600" />}
          </button>

          {/* System status dot */}
          <div
            onClick={() => navigate('/system')}
            className="hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-md bg-stone-100 dark:bg-slate-800/80 border border-stone-200 dark:border-slate-700 text-[10px] font-mono cursor-pointer hover:border-stone-300 transition-colors"
            title="حالة اتصال قاعدة البيانات"
          >
            <span
              className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-red-500'}`}
            />
            <span className="text-stone-600 dark:text-slate-300">
              {isOnline ? `${health.rows.toLocaleString()} خبر` : 'OFFLINE'}
            </span>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-1.5 rounded-md text-stone-600 dark:text-slate-400 hover:bg-stone-100 dark:hover:bg-slate-800"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </header>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-white dark:bg-[#111726] border-b border-stone-200 dark:border-slate-800 p-3 space-y-1">
          {mainNav.map((link) => {
            const Icon = link.icon;
            const isActive = link.exact
              ? location.pathname === link.to
              : location.pathname.startsWith(link.to);

            return (
              <NavLink
                key={link.to}
                to={link.to}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                  isActive
                    ? 'bg-stone-100 dark:bg-slate-800 text-stone-900 dark:text-stone-100 font-semibold'
                    : 'text-stone-600 dark:text-slate-400 hover:bg-stone-50'
                }`}
              >
                <Icon className="w-4 h-4 text-sky-600 dark:text-sky-400" />
                <span>{link.label}</span>
              </NavLink>
            );
          })}
        </div>
      )}

      {/* Main Workspace */}
      <main className="flex-1 p-4 sm:p-6 lg:p-8">
        <div key={arabicOnly ? 'ar' : 'all'}>{children}</div>
      </main>
    </div>
  );
};

export default Layout;
