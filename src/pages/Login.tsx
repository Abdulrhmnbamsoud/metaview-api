// src/pages/Login.tsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, Lock, Mail, ArrowRight, UserCheck, CheckCircle2, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { UserRole } from '../types/auth';

export const Login: React.FC = () => {
  const { login, switchPersona, loading } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('يرجى إدخال البريد الإلكتروني وكلمة المرور.');
      return;
    }
    setError(null);
    setIsSubmitting(true);
    try {
      await login(email.trim(), password);
      navigate('/');
    } catch (err: any) {
      setError(err.message || 'بيانات الدخول غير صحيحة.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleQuickLogin = async (role: UserRole) => {
    setError(null);
    setIsSubmitting(true);
    try {
      await switchPersona(role);
      navigate('/');
    } catch (err: any) {
      setError(err.message || 'تعذر تسجيل الدخول بالحساب التجريبي.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-stone-100/70 dark:bg-[#070b14] flex flex-col justify-center items-center p-4">
      <div className="w-full max-w-md bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 shadow-xl overflow-hidden p-6 sm:p-8 space-y-6">
        {/* Logo and Header */}
        <div className="text-center space-y-2">
          <div className="w-12 h-12 rounded-xl bg-blue-600 text-white mx-auto flex items-center justify-center shadow-md">
            <Shield className="w-6 h-6" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-stone-900 dark:text-white">
            MetaView 3 — تسجيل الدخول
          </h1>
          <p className="text-xs text-stone-500 dark:text-slate-400">
            منظومة ميتا فيو لمكافحة ورصد الاحتيال المالي
          </p>
        </div>

        {/* Error message */}
        {error && (
          <div className="p-3 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800/60 text-red-700 dark:text-red-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-red-500" />
            <span>{error}</span>
          </div>
        )}

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
              البريد الإلكتروني
            </label>
            <div className="relative">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@organization.sa"
                dir="ltr"
                className="w-full px-3.5 py-2.5 pl-10 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs text-stone-900 dark:text-white placeholder:text-stone-400 focus:outline-none focus:border-blue-500 transition-colors"
                required
              />
              <Mail className="w-4 h-4 text-stone-400 absolute top-1/2 -translate-y-1/2 left-3 pointer-events-none" />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
              كلمة المرور
            </label>
            <div className="relative">
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                dir="ltr"
                className="w-full px-3.5 py-2.5 pl-10 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs text-stone-900 dark:text-white placeholder:text-stone-400 focus:outline-none focus:border-blue-500 transition-colors"
                required
              />
              <Lock className="w-4 h-4 text-stone-400 absolute top-1/2 -translate-y-1/2 left-3 pointer-events-none" />
            </div>
          </div>

          <button
            type="submit"
            disabled={isSubmitting || loading}
            className="w-full py-2.5 px-4 rounded-xl bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white font-semibold text-xs shadow-sm flex items-center justify-center gap-2 transition-all disabled:opacity-50"
          >
            {isSubmitting ? 'جارِ التحقق...' : 'تسجيل الدخول'}
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>

        {/* Quick Demo Personas (Evaluation Tool) */}
        <div className="pt-4 border-t border-stone-200/80 dark:border-slate-800 space-y-2.5">
          <div className="text-[11px] font-semibold text-stone-500 dark:text-slate-400 text-center">
            اختيار حساب تجريبي سريع (للتقييم والمراجعة):
          </div>
          <div className="grid grid-cols-3 gap-2">
            <button
              type="button"
              onClick={() => handleQuickLogin('admin')}
              disabled={isSubmitting}
              className="p-2.5 rounded-xl border border-stone-200 dark:border-slate-700 hover:border-blue-500 hover:bg-blue-50/50 dark:hover:bg-blue-950/30 text-center transition-all group"
            >
              <span className="block text-xs font-bold text-stone-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400">
                مدير
              </span>
              <span className="text-[10px] text-stone-400 dark:text-slate-400">
                ADMIN
              </span>
            </button>

            <button
              type="button"
              onClick={() => handleQuickLogin('analyst')}
              disabled={isSubmitting}
              className="p-2.5 rounded-xl border border-stone-200 dark:border-slate-700 hover:border-amber-500 hover:bg-amber-50/50 dark:hover:bg-amber-950/30 text-center transition-all group"
            >
              <span className="block text-xs font-bold text-stone-900 dark:text-white group-hover:text-amber-600 dark:group-hover:text-amber-400">
                محلل
              </span>
              <span className="text-[10px] text-stone-400 dark:text-slate-400">
                ANALYST
              </span>
            </button>

            <button
              type="button"
              onClick={() => handleQuickLogin('viewer')}
              disabled={isSubmitting}
              className="p-2.5 rounded-xl border border-stone-200 dark:border-slate-700 hover:border-emerald-500 hover:bg-emerald-50/50 dark:hover:bg-emerald-950/30 text-center transition-all group"
            >
              <span className="block text-xs font-bold text-stone-900 dark:text-white group-hover:text-emerald-600 dark:group-hover:text-emerald-400">
                مستعرض
              </span>
              <span className="text-[10px] text-stone-400 dark:text-slate-400">
                VIEWER
              </span>
            </button>
          </div>
        </div>

        <div className="text-[11px] text-center text-stone-400 dark:text-slate-500">
          التسجيل في MetaView 3 يتم عبر الدعوات المعتمدة من مدير المنصة فقط.
        </div>
      </div>
    </div>
  );
};
