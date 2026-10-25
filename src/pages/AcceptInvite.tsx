// src/pages/AcceptInvite.tsx
import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { ShieldCheck, Lock, User, CheckCircle2, AlertCircle, ArrowRight } from 'lucide-react';
import { apiService } from '../services/api';
import { useAuth } from '../context/AuthContext';

export const AcceptInvite: React.FC = () => {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token') || '';
  const navigate = useNavigate();
  const { user } = useAuth();

  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!token) {
      setError('رابط الدعوة غير متوفر أو غير مكتمل.');
    }
  }, [token]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) {
      setError('رمز الدعوة مفقود.');
      return;
    }
    if (fullName.trim().length < 2) {
      setError('يرجى إدخال الاسم الكامل.');
      return;
    }
    if (password.length < 8) {
      setError('كلمة المرور يجب أن لا تقل عن 8 خانات.');
      return;
    }
    if (password !== confirmPassword) {
      setError('كلمتا المرور غير متطابقتين.');
      return;
    }

    setError(null);
    setIsSubmitting(true);
    try {
      const res = await apiService.acceptInvitation({
        token: token.trim(),
        full_name: fullName.trim(),
        password,
      });
      setSuccess(`مرحباً بك يا ${res.user.full_name}! تم تفعيل حسابك بنجاح.`);
      setTimeout(() => {
        navigate('/');
      }, 1500);
    } catch (err: any) {
      setError(err.message || 'تعذر قبول الدعوة. قد يكون الرابط منتهياً أو تم استخدامه مسبقاً.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-stone-100/70 dark:bg-[#070b14] flex flex-col justify-center items-center p-4">
      <div className="w-full max-w-md bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 shadow-xl overflow-hidden p-6 sm:p-8 space-y-6">
        <div className="text-center space-y-2">
          <div className="w-12 h-12 rounded-xl bg-emerald-600 text-white mx-auto flex items-center justify-center shadow-md">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-stone-900 dark:text-white">
            قبول دعوة الانضمام لفريق MetaView
          </h1>
          <p className="text-xs text-stone-500 dark:text-slate-400">
            أكمل إنشاء حسابك للانضمام إلى منصة ميتا فيو لرصد الاحتيال المالي
          </p>
        </div>

        {error && (
          <div className="p-3 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800/60 text-red-700 dark:text-red-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-red-500" />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="p-3 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/60 text-emerald-700 dark:text-emerald-300 text-xs flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-500" />
            <span>{success}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
              الاسم الكامل
            </label>
            <div className="relative">
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="مثال: د. عبدالعزيز السالم"
                className="w-full px-3.5 py-2.5 pl-10 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs text-stone-900 dark:text-white placeholder:text-stone-400 focus:outline-none focus:border-blue-500 transition-colors"
                required
              />
              <User className="w-4 h-4 text-stone-400 absolute top-1/2 -translate-y-1/2 left-3 pointer-events-none" />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
              تعيين كلمة المرور
            </label>
            <div className="relative">
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="•••••••••••• (8 خانات كحد أدنى)"
                dir="ltr"
                className="w-full px-3.5 py-2.5 pl-10 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs text-stone-900 dark:text-white placeholder:text-stone-400 focus:outline-none focus:border-blue-500 transition-colors"
                required
              />
              <Lock className="w-4 h-4 text-stone-400 absolute top-1/2 -translate-y-1/2 left-3 pointer-events-none" />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
              تأكيد كلمة المرور
            </label>
            <div className="relative">
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
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
            disabled={isSubmitting || !token}
            className="w-full py-2.5 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 text-white font-semibold text-xs shadow-sm flex items-center justify-center gap-2 transition-all disabled:opacity-50"
          >
            {isSubmitting ? 'جارِ تفعيل الحساب...' : 'تفعيل الحساب والدخول'}
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
};
