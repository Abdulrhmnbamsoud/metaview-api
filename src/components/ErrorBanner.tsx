// src/components/ErrorBanner.tsx
import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorBannerProps {
  message?: string;
  onRetry?: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  message = 'تعذر الاتصال بخادم الأخبار. يرجى التحقق من اتصال الشبكة أو حالة الـ API.',
  onRetry,
}) => {
  return (
    <div className="p-5 rounded-2xl bg-red-950/40 border border-red-500/30 flex flex-col sm:flex-row items-center justify-between gap-4 my-6">
      <div className="flex items-center gap-3 text-red-200">
        <AlertTriangle className="w-6 h-6 text-red-400 shrink-0" />
        <span className="text-sm leading-relaxed">{message}</span>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/20 hover:bg-red-500/30 text-red-300 font-semibold text-xs border border-red-500/40 shrink-0 transition-all"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>إعادة المحاولة (Retry)</span>
        </button>
      )}
    </div>
  );
};
