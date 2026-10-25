// src/components/FraudNewsDetailModal.tsx
import React, { useEffect, useState } from 'react';
import {
  X,
  ExternalLink,
  Building2,
  Globe,
  Landmark,
  ShieldAlert,
  Calendar,
  Layers,
  Sparkles,
  CheckCircle2,
} from 'lucide-react';
import { SaudiFraudItem } from '../types/saudi';
import { apiService } from '../services/api';
import { Chip, EpistemicChip, fmtDay } from './ui';
import { getNewsImage } from '../utils/newsImages';

const COUNTRY_MAP: Record<string, { name: string; flag: string }> = {
  SA: { name: 'المملكة العربية السعودية', flag: '🇸🇦' },
  US: { name: 'الولايات المتحدة', flag: '🇺🇸' },
  GB: { name: 'المملكة المتحدة', flag: '🇬🇧' },
  CA: { name: 'كندا', flag: '🇨🇦' },
  AU: { name: 'أستراليا', flag: '🇦🇺' },
  EG: { name: 'مصر', flag: '🇪🇬' },
  AE: { name: 'الإمارات العربية المتحدة', flag: '🇦🇪' },
  NL: { name: 'هولندا', flag: '🇳🇱' },
  Global: { name: 'دولي / عالمي', flag: '🌐' },
};

interface FraudNewsDetailModalProps {
  item: SaudiFraudItem | null;
  onClose: () => void;
}

export const FraudNewsDetailModal: React.FC<FraudNewsDetailModalProps> = ({
  item: initialItem,
  onClose,
}) => {
  const [item, setItem] = useState<SaudiFraudItem | null>(initialItem);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setItem(initialItem);
    if (initialItem?.id && !initialItem.content_excerpt) {
      setLoading(true);
      apiService
        .getSaudiNewsItem(initialItem.id)
        .then((full) => setItem(full))
        .catch(() => {})
        .finally(() => setLoading(false));
    }
  }, [initialItem]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!item) return null;

  const countryInfo = (item.country && COUNTRY_MAP[item.country]) || {
    name: item.country || (item.is_saudi ? 'المملكة العربية السعودية' : 'دولي / عالمي'),
    flag: item.is_saudi ? '🇸🇦' : '🌐',
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-stone-900/60 dark:bg-black/75 backdrop-blur-sm flex items-start justify-center p-3 sm:p-5 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="w-full max-w-3xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 rounded-xl shadow-xl overflow-hidden my-6 sm:my-10 space-y-0 text-right"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-5 border-b border-stone-100 dark:border-slate-800/80 flex items-start justify-between gap-4">
          <div className="space-y-2 min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <Chip tone="brand">
                {countryInfo.flag} {countryInfo.name}
              </Chip>
              <Chip tone="critical">{item.fraud_category_ar || item.fraud_category}</Chip>
              {item.bank_relevance && (
                <Chip tone={item.bank_relevance === 'high' ? 'critical' : 'medium'}>
                  صلة بنكية {item.bank_relevance === 'high' ? 'عالية' : 'متوسطة'}
                </Chip>
              )}
              <span className="text-xs font-mono text-stone-500 dark:text-slate-400">
                {fmtDay(item.published_at || item.fetched_at)}
              </span>
            </div>
            <h2 className="text-base sm:text-lg font-bold text-stone-900 dark:text-stone-100 leading-snug">
              {item.title}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 hover:bg-stone-100 dark:hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-5 space-y-5 text-sm">
          {/* Photo Banner */}
          <div className="w-full h-44 sm:h-56 rounded-xl overflow-hidden border border-stone-200 dark:border-slate-800 bg-stone-100 dark:bg-slate-900 shadow-xs">
            <img
              src={getNewsImage({
                title: item.title,
                summary: item.summary,
                fraud_category: item.fraud_category,
              })}
              alt=""
              className="w-full h-full object-cover"
              onError={(e) => {
                (e.target as HTMLImageElement).src =
                  'https://images.unsplash.com/photo-1559526324-4b87b5e36e44?auto=format&fit=crop&w=800&q=80';
              }}
            />
          </div>

          {/* Metadata Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-3.5 rounded-lg bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-800 text-xs">
            <div>
              <span className="text-stone-500 dark:text-slate-400 block mb-0.5">المصدر</span>
              <span className="font-semibold text-stone-900 dark:text-stone-100 flex items-center gap-1">
                <Building2 className="w-3.5 h-3.5 text-sky-600" />
                {item.source || 'غير محدد'}
              </span>
            </div>
            <div>
              <span className="text-stone-500 dark:text-slate-400 block mb-0.5">النطاق الإخباري</span>
              <span className="font-mono text-stone-700 dark:text-slate-300 truncate block">
                {item.domain || '—'}
              </span>
            </div>
            <div>
              <span className="text-stone-500 dark:text-slate-400 block mb-0.5">الدولة</span>
              <span className="font-semibold text-stone-900 dark:text-stone-100 flex items-center gap-1">
                <Globe className="w-3.5 h-3.5 text-sky-600" />
                {countryInfo.name}
              </span>
            </div>
            <div>
              <span className="text-stone-500 dark:text-slate-400 block mb-0.5">نوع المحتوى</span>
              <span className="font-semibold text-stone-900 dark:text-stone-100">
                {item.content_type_ar || 'خبر ميتا فيو'}
              </span>
            </div>
          </div>

          {/* Summary / Lead */}
          {item.summary && (
            <div className="space-y-1.5">
              <h4 className="text-xs font-semibold text-stone-500 dark:text-slate-400 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5" />
                ملخص ميتا فيو
              </h4>
              <p className="text-stone-800 dark:text-stone-200 leading-relaxed bg-stone-50/70 dark:bg-slate-900/60 p-3 rounded-lg border border-stone-200/70 dark:border-slate-800">
                {item.summary}
              </p>
            </div>
          )}

          {/* Modus Operandi / Analytical Assessment */}
          {item.modus_operandi && (
            <div className="space-y-1.5">
              <h4 className="text-xs font-semibold text-stone-500 dark:text-slate-400 flex items-center gap-1.5">
                <ShieldAlert className="w-3.5 h-3.5 text-amber-600" />
                طريقة التنفيذ ومسار الاحتيال
              </h4>
              <p className="text-stone-700 dark:text-slate-300 text-xs leading-relaxed bg-amber-50/40 dark:bg-amber-950/20 p-3 rounded-lg border border-amber-200/50 dark:border-amber-900/40">
                {item.modus_operandi}
              </p>
            </div>
          )}

          {/* Detection Rule (if available) */}
          {item.detection_rule && (
            <div className="space-y-2 p-3.5 rounded-lg bg-sky-50/50 dark:bg-sky-950/20 border border-sky-200/60 dark:border-sky-900/40">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-sky-800 dark:text-sky-300 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5" />
                  قاعدة الرصد البنكية المقترحة: {item.detection_rule.name}
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-100 dark:bg-sky-900/60 text-sky-700 dark:text-sky-300">
                  {item.detection_rule.rule_id}
                </span>
              </div>
              <p className="text-xs text-stone-700 dark:text-slate-300 leading-relaxed">
                <strong className="font-semibold text-stone-900 dark:text-stone-100">الفرضية: </strong>
                {item.detection_rule.hypothesis}
              </p>
              {item.detection_rule.logic && (
                <div className="p-2 rounded bg-white dark:bg-slate-900 border border-stone-200 dark:border-slate-800 text-[11px] font-mono text-stone-800 dark:text-slate-200 overflow-x-auto" dir="ltr">
                  {item.detection_rule.logic}
                </div>
              )}
            </div>
          )}

          {/* Excerpt if present */}
          {item.content_excerpt && item.content_excerpt.length > 50 && (
            <div className="space-y-1.5">
              <h4 className="text-xs font-semibold text-stone-500 dark:text-slate-400">
                مقتطف من المقال
              </h4>
              <p className="text-xs text-stone-600 dark:text-slate-400 leading-relaxed font-serif line-clamp-4 bg-stone-50 dark:bg-slate-900/50 p-3 rounded-lg border border-stone-200 dark:border-slate-800">
                {item.content_excerpt}
              </p>
            </div>
          )}

          {/* Action button: Original URL */}
          {item.original_url && (
            <div className="pt-2 flex items-center justify-between">
              <a
                href={item.original_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-700 text-white font-medium text-xs shadow-sm transition-colors"
              >
                <span>فتح المقال والمصدر الأصلي</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
              <span className="text-xs text-stone-400">
                المصدر الأصلي: {item.source}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
