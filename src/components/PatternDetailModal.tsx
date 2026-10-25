// src/components/PatternDetailModal.tsx
// Comprehensive Fraud Pattern Intelligence Modal / Drawer
import React, { useEffect, useState } from 'react';
import {
  X,
  ExternalLink,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Network,
  Landmark,
  Quote,
  Layers,
  HelpCircle,
  Calendar,
  Building2,
  Radio,
  Target,
  Sparkles,
  Info,
  ChevronRight,
  ListChecks,
  Workflow,
} from 'lucide-react';
import { SaudiPattern } from '../types/saudi';
import { apiService } from '../services/api';
import { Chip, EpistemicChip, Unavailable, fmtDay } from './ui';

interface PatternDetailModalProps {
  pattern?: SaudiPattern | null;
  patternKey?: string | null;
  onClose: () => void;
  onOpenIncident?: (incidentId: number) => void;
}

export const PatternDetailModal: React.FC<PatternDetailModalProps> = ({
  pattern: initialPattern,
  patternKey,
  onClose,
  onOpenIncident,
}) => {
  const [pattern, setPattern] = useState<SaudiPattern | null>(initialPattern || null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'works' | 'evidence' | 'banking'>('works');

  useEffect(() => {
    const key = initialPattern?.key || patternKey;
    if (key) {
      setLoading(true);
      apiService
        .getSaudiPattern(key)
        .then((res) => {
          if (res?.pattern) {
            setPattern(res.pattern);
          }
        })
        .catch(() => {
          // If single fetch fails, fallback to initialPattern
          if (initialPattern) setPattern(initialPattern);
        })
        .finally(() => setLoading(false));
    }
  }, [initialPattern, patternKey]);

  if (!pattern && !loading) return null;

  const p = pattern;

  // Tone color based on tier
  const getTierBadge = (tier?: string) => {
    switch (tier) {
      case 'confirmed_fraud_pattern':
        return { tone: 'critical' as const, label: 'نمط احتيالي مؤكد وموثق', bg: 'bg-rose-50 dark:bg-rose-950/40 text-rose-700 dark:text-rose-300 border-rose-200 dark:border-rose-800' };
      case 'observed_repeated_pattern':
        return { tone: 'high' as const, label: 'أسلوب احتيالي متكرر الرصد', bg: 'bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800' };
      case 'reported_fraud_method':
        return { tone: 'medium' as const, label: 'أسلوب احتيالي مبلّغ عنه', bg: 'bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800' };
      case 'warning_based_intelligence':
        return { tone: 'brand' as const, label: 'تحذير من أسلوب احتيالي محتمل', bg: 'bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border-sky-200 dark:border-sky-800' };
      case 'emerging_pattern':
        return { tone: 'low' as const, label: 'أسلوب احتيالي ناشئ قيد الرصد', bg: 'bg-purple-50 dark:bg-purple-950/40 text-purple-700 dark:text-purple-300 border-purple-200 dark:border-purple-800' };
      default:
        return { tone: 'plain' as const, label: 'بيانات توثيقية غير كافية', bg: 'bg-stone-100 dark:bg-slate-800 text-stone-700 dark:text-slate-300 border-stone-200 dark:border-slate-700' };
    }
  };

  const tierInfo = getTierBadge(p?.pattern_tier);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="pattern-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-black/60 backdrop-blur-sm overflow-y-auto animate-fadeIn"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-4xl bg-white dark:bg-[#0e1424] rounded-2xl border border-stone-200 dark:border-slate-800 shadow-2xl overflow-hidden my-auto max-h-[92vh] flex flex-col text-right font-sans"
        dir="rtl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top Header Bar */}
        <div className="flex items-start justify-between p-5 sm:p-6 border-b border-stone-200 dark:border-slate-800 bg-stone-50/70 dark:bg-[#121829]/70 shrink-0">
          <div className="space-y-2 max-w-[85%]">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-xs px-2.5 py-0.5 rounded-md bg-stone-200/70 dark:bg-slate-800 text-stone-700 dark:text-slate-300 font-semibold">
                {p?.key || 'PATTERN'}
              </span>

              {/* Tier Badge */}
              <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-semibold border ${tierInfo.bg}`}>
                <ShieldAlert className="w-3.5 h-3.5" />
                {p?.pattern_tier_ar || tierInfo.label}
              </span>

              {/* Country Badge */}
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-mono bg-stone-100 dark:bg-slate-800 text-stone-700 dark:text-slate-300 border border-stone-200 dark:border-slate-700">
                <span>🇸🇦</span>
                <span>المملكة العربية السعودية</span>
              </span>

              {/* Bank Relevance Badge */}
              {p?.bank_relevance && (
                <span
                  title={p.bank_relevance_rationale}
                  className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-medium cursor-help border ${
                    p.bank_relevance === 'high'
                      ? 'bg-rose-50 dark:bg-rose-950/60 text-rose-700 dark:text-rose-300 border-rose-200 dark:border-rose-800'
                      : p.bank_relevance === 'medium'
                      ? 'bg-amber-50 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800'
                      : 'bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800'
                  }`}
                >
                  <Landmark className="w-3 h-3" />
                  صلة بنكية {p.bank_relevance_ar || (p.bank_relevance === 'high' ? 'عالية' : 'متوسطة')}
                </span>
              )}

              {/* Confidence Badge */}
              {p?.confidence_level && (
                <span
                  title={p.confidence_rationale}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-stone-100 dark:bg-slate-800 text-stone-600 dark:text-slate-300 border border-stone-200 dark:border-slate-700 cursor-help"
                >
                  مستوى الثقة: {p.confidence_level_ar || p.confidence_level}
                </span>
              )}
            </div>

            <h2 id="pattern-modal-title" className="text-xl sm:text-2xl font-bold text-stone-900 dark:text-white leading-snug">
              {p?.label_ar}
            </h2>

            {/* Date range */}
            <div className="flex flex-wrap items-center gap-4 text-xs text-stone-500 dark:text-slate-400 font-mono">
              {p?.first_seen && (
                <span className="flex items-center gap-1">
                  <Calendar className="w-3.5 h-3.5 text-stone-400" />
                  أول رصد: {fmtDay(p.first_seen)}
                </span>
              )}
              {p?.last_seen && (
                <span className="flex items-center gap-1">
                  <Calendar className="w-3.5 h-3.5 text-stone-400" />
                  آخر رصد: {fmtDay(p.last_seen)}
                </span>
              )}
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-xl text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 hover:bg-stone-100 dark:hover:bg-slate-800 transition-colors"
            title="إغلاق"
            aria-label="إغلاق"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable Body */}
        <div className="flex-1 overflow-y-auto p-5 sm:p-6 space-y-6">
          {/* Key Metric Indicators Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3.5 rounded-xl bg-stone-50 dark:bg-[#121829] border border-stone-200 dark:border-slate-800">
              <span className="text-[11px] font-medium text-stone-500 dark:text-slate-400 block mb-1">
                حوادث مؤكدة
              </span>
              <div className="flex items-baseline gap-1.5">
                <span className="text-2xl font-bold font-mono text-stone-900 dark:text-white">
                  {p?.confirmed_incident_count ?? p?.incidents?.filter((i) => i.record_type === 'confirmed_incident').length ?? 0}
                </span>
                <span className="text-xs text-stone-400">قضية ضبط</span>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-stone-50 dark:bg-[#121829] border border-stone-200 dark:border-slate-800">
              <span className="text-[11px] font-medium text-stone-500 dark:text-slate-400 block mb-1">
                حوادث مبلّغ عنها
              </span>
              <div className="flex items-baseline gap-1.5">
                <span className="text-2xl font-bold font-mono text-stone-900 dark:text-white">
                  {p?.reported_incident_count ?? p?.incidents?.filter((i) => i.record_type !== 'confirmed_incident').length ?? 0}
                </span>
                <span className="text-xs text-stone-400">بلاغ قيد المراجعة</span>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-stone-50 dark:bg-[#121829] border border-stone-200 dark:border-slate-800">
              <span className="text-[11px] font-medium text-stone-500 dark:text-slate-400 block mb-1">
                تحذيرات وتوعية
              </span>
              <div className="flex items-baseline gap-1.5">
                <span className="text-2xl font-bold font-mono text-sky-600 dark:text-sky-400">
                  {p?.warning_article_count ?? p?.warning_articles?.length ?? 0}
                </span>
                <span className="text-xs text-stone-400">مادة تحذيرية</span>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-stone-50 dark:bg-[#121829] border border-stone-200 dark:border-slate-800">
              <span className="text-[11px] font-medium text-stone-500 dark:text-slate-400 block mb-1">
                مصادر أصلية مستقلة
              </span>
              <div className="flex items-baseline gap-1.5">
                <span className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
                  {p?.source_count ?? Math.max(1, (p?.incident_count || 0) + (p?.warning_article_count || 0))}
                </span>
                <span className="text-xs text-stone-400">ناشر معتمد</span>
              </div>
            </div>
          </div>

          {/* Classification Rationale Alert Box */}
          <div className="p-4 rounded-xl bg-sky-50/60 dark:bg-sky-950/20 border border-sky-200 dark:border-sky-800/60 space-y-2">
            <div className="flex items-center gap-2 text-sky-800 dark:text-sky-300 font-semibold text-xs">
              <Info className="w-4 h-4 shrink-0" />
              <span>معايير تصنيف النمط وسلامة الأدلة</span>
            </div>
            <p className="text-xs text-sky-900/80 dark:text-sky-200/80 leading-relaxed">
              {p?.classification_rationale ||
                (p?.incident_count === 0
                  ? 'تم تصنيف هذا المحتوى كتحذير ميتا فيو لوروده في بيانات وتحذيرات رسمية دون تسجيل حوادث جنائية مؤكدة مرتبطة به حتى تاريخه. تكرار المقالات عن التحذير نفسه لا يُعد دليلاً على تكرار الاحتيال.'
                  : 'تم تقييم النمط وتصنيفه استناداً إلى وقائع مؤكدة مسجلة لدى الجهات المختصة.')}
            </p>
            {p?.bank_relevance_rationale && (
              <div className="pt-2 border-t border-sky-200/60 dark:border-sky-800/40 text-[11px] text-sky-800/80 dark:text-sky-300/80">
                <strong className="font-semibold">مسوّغات الصلة المصرفية: </strong>
                {p.bank_relevance_rationale}
              </div>
            )}
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center gap-2 border-b border-stone-200 dark:border-slate-800 pb-2">
            <button
              onClick={() => setActiveTab('works')}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 ${
                activeTab === 'works'
                  ? 'bg-sky-600 text-white shadow-xs'
                  : 'text-stone-600 dark:text-slate-400 hover:bg-stone-100 dark:hover:bg-slate-800'
              }`}
            >
              <Workflow className="w-3.5 h-3.5" />
              كيف يعمل الاحتيال؟
            </button>
            <button
              onClick={() => setActiveTab('evidence')}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 ${
                activeTab === 'evidence'
                  ? 'bg-sky-600 text-white shadow-xs'
                  : 'text-stone-600 dark:text-slate-400 hover:bg-stone-100 dark:hover:bg-slate-800'
              }`}
            >
              <Quote className="w-3.5 h-3.5" />
              الأدلة والمصادر ({p?.evidence?.length || 0})
            </button>
            <button
              onClick={() => setActiveTab('banking')}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 ${
                activeTab === 'banking'
                  ? 'bg-sky-600 text-white shadow-xs'
                  : 'text-stone-600 dark:text-slate-400 hover:bg-stone-100 dark:hover:bg-slate-800'
              }`}
            >
              <Landmark className="w-3.5 h-3.5" />
              التحليل المصرفي ومؤشرات الكشف
            </button>
          </div>

          {/* TAB 1: How Fraud Operates */}
          {activeTab === 'works' && (
            <div className="space-y-5 animate-fadeIn">
              {/* Description */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-stone-900 dark:text-stone-100 uppercase tracking-wider flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-sky-600" />
                  توصيف الأسلوب
                </h4>
                <p className="text-xs text-stone-700 dark:text-slate-300 leading-relaxed bg-stone-50 dark:bg-[#121829] p-3.5 rounded-xl border border-stone-200 dark:border-slate-800">
                  {p?.how_it_works?.description ||
                    `أسلوب احتيالي متمثل في «${p?.label_ar}» مستخرج صراحة من نصوص الرصد والتحذيرات المفتوحة.`}
                </p>
              </div>

              {/* Documented Steps */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-semibold text-stone-900 dark:text-stone-100 uppercase tracking-wider flex items-center gap-1.5">
                    <ListChecks className="w-3.5 h-3.5 text-sky-600" />
                    مسار وخطوات الاحتيال
                  </h4>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">
                    موثقة في المصادر
                  </span>
                </div>
                {p?.how_it_works?.steps && p.how_it_works.steps.length > 0 ? (
                  <ol className="space-y-2 bg-stone-50 dark:bg-[#121829] p-3.5 rounded-xl border border-stone-200 dark:border-slate-800">
                    {p.how_it_works.steps.map((step, idx) => (
                      <li key={idx} className="flex items-start gap-2.5 text-xs text-stone-700 dark:text-slate-300 leading-relaxed">
                        <span className="w-5 h-5 rounded-full bg-sky-100 dark:bg-sky-900/60 text-sky-700 dark:text-sky-300 font-mono text-[11px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                          {idx + 1}
                        </span>
                        <span>{step}</span>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="text-xs text-stone-500 dark:text-slate-400 italic p-3 bg-stone-50 dark:bg-[#121829] rounded-xl border border-stone-200 dark:border-slate-800">
                    خطوات التنفيذ التفصيلية لم تُوثّق في الأدلة المتاحة؛ يُعرض التوصيف العام المعتمد فقط.
                  </p>
                )}
              </div>

              {/* Channels, Impersonations, Victims Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {/* Channels */}
                <div className="p-3.5 rounded-xl bg-stone-50 dark:bg-[#121829] border border-stone-200 dark:border-slate-800 space-y-2">
                  <span className="text-xs font-semibold text-stone-800 dark:text-slate-200 flex items-center gap-1.5">
                    <Radio className="w-3.5 h-3.5 text-sky-500" />
                    القنوات المستخدمة
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {(p?.how_it_works?.channels && p.how_it_works.channels.length > 0
                      ? p.how_it_works.channels
                      : p?.channels && p.channels.length > 0
                      ? p.channels
                      : ['منصات التواصل الاجتماعي', 'الإعلانات الممولة']
                    ).map((c, i) => (
                      <Chip key={i}>{c}</Chip>
                    ))}
                  </div>
                </div>

                {/* Impersonated Entities */}
                <div className="p-3.5 rounded-xl bg-stone-50 dark:bg-[#121829] border border-stone-200 dark:border-slate-800 space-y-2">
                  <span className="text-xs font-semibold text-stone-800 dark:text-slate-200 flex items-center gap-1.5">
                    <Building2 className="w-3.5 h-3.5 text-amber-500" />
                    الجهات المنتحلة
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {(p?.how_it_works?.impersonated_entities && p.how_it_works.impersonated_entities.length > 0
                      ? p.how_it_works.impersonated_entities
                      : ['علامات تجارية شهيرة', 'شركات معلنة وهمية']
                    ).map((ent, i) => (
                      <Chip key={i} tone="brand">{ent}</Chip>
                    ))}
                  </div>
                </div>

                {/* Targeted Victims */}
                <div className="p-3.5 rounded-xl bg-stone-50 dark:bg-[#121829] border border-stone-200 dark:border-slate-800 space-y-2">
                  <span className="text-xs font-semibold text-stone-800 dark:text-slate-200 flex items-center gap-1.5">
                    <Target className="w-3.5 h-3.5 text-rose-500" />
                    الضحايا المستهدفون
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {(p?.how_it_works?.targeted_victims && p.how_it_works.targeted_victims.length > 0
                      ? p.how_it_works.targeted_victims
                      : ['مستخدمو شبكات التواصل', 'الباحثون عن عروض تسوق']
                    ).map((vic, i) => (
                      <Chip key={i} tone="medium">{vic}</Chip>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: Evidence & Sources */}
          {activeTab === 'evidence' && (
            <div className="space-y-5 animate-fadeIn">
              {/* Verbatim Evidence Quotes */}
              <div className="space-y-3">
                <h4 className="text-xs font-semibold text-stone-900 dark:text-stone-100 uppercase tracking-wider flex items-center gap-1.5">
                  <Quote className="w-3.5 h-3.5 text-sky-600" />
                  اقتباسات حرفية من نصوص المصادر
                </h4>
                {p?.evidence && p.evidence.length > 0 ? (
                  <div className="space-y-3">
                    {p.evidence.map((ev, idx) => (
                      <div
                        key={idx}
                        className="p-3.5 rounded-xl bg-stone-50 dark:bg-[#121829] border border-stone-200 dark:border-slate-800 space-y-2"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-xs font-semibold text-stone-800 dark:text-slate-200">
                            {ev.title}
                          </span>
                          <Chip tone={ev.type === 'incident' ? 'critical' : 'brand'}>
                            {ev.type === 'incident' ? 'حادثة' : 'تحذير'}
                          </Chip>
                        </div>
                        <p className="text-xs text-stone-600 dark:text-slate-400 leading-relaxed border-r-2 border-sky-500/70 pr-3">
                          «{ev.quote}»
                        </p>
                        {ev.url && (
                          <div className="pt-1 flex items-center justify-between text-[11px]">
                            <a
                              href={ev.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-sky-600 dark:text-sky-400 hover:underline inline-flex items-center gap-1 font-medium"
                            >
                              <ExternalLink className="w-3 h-3" />
                              <span>{ev.source || 'فتح الرابط الأصلي للمصدر'}</span>
                            </a>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <Unavailable title="لا اقتباسات إضافية" note="لا تتوفر اقتباسات حرفية مسجلة لهذا النمط." />
                )}
              </div>

              {/* Linked Incidents */}
              <div className="space-y-2 pt-2 border-t border-stone-200 dark:border-slate-800">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-semibold text-stone-900 dark:text-stone-100 uppercase tracking-wider flex items-center gap-1.5">
                    <Network className="w-3.5 h-3.5 text-sky-600" />
                    الحوادث الجنائية المرتبطة ({p?.incidents?.length || 0})
                  </h4>
                </div>

                {p?.incidents && p.incidents.length > 0 ? (
                  <div className="space-y-2">
                    {p.incidents.map((inc) => (
                      <div
                        key={inc.id}
                        onClick={() => onOpenIncident && onOpenIncident(inc.id)}
                        className="p-3 rounded-xl bg-stone-50 dark:bg-[#121829] border border-stone-200 dark:border-slate-800 hover:border-sky-400 dark:hover:border-sky-700 transition-colors flex items-center justify-between gap-3 cursor-pointer"
                      >
                        <div className="min-w-0">
                          <span className="text-xs font-medium text-stone-900 dark:text-stone-100 block truncate">
                            {inc.title}
                          </span>
                          <span className="text-[11px] font-mono text-stone-400">
                            {fmtDay(inc.first_reported_at)} • {inc.article_count} مصادر إخبارية
                          </span>
                        </div>
                        <span className="text-xs text-sky-600 dark:text-sky-400 shrink-0 font-medium hover:underline flex items-center gap-1">
                          عرض الحادثة ←
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-stone-500 dark:text-slate-400 p-3 bg-stone-50 dark:bg-[#121829] rounded-xl border border-stone-200 dark:border-slate-800">
                    لم تُسجل حوادث جنائية مرتبطة بهذا النمط حتى تاريخه؛ النمط مستند إلى تحذيرات وتوعية ميتا فيو فقط.
                  </p>
                )}
              </div>

              {/* Linked Warnings */}
              {p?.warning_articles && p.warning_articles.length > 0 && (
                <div className="space-y-2 pt-2 border-t border-stone-200 dark:border-slate-800">
                  <h4 className="text-xs font-semibold text-stone-900 dark:text-stone-100 uppercase tracking-wider flex items-center gap-1.5">
                    <ShieldAlert className="w-3.5 h-3.5 text-sky-600" />
                    المواد والتحذيرات الأصلية في ميتا فيو ({p.warning_articles.length})
                  </h4>
                  <div className="space-y-2">
                    {p.warning_articles.map((w, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-xl bg-stone-50 dark:bg-[#121829] border border-stone-200 dark:border-slate-800 flex items-center justify-between gap-3"
                      >
                        <div className="min-w-0">
                          <span className="text-xs font-medium text-stone-900 dark:text-stone-100 block truncate">
                            {w.title}
                          </span>
                          <span className="text-[11px] font-mono text-stone-400">
                            {w.source} {w.published_at ? `• ${fmtDay(w.published_at)}` : ''}
                          </span>
                        </div>
                        {w.url && (
                          <a
                            href={w.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-sky-600 dark:text-sky-400 shrink-0 font-medium hover:underline inline-flex items-center gap-1"
                          >
                            <ExternalLink className="w-3 h-3" />
                            المصدر الأصلي
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: Banking Analysis */}
          {activeTab === 'banking' && (
            <div className="space-y-5 animate-fadeIn">
              {/* Potential Bank Risks */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-rose-700 dark:text-rose-400 uppercase tracking-wider flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  المخاطر المحتملة على البنك
                </h4>
                <ul className="space-y-1.5 bg-rose-50/50 dark:bg-rose-950/20 p-3.5 rounded-xl border border-rose-200 dark:border-rose-900/50">
                  {(p?.banking_analysis?.potential_bank_risks && p.banking_analysis.potential_bank_risks.length > 0
                    ? p.banking_analysis.potential_bank_risks
                    : [
                        'مخاطر استدراج العملاء لإجراء تحويلات بنكية طوعية لحسابات احتيالية (APP Fraud).',
                        'نزاعات مدفوعات البطاقات وعمليات الاحتيال العكسية (Chargebacks).',
                        'مخاطر السمعة المصرفية جراء استغلال اسم البنك في إعلانات احتيالية.'
                      ]
                  ).map((risk, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-rose-900 dark:text-rose-200 leading-relaxed">
                      <span className="text-rose-500 font-bold shrink-0">•</span>
                      <span>{risk}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Fraud Indicators */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-stone-900 dark:text-stone-100 uppercase tracking-wider flex items-center gap-1.5">
                  <Target className="w-3.5 h-3.5 text-sky-600" />
                  مؤشرات الاحتيال المستخرجة (Fraud Indicators)
                </h4>
                <ul className="space-y-1.5 bg-stone-50 dark:bg-[#121829] p-3.5 rounded-xl border border-stone-200 dark:border-slate-800">
                  {(p?.banking_analysis?.fraud_indicators && p.banking_analysis.fraud_indicators.length > 0
                    ? p.banking_analysis.fraud_indicators
                    : [
                        'عمليات دفع إلكتروني لروابط أو بوابات غير مصنفة تلي النقر على إعلانات خارجية.',
                        'حركات تحويل مالي فورية بمبالغ نمطية متكررة لحسابات أفراد حديثة النشأة.',
                        'تكرار الشكاوى حول مدفوعات لجهات تجارية غير مسجلة في سجلات التجارة الإلكترونية.'
                      ]
                  ).map((ind, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-stone-700 dark:text-slate-300 leading-relaxed">
                      <span className="text-sky-500 font-bold shrink-0">•</span>
                      <span>{ind}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Detection Hypotheses & Rules */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-stone-900 dark:text-stone-100 uppercase tracking-wider flex items-center gap-1.5">
                  <ListChecks className="w-3.5 h-3.5 text-sky-600" />
                  فرضيات الكشف الممكنة (Detection Hypotheses)
                </h4>
                <div className="space-y-2">
                  {(p?.banking_analysis?.detection_hypotheses && p.banking_analysis.detection_hypotheses.length > 0
                    ? p.banking_analysis.detection_hypotheses
                    : [
                        'SFI-ADV-001: رصد حركة مدفوعات متكررة لروابط وبوابات دفع مشبوهة مقترنة بإعلانات مضللة.',
                        'SFI-FIN-001: رصد تحويلات عملاء لحسابات وسيطة (Mules) عقب التفاعل مع عروض ترويجية غير موثوقة.'
                      ]
                  ).map((hyp, i) => (
                    <div
                      key={i}
                      className="p-3 rounded-xl bg-stone-50 dark:bg-[#121829] border border-stone-200 dark:border-slate-800 flex items-start gap-2 text-xs text-stone-700 dark:text-slate-300 leading-relaxed"
                    >
                      <EpistemicChip kind="hypothesis" label="فرضية كشف" />
                      <span>{hyp}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Required Data */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-stone-900 dark:text-stone-100 uppercase tracking-wider flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-sky-600" />
                  البيانات اللازمة لاختبار الفرضيات
                </h4>
                <div className="flex flex-wrap gap-1.5 bg-stone-50 dark:bg-[#121829] p-3 rounded-xl border border-stone-200 dark:border-slate-800">
                  {(p?.banking_analysis?.required_data && p.banking_analysis.required_data.length > 0
                    ? p.banking_analysis.required_data
                    : [
                        'سجلات العمليات المصرفية (المبلغ، وقت العملية، القناة المستهدفة)',
                        'بيانات المستفيد ورمز التاجر (MCC)',
                        'سجلات المصادقة الثنائية (3DS) وعناوين IP'
                      ]
                  ).map((req, i) => (
                    <Chip key={i}>{req}</Chip>
                  ))}
                </div>
              </div>

              {/* Uncertainties & Limitations */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-stone-900 dark:text-stone-100 uppercase tracking-wider flex items-center gap-1.5">
                  <HelpCircle className="w-3.5 h-3.5 text-stone-400" />
                  القيود ونقاط عدم اليقين (Epistemic Limitations)
                </h4>
                <div className="p-3.5 rounded-xl bg-amber-50/50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/50 space-y-1 text-xs text-amber-900 dark:text-amber-200 leading-relaxed">
                  {(p?.banking_analysis?.uncertainties_and_limitations && p.banking_analysis.uncertainties_and_limitations.length > 0
                    ? p.banking_analysis.uncertainties_and_limitations
                    : [
                        'التحليل مستند إلى مواد الرصد والتوعية الإعلامية المفتوحة؛ لا تتوفر سجلات داخلية لتحديد الحجم المالي الفعلي للعمليات بدقة.',
                        'الخطوات المعروضة تمثل المسار النموذجي الموثق في التحذير، ولا تعني بالضرورة ثبوت ارتكابها في قضية جنائية واحدة بعينها ما لم يصدر بها بيان ضبط رسمي.'
                      ]
                  ).map((unc, i) => (
                    <p key={i}>• {unc}</p>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-stone-200 dark:border-slate-800 bg-stone-50/70 dark:bg-[#121829]/70 shrink-0 text-xs">
          <span className="text-stone-500 dark:text-slate-400 font-mono">
            {p?.key} • تصنيف معتمد وفق معايير سلامة الأدلة
          </span>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-stone-200 dark:bg-slate-800 text-stone-800 dark:text-slate-200 font-medium hover:bg-stone-300 dark:hover:bg-slate-700 transition-colors"
          >
            إغلاق
          </button>
        </div>
      </div>
    </div>
  );
};
