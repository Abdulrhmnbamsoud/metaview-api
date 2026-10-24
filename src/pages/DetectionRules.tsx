// src/pages/DetectionRules.tsx
// صفحة قواعد الكشف: جدول قابل للبحث، مع عرض المنطق والافتراضات، ودون الادعاء بفعالية القواعد غير المختبرة
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Search,
  X,
  ExternalLink,
  FlaskConical,
  AlertCircle,
  CheckCircle2,
  FileCode,
  ShieldCheck,
  ChevronLeft,
} from 'lucide-react';
import { apiService } from '../services/api';
import { SaudiRulesResponse, SaudiRule } from '../types/saudi';
import { Chip, EpistemicChip, LoadingBlock, ErrorBlock, Unavailable, fmtDay } from '../components/ui';

export const DetectionRules: React.FC = () => {
  const [data, setData] = useState<SaudiRulesResponse | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedRule, setSelectedRule] = useState<SaudiRule | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await apiService.getSaudiRules());
    } catch (e: any) {
      setError(e?.message || 'تعذر جلب قواعد الكشف');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const categories = useMemo(() => {
    if (!data) return [];
    const set = new Set(data.rules.map((r) => r.category_ar));
    return Array.from(set).filter(Boolean);
  }, [data]);

  const filteredRules = useMemo(() => {
    if (!data) return [];
    return data.rules.filter((r) => {
      const matchCat = selectedCategory === 'all' || r.category_ar === selectedCategory;
      const q = searchQuery.trim().toLowerCase();
      const matchQuery =
        !q ||
        r.name.toLowerCase().includes(q) ||
        r.rule_id.toLowerCase().includes(q) ||
        r.hypothesis.toLowerCase().includes(q) ||
        r.logic.toLowerCase().includes(q);
      return matchCat && matchQuery;
    });
  }, [data, selectedCategory, searchQuery]);

  if (loading && !data) return <LoadingBlock label="جارٍ تحميل قواعد الكشف والفرضيات…" />;
  if (error && !data) return <ErrorBlock message={error} onRetry={load} />;
  if (!data) return null;

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="space-y-1 pb-3 border-b border-stone-200 dark:border-slate-800">
        <h1 className="text-2xl font-bold tracking-tight text-stone-900 dark:text-stone-100">
          قواعد الكشف المقترحة
        </h1>
        <p className="text-sm text-stone-600 dark:text-slate-400">
          قواعد مصممة بناءً على وقائع الاحتيال المرصودة؛ تعتبر فرضيات قابلة للاختبار ولا تُعتمد حتى تُجرَّب على بيانات بنكية حقيقية
        </p>
      </div>

      {/* Honest Scientific Caution Banner */}
      <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-amber-50/70 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/40 text-xs text-amber-900 dark:text-amber-300 leading-relaxed">
        <FlaskConical className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
        <span>
          <strong>تنبيه منهجي:</strong> جميع القواعد الحالية مقترحة بناءً على الأنماط والأدلة (Hypotheses). لم تُختبر على أنظمة بنكية فعلية، وبالتالي لا تُعرض كقواعد ناجحة أو ذات معدل دقة مثبت حتى اكتمال الاختبارات في بيئة البنك.
        </span>
      </div>

      {/* Search & Category Filter */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        <div className="relative flex-1">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="بحث بالاسم، المعرف، الفرضية، أو منطق الكشف..."
            className="w-full bg-white dark:bg-[#111726] border border-stone-300 dark:border-slate-800 rounded-xl py-2.5 pr-10 pl-3 text-xs text-stone-900 dark:text-stone-100 placeholder:text-stone-400 dark:placeholder:text-slate-500 shadow-sm focus:border-sky-500 outline-none"
          />
          <Search className="w-4 h-4 text-stone-400 dark:text-slate-500 absolute top-1/2 -translate-y-1/2 right-3 pointer-events-none" />
        </div>

        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="bg-white dark:bg-[#111726] border border-stone-300 dark:border-slate-800 rounded-xl px-3 py-2 text-xs text-stone-700 dark:text-slate-300 shadow-sm outline-none"
        >
          <option value="all">كافة التصنيفات ({data.rules.length})</option>
          {categories.map((cat) => (
            <option key={cat} value={cat}>
              {cat}
            </option>
          ))}
        </select>
      </div>

      {/* Rules Table */}
      {filteredRules.length === 0 ? (
        <Unavailable
          title="لا توجد قواعد مطابقة"
          note="جرب تغيير معايير البحث أو اختيار تصنيف آخر."
        />
      ) : (
        <div className="panel overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-right text-xs">
              <thead>
                <tr className="bg-stone-50 dark:bg-slate-900/60 text-stone-500 dark:text-slate-400 border-b border-stone-200 dark:border-slate-800 font-mono text-[11px]">
                  <th className="py-3 px-4 font-medium">معرف القاعدة</th>
                  <th className="py-3 px-4 font-medium">اسم القاعدة والهدف</th>
                  <th className="py-3 px-4 font-medium">البيانات المطلوبة</th>
                  <th className="py-3 px-4 font-medium">الأدلة المرتبطة</th>
                  <th className="py-3 px-4 font-medium">حالة الاختبار</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100 dark:divide-slate-800/60">
                {filteredRules.map((r) => {
                  const tested = r.test_results && r.test_results.length > 0;
                  return (
                    <tr
                      key={r.rule_id}
                      onClick={() => setSelectedRule(r)}
                      className="hover:bg-stone-50/80 dark:hover:bg-slate-800/40 transition-colors cursor-pointer group"
                    >
                      <td className="py-3.5 px-4 font-mono font-bold text-sky-700 dark:text-sky-300 whitespace-nowrap">
                        {r.rule_id}
                      </td>
                      <td className="py-3.5 px-4 max-w-sm">
                        <div className="font-semibold text-stone-900 dark:text-stone-100 group-hover:text-sky-600 dark:group-hover:text-sky-400 transition-colors">
                          {r.name}
                        </div>
                        <div className="text-stone-500 dark:text-slate-400 text-[11px] line-clamp-1 mt-0.5">
                          {r.hypothesis}
                        </div>
                      </td>
                      <td className="py-3.5 px-4">
                        <div className="flex flex-wrap gap-1 max-w-xs">
                          {r.required_data.slice(0, 2).map((d) => (
                            <span
                              key={d}
                              className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-stone-100 dark:bg-slate-800 text-stone-600 dark:text-slate-300"
                            >
                              {d}
                            </span>
                          ))}
                          {r.required_data.length > 2 && (
                            <span className="text-[10px] text-stone-400 font-mono">
                              +{r.required_data.length - 2}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-3.5 px-4 whitespace-nowrap text-stone-600 dark:text-slate-300 font-mono text-[11px]">
                        {r.evidence.incident_count > 0 && (
                          <span className="text-sky-600 dark:text-sky-400">
                            {r.evidence.incident_count} حوادث
                          </span>
                        )}
                        {r.evidence.warning_article_count > 0 && (
                          <span className="text-stone-400 mr-1.5">
                            • {r.evidence.warning_article_count} تحذير
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 whitespace-nowrap">
                        {tested ? (
                          <span className="inline-flex items-center gap-1 text-[11px] font-mono text-emerald-700 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-300">
                            <CheckCircle2 className="w-3 h-3" />
                            مُختبرة
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[11px] font-mono text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/40 px-2 py-0.5 rounded border border-amber-300">
                            فرضية مقترحة
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Rule Detail Modal / Drawer */}
      {selectedRule && (
        <div
          className="fixed inset-0 z-50 bg-stone-900/60 dark:bg-black/75 backdrop-blur-sm flex items-start justify-center p-3 sm:p-5 overflow-y-auto"
          onClick={() => setSelectedRule(null)}
        >
          <div
            className="w-full max-w-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 rounded-xl shadow-xl overflow-hidden my-8 space-y-0 text-right"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="p-5 border-b border-stone-200 dark:border-slate-800 bg-stone-50/60 dark:bg-slate-900/40 flex items-start justify-between gap-4">
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-sky-100 dark:bg-sky-950 text-sky-800 dark:text-sky-300">
                    {selectedRule.rule_id}
                  </span>
                  <Chip tone="plain">{selectedRule.category_ar}</Chip>
                  <Chip tone={selectedRule.test_results?.length ? 'low' : 'medium'}>
                    {selectedRule.test_results?.length ? 'مُختبرة' : 'فرضية غير مُثبتة'}
                  </Chip>
                </div>
                <h2 className="text-base font-bold text-stone-900 dark:text-stone-100 leading-snug">
                  {selectedRule.name}
                </h2>
              </div>
              <button
                onClick={() => setSelectedRule(null)}
                className="p-1.5 rounded-lg text-stone-400 hover:text-stone-700 dark:hover:text-stone-200"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Body */}
            <div className="p-5 space-y-5 text-xs">
              {/* Hypothesis */}
              <div className="space-y-1.5">
                <div className="font-semibold text-stone-900 dark:text-stone-100 text-xs">
                  الفرضية التشغيلية (Hypothesis)
                </div>
                <p className="text-stone-700 dark:text-slate-300 leading-relaxed bg-stone-50 dark:bg-slate-900/40 p-3 rounded-lg border border-stone-200 dark:border-slate-800">
                  {selectedRule.hypothesis}
                </p>
              </div>

              {/* Logic */}
              <div className="space-y-1.5">
                <div className="font-semibold text-stone-900 dark:text-stone-100 text-xs">
                  منطق الكشف المقترح (Detection Logic)
                </div>
                <div
                  className="p-3 rounded-lg bg-stone-900 text-sky-400 dark:bg-slate-950 dark:text-sky-300 font-mono text-[11px] leading-relaxed overflow-x-auto"
                  dir="ltr"
                >
                  {selectedRule.logic}
                </div>
              </div>

              {/* Required Data */}
              <div className="space-y-1.5">
                <div className="font-semibold text-stone-900 dark:text-stone-100 text-xs">
                  حقول وبيانات البنك المطلوبة للتنفيذ
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {selectedRule.required_data.map((d) => (
                    <Chip key={d} tone="brand">
                      {d}
                    </Chip>
                  ))}
                </div>
              </div>

              {/* Validation Plan */}
              {selectedRule.validation_plan?.length > 0 && (
                <div className="space-y-1.5">
                  <div className="font-semibold text-stone-900 dark:text-stone-100 text-xs">
                    خطة التحقق والتجربة
                  </div>
                  <ul className="space-y-1 text-stone-600 dark:text-slate-400 pr-4 list-disc text-[11px]">
                    {selectedRule.validation_plan.map((step, idx) => (
                      <li key={idx}>{step}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Supporting Incidents */}
              <div className="space-y-2 pt-2 border-t border-stone-200 dark:border-slate-800">
                <div className="font-semibold text-stone-900 dark:text-stone-100 text-xs">
                  الحوادث والأدلة الواقعية الداعمة لهذه الفرضية
                </div>
                {selectedRule.evidence.incidents.length > 0 ? (
                  <div className="space-y-2">
                    {selectedRule.evidence.incidents.map((inc) => (
                      <div
                        key={inc.id}
                        className="p-2.5 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200 dark:border-slate-800 flex items-center justify-between gap-2"
                      >
                        <span className="text-stone-800 dark:text-stone-200 font-medium">
                          {inc.title}
                        </span>
                        <Link
                          to={`/explore?q=${encodeURIComponent(inc.title)}`}
                          onClick={() => setSelectedRule(null)}
                          className="text-sky-600 dark:text-sky-400 hover:underline shrink-0 text-[11px]"
                        >
                          عرض الحادثة ←
                        </Link>
                      </div>
                    ))}
                  </div>
                ) : (
                  <Unavailable
                    title="لا توجد حوادث مسندة حالياً"
                    note="تم اشتقاق هذه الفرضية من تحذيرات وتوعيات إعلامية."
                  />
                )}
              </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-stone-200 dark:border-slate-800 bg-stone-50/80 dark:bg-slate-900/60 flex items-center justify-between">
              <span className="text-[11px] text-stone-500 dark:text-slate-400 font-mono">
                الحالة: فرضية غير مفعلة تلقائياً
              </span>
              <button
                onClick={() => setSelectedRule(null)}
                className="px-4 py-1.5 rounded-lg bg-stone-800 dark:bg-slate-200 text-white dark:text-stone-900 font-medium text-xs hover:bg-stone-900 transition-colors"
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

export default DetectionRules;
