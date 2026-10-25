// src/components/IncidentDetailModal.tsx
// شاشة تفاصيل الحادثة: تجيب فوراً على الأسئلة الخمسة مع الإفصاح التدريجي
import React, { useEffect, useState } from 'react';
import {
  X,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  MapPin,
  Building2,
  AlertTriangle,
  CheckCircle2,
  FileText,
  Workflow,
  Landmark,
  Eye,
  ShieldCheck,
  Share2,
  Bookmark,
  MessageSquare,
  Send,
  Trash2,
} from 'lucide-react';
import { SaudiIncident, IncidentArticleRef } from '../types/saudi';
import { AnalystNote } from '../types/auth';
import { apiService } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { Chip, EpistemicChip, Unavailable, fmtDay, fmtDateTime } from './ui';

interface IncidentDetailModalProps {
  incident?: SaudiIncident | null;
  incidentId?: number | null;
  onClose: () => void;
}

export const IncidentDetailModal: React.FC<IncidentDetailModalProps> = ({
  incident: initialIncident,
  incidentId,
  onClose,
}) => {
  const { user, can } = useAuth();
  const [incident, setIncident] = useState<SaudiIncident | null>(initialIncident || null);
  const [loading, setLoading] = useState(false);
  const [showRawFacts, setShowRawFacts] = useState(false);
  const [showAllArticles, setShowAllArticles] = useState(false);

  // Watchlist & Analyst Notes State
  const [isSaved, setIsSaved] = useState(false);
  const [notes, setNotes] = useState<AnalystNote[]>([]);
  const [newNoteContent, setNewNoteContent] = useState('');
  const [newNoteType, setNewNoteType] = useState<'analysis' | 'classification_correction' | 'missing_info' | 'conflicting_info'>('analysis');
  const [savingNote, setSavingNote] = useState(false);
  const [feedbackMsg, setFeedbackMsg] = useState<string | null>(null);

  useEffect(() => {
    const targetId = initialIncident?.id || incidentId;
    if (targetId) {
      setLoading(true);
      apiService
        .getSaudiIncident(targetId)
        .then((full) => {
          setIncident(full);
          if (can('incident.annotate')) {
            apiService.getIncidentNotes(targetId).then((res) => setNotes(res.notes || []));
          }
        })
        .catch(() => {})
        .finally(() => setLoading(false));
    }
  }, [initialIncident, incidentId]);

  const handleToggleWatchlist = async () => {
    if (!incident) return;
    try {
      if (isSaved) {
        await apiService.unsaveIncident(incident.id);
        setIsSaved(false);
        setFeedbackMsg('تمت إزالة الحادثة من قائمة المتابعة.');
      } else {
        await apiService.saveIncident(incident.id, 'مضافة من نافذة التفاصيل');
        setIsSaved(true);
        setFeedbackMsg('تم حفظ الحادثة في قائمة متابعتك الشخصية.');
      }
      setTimeout(() => setFeedbackMsg(null), 3000);
    } catch (err: any) {
      setFeedbackMsg(err.message || 'تعذر تحديث قائمة المتابعة.');
    }
  };

  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!incident || !newNoteContent.trim()) return;
    setSavingNote(true);
    try {
      const added = await apiService.addIncidentNote(incident.id, {
        note_type: newNoteType,
        content: newNoteContent.trim(),
      });
      setNotes((prev) => [added, ...prev]);
      setNewNoteContent('');
      setFeedbackMsg('تم حفظ الملاحظة التحليلية بنجاح.');
      setTimeout(() => setFeedbackMsg(null), 3000);
    } catch (err: any) {
      setFeedbackMsg(err.message || 'تعذر إضافة الملاحظة.');
    } finally {
      setSavingNote(false);
    }
  };

  const handleDeleteNote = async (noteId: number) => {
    if (!incident) return;
    try {
      await apiService.deleteIncidentNote(incident.id, noteId);
      setNotes((prev) => prev.filter((n) => n.id !== noteId));
    } catch (err: any) {
      alert(err.message || 'تعذر حذف الملاحظة.');
    }
  };

  if (!incident) return null;

  const facts = incident.facts;
  const methods = incident.methods || [];
  const indicators = incident.indicators;
  const articles = incident.articles || [];

  return (
    <div
      className="fixed inset-0 z-50 bg-stone-900/60 dark:bg-black/75 backdrop-blur-sm flex items-start justify-center p-3 sm:p-5 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="w-full max-w-3xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 rounded-xl shadow-xl overflow-hidden my-6 sm:my-10 space-y-0 text-right"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header bar */}
        <div className="p-5 border-b border-stone-200 dark:border-slate-800/80 bg-stone-50/60 dark:bg-slate-900/40 flex items-start justify-between gap-4">
          <div className="space-y-1.5 min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-stone-200/80 dark:bg-slate-800 text-stone-700 dark:text-slate-300">
                {incident.ref}
              </span>
              <Chip tone={incident.record_type === 'confirmed_incident' ? 'critical' : 'high'}>
                {incident.record_type_ar}
              </Chip>
              <Chip tone="plain">{incident.review_status_ar}</Chip>
              {incident.bank_relevance && (
                <Chip tone={incident.bank_relevance === 'high' ? 'critical' : 'medium'}>
                  الصلة البنكية: {incident.bank_relevance === 'high' ? 'عالية' : incident.bank_relevance === 'medium' ? 'متوسطة' : 'منخفضة'}
                </Chip>
              )}
            </div>
            <h2 className="text-lg font-bold text-stone-900 dark:text-stone-100 leading-snug">
              {incident.title}
            </h2>
            <div className="flex flex-wrap items-center gap-3 text-xs text-stone-500 dark:text-slate-400 font-mono">
              <span>{incident.article_count} مصادر إخبارية</span>
              <span>•</span>
              <span>أول رصد: {fmtDay(incident.first_reported_at)}</span>
              {incident.locations.length > 0 && (
                <>
                  <span>•</span>
                  <span className="flex items-center gap-1">
                    <MapPin className="w-3 h-3 text-sky-600 dark:text-sky-400" />
                    {incident.locations.join('، ')}
                  </span>
                </>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {can('incident.annotate') && (
              <button
                onClick={handleToggleWatchlist}
                className={`p-2 rounded-lg border text-xs flex items-center gap-1.5 transition-colors ${
                  isSaved
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400 font-semibold'
                    : 'border-stone-200 dark:border-slate-700 text-stone-600 dark:text-slate-400 hover:bg-stone-100'
                }`}
                title={isSaved ? 'إزالة من قائمة المتابعة' : 'حفظ في قائمة المتابعة'}
              >
                <Bookmark className="w-4 h-4" />
                <span className="hidden sm:inline">{isSaved ? 'محفوظة' : 'حفظ'}</span>
              </button>
            )}
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 hover:bg-stone-200/60 dark:hover:bg-slate-800 transition-colors"
              title="إغلاق"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {feedbackMsg && (
          <div className="px-5 py-2.5 bg-blue-50 dark:bg-blue-950/40 border-b border-blue-200 dark:border-blue-900/60 text-blue-800 dark:text-blue-200 text-xs flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-blue-600 shrink-0" />
            <span>{feedbackMsg}</span>
          </div>
        )}

        {/* Content sections answering the 5 core analyst questions */}
        <div className="p-5 sm:p-6 space-y-6 divide-y divide-stone-100 dark:divide-slate-800/60">
          {/* 1. ماذا حدث؟ (Summary & What Happened) */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-stone-900 dark:text-stone-100 font-semibold text-sm">
              <FileText className="w-4 h-4 text-sky-600 dark:text-sky-400" />
              <span>١. ملخص الواقعة (ماذا حدث؟)</span>
            </div>
            <p className="text-sm text-stone-700 dark:text-slate-300 leading-relaxed">
              {incident.summary || incident.title}
            </p>

            {/* Factual snapshot pill tags */}
            {facts && (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-2">
                {facts.persons_count != null && (
                  <div className="p-2.5 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200/80 dark:border-slate-800 text-xs">
                    <span className="text-stone-500 dark:text-slate-400 block text-[11px]">المتهمون / الأشخاص:</span>
                    <span className="font-semibold text-stone-900 dark:text-stone-100 font-mono">{facts.persons_count} أشخاص</span>
                  </div>
                )}
                {facts.amounts.length > 0 && (
                  <div className="p-2.5 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200/80 dark:border-slate-800 text-xs">
                    <span className="text-stone-500 dark:text-slate-400 block text-[11px]">المبالغ المرصودة:</span>
                    <span className="font-semibold text-stone-900 dark:text-stone-100 font-mono">
                      {facts.amounts.map((a) => a.value.toLocaleString('ar-SA')).join(' / ')} ريال
                    </span>
                  </div>
                )}
                {facts.nationalities.length > 0 && (
                  <div className="p-2.5 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200/80 dark:border-slate-800 text-xs">
                    <span className="text-stone-500 dark:text-slate-400 block text-[11px]">الجنسيات:</span>
                    <span className="font-semibold text-stone-900 dark:text-stone-100">
                      {facts.nationalities.map((n) => n.label).join('، ')}
                    </span>
                  </div>
                )}
                {facts.agencies.length > 0 && (
                  <div className="p-2.5 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200/80 dark:border-slate-800 text-xs">
                    <span className="text-stone-500 dark:text-slate-400 block text-[11px]">الجهات المختصة:</span>
                    <span className="font-semibold text-stone-900 dark:text-stone-100">
                      {facts.agencies.map((g) => g.label).join('، ')}
                    </span>
                  </div>
                )}
                {facts.seized_items.length > 0 && (
                  <div className="p-2.5 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200/80 dark:border-slate-800 text-xs sm:col-span-2">
                    <span className="text-stone-500 dark:text-slate-400 block text-[11px]">المضبوطات:</span>
                    <span className="font-semibold text-stone-900 dark:text-stone-100">
                      {facts.seized_items.map((i) => i.label).join('، ')}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 2. كيف نُفّذ الاحتيال؟ (How it worked) */}
          <div className="pt-5 space-y-3">
            <div className="flex items-center gap-2 text-stone-900 dark:text-stone-100 font-semibold text-sm">
              <Workflow className="w-4 h-4 text-sky-600 dark:text-sky-400" />
              <span>٢. أسلوب التنفيذ (كيف نُفّذ الاحتيال؟)</span>
            </div>

            {methods.length > 0 ? (
              <div className="space-y-2">
                {methods.map((m) => (
                  <div
                    key={m.tag}
                    className="p-3 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200/80 dark:border-slate-800 text-xs space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-sky-800 dark:text-sky-300">{m.label}</span>
                      <EpistemicChip kind="fact" label="موثق في الخبر" />
                    </div>
                    {m.quotes && m.quotes.length > 0 && (
                      <p className="text-stone-600 dark:text-slate-400 text-[11px] leading-relaxed border-r-2 border-sky-400 pr-2 mt-1">
                        «{m.quotes[0].quote}»
                      </p>
                    )}
                  </div>
                ))}
              </div>
            ) : incident.channels.length > 0 ? (
              <div className="p-3 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200/80 dark:border-slate-800 text-xs">
                <span className="text-stone-500 dark:text-slate-400">القنوات والوسائط المرصودة: </span>
                <span className="font-semibold text-stone-800 dark:text-stone-200">{incident.channels.join('، ')}</span>
              </div>
            ) : (
              <Unavailable title="لا تفاصيل إضافية عن أسلوب التنفيذ" note="لم تذكر المصادر تسلسلاً تقنياً مفصلاً." />
            )}
          </div>

          {/* 3. لماذا يهم البنك؟ (Why it matters to a bank) */}
          <div className="pt-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-stone-900 dark:text-stone-100 font-semibold text-sm">
                <Landmark className="w-4 h-4 text-sky-600 dark:text-sky-400" />
                <span>٣. الصلة البنكية (لماذا يهم البنك؟)</span>
              </div>
              <EpistemicChip
                kind={incident.bank_relevance === 'high' ? 'inference' : 'unknown'}
                label={incident.bank_relevance === 'high' ? 'صلة مباشرة بالقنوات البنكية' : 'تأثير غير مباشر'}
              />
            </div>

            <div className="p-3.5 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200/80 dark:border-slate-800 text-xs text-stone-700 dark:text-slate-300 leading-relaxed space-y-2">
              <p>
                {incident.bank_relevance === 'high'
                  ? 'تمس هذه الحادثة قنوات التحويل أو الحسابات أو استغلال هويات العملاء لإجراء سحوبات أو تمرير حوالات غير مشروعة.'
                  : 'تمس الحادثة عمليات احتيال تعاقدي أو نصب واستثمار وهمي قد تنعكس على مطالبات استرداد أموال وشكاوى نزاع.'}
              </p>
              {incident.channels.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5 pt-1">
                  <span className="text-stone-500 dark:text-slate-400 text-[11px]">نقاط الملاحظة البنكية:</span>
                  {incident.channels.map((c) => (
                    <Chip key={c} tone="brand">{c}</Chip>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* 4. ما الذي يستطيع البنك كشفه؟ (Potential Detection) */}
          <div className="pt-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-stone-900 dark:text-stone-100 font-semibold text-sm">
                <ShieldCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                <span>٤. مؤشرات الكشف الممكنة</span>
              </div>
              {incident.rule && (
                <span className="font-mono text-[11px] text-sky-700 dark:text-sky-300 bg-sky-50 dark:bg-sky-950/60 px-2 py-0.5 rounded border border-sky-200 dark:border-sky-800">
                  قاعدة مقترحة: {incident.rule.rule_id}
                </span>
              )}
            </div>

            {indicators ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                {/* مذكورة في المصدر */}
                <div className="p-3 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200/80 dark:border-slate-800 space-y-1.5">
                  <div className="flex items-center gap-1.5 font-semibold text-emerald-700 dark:text-emerald-400 text-[11px]">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>مؤشرات ذكرت في المصدر (حقائق)</span>
                  </div>
                  {indicators.mentioned_in_source.length > 0 ? (
                    <ul className="space-y-1 text-stone-600 dark:text-slate-300 pr-4 list-disc text-[11px]">
                      {indicators.mentioned_in_source.map((item, idx) => (
                        <li key={idx}>{item.label}</li>
                      ))}
                    </ul>
                  ) : (
                    <span className="text-stone-400 dark:text-slate-500 text-[11px] block">لا مؤشرات صريحة في النص.</span>
                  )}
                </div>

                {/* مقترحة من المنظومة كاستنتاج */}
                <div className="p-3 rounded-lg bg-stone-50 dark:bg-slate-900/60 border border-stone-200/80 dark:border-slate-800 space-y-1.5">
                  <div className="flex items-center gap-1.5 font-semibold text-sky-700 dark:text-sky-400 text-[11px]">
                    <Eye className="w-3.5 h-3.5" />
                    <span>مؤشرات استنتاجية مقترحة (تحليل)</span>
                  </div>
                  {indicators.suggested_by_system?.items?.length > 0 ? (
                    <ul className="space-y-1 text-stone-600 dark:text-slate-300 pr-4 list-disc text-[11px]">
                      {indicators.suggested_by_system.items.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  ) : (
                    <span className="text-stone-400 dark:text-slate-500 text-[11px] block">لا توجد مؤشرات مقترحة.</span>
                  )}
                </div>
              </div>
            ) : (
              <Unavailable title="لا مؤشرات مسجلة" note="لم تُستخرج مؤشرات مخاطر محددة لهذه الحادثة." />
            )}
          </div>

          {/* 5. الأدلة والمصادر الداعمة (Evidence) */}
          <div className="pt-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-stone-900 dark:text-stone-100 font-semibold text-sm">
                <CheckCircle2 className="w-4 h-4 text-sky-600 dark:text-sky-400" />
                <span>٥. الأدلة وروابط التغطية الإخبارية ({articles.length})</span>
              </div>
              <span className="text-xs text-stone-500 dark:text-slate-400 font-mono">
                توثيق مباشر من الصحف والجهات
              </span>
            </div>

            <div className="space-y-2">
              {(showAllArticles ? articles : articles.slice(0, 3)).map((art) => (
                <div
                  key={art.article_id}
                  className="p-3 rounded-lg bg-stone-50 dark:bg-slate-900/40 border border-stone-200/80 dark:border-slate-800 text-xs space-y-1.5 hover:border-sky-300 dark:hover:border-sky-800 transition-colors"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-stone-800 dark:text-stone-200">{art.source || art.domain}</span>
                    <span className="text-[11px] font-mono text-stone-500 dark:text-slate-400">{fmtDay(art.published_at)}</span>
                  </div>
                  <div className="text-stone-700 dark:text-slate-300 leading-snug">{art.title}</div>
                  <div className="flex items-center justify-between pt-1 text-[11px]">
                    <span className="text-stone-500 dark:text-slate-400">
                      {art.match_basis?.type === 'seed'
                        ? 'الخبر المرجعي الأول'
                        : 'تم الدمج لتطابق الوقائع والبيانات'}
                    </span>
                    <a
                      href={art.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sky-600 dark:text-sky-400 font-medium hover:underline inline-flex items-center gap-1"
                    >
                      <span>المصدر الأصلي</span>
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>
                </div>
              ))}

              {articles.length > 3 && (
                <button
                  onClick={() => setShowAllArticles(!showAllArticles)}
                  className="w-full py-1.5 text-xs text-sky-600 dark:text-sky-400 hover:underline flex items-center justify-center gap-1 font-medium"
                >
                  <span>{showAllArticles ? 'عرض أقل' : `عرض كافة المقالات والمصادر (${articles.length})`}</span>
                  {showAllArticles ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>
              )}
            </div>

            {/* Expandable raw sentences from sources */}
            {facts?.fact_sentences && facts.fact_sentences.length > 0 && (
              <div className="pt-2">
                <button
                  onClick={() => setShowRawFacts(!showRawFacts)}
                  className="text-xs text-stone-500 dark:text-slate-400 hover:text-stone-800 dark:hover:text-stone-200 flex items-center gap-1"
                >
                  <span>{showRawFacts ? 'إخفاء الجمل الحرفية المقتبسة' : 'عرض الجمل والفقرات الحرفية من المصادر'}</span>
                  {showRawFacts ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>

                {showRawFacts && (
                  <div className="mt-2 space-y-1.5 p-3 rounded-lg bg-stone-100/70 dark:bg-slate-900 border border-stone-200 dark:border-slate-800 text-xs">
                    {facts.fact_sentences.map((sentence, idx) => (
                      <p key={idx} className="text-stone-700 dark:text-slate-300 leading-relaxed border-r-2 border-stone-400 pr-2">
                        {sentence.text}{' '}
                        <a
                          href={sentence.url}
                          target="_blank"
                          rel="noreferrer"
                          className="text-sky-600 dark:text-sky-400 hover:underline font-mono text-[10px]"
                        >
                          [{sentence.source}]
                        </a>
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 6. مساحة التحقيق والتعاون الداخلي للمحللين (Analyst Notes & Internal Investigation) */}
          {can('incident.annotate') && (
            <div className="pt-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                  <h3 className="text-sm font-bold text-stone-900 dark:text-white">
                    ملاحظات التحقيق الداخلي للفريق ({notes.length})
                  </h3>
                </div>
                <span className="text-[10px] text-stone-400 font-mono">
                  خاص بالمحللين • لا تظهر للمستعرضين
                </span>
              </div>

              {/* Add Note Form */}
              <form onSubmit={handleAddNote} className="space-y-2 p-3.5 rounded-xl bg-stone-50 dark:bg-slate-900/60 border border-stone-200 dark:border-slate-800">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-stone-700 dark:text-slate-300">نوع الملاحظة:</span>
                  <select
                    value={newNoteType}
                    onChange={(e) => setNewNoteType(e.target.value as any)}
                    className="bg-white dark:bg-slate-800 border border-stone-200 dark:border-slate-700 text-xs rounded-lg px-2 py-1 text-stone-800 dark:text-slate-200 focus:outline-none"
                  >
                    <option value="analysis">تحليل واستنتاج (Analysis)</option>
                    <option value="classification_correction">اقتراح تصحيح تصنيف (Correction)</option>
                    <option value="conflicting_info">معلومات متعارضة بين المصادر (Conflict)</option>
                    <option value="missing_info">معلومات ناقصة تتطلب تتبع (Missing Info)</option>
                  </select>
                </div>

                <div className="flex gap-2">
                  <textarea
                    value={newNoteContent}
                    onChange={(e) => setNewNoteContent(e.target.value)}
                    placeholder="دوّن استنتاجك، الأدلة المتقاطعة، أو ما إذا كانت هناك معلومات ناقصة أو تصنيف مقترح..."
                    rows={2}
                    className="flex-1 px-3 py-2 rounded-xl bg-white dark:bg-slate-800 border border-stone-200 dark:border-slate-700 text-xs text-stone-900 dark:text-white placeholder:text-stone-400 focus:outline-none focus:border-blue-500"
                    required
                  />
                  <button
                    type="submit"
                    disabled={savingNote || !newNoteContent.trim()}
                    className="px-3.5 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold self-end flex items-center gap-1 shadow-sm disabled:opacity-50 transition-colors"
                  >
                    <Send className="w-3.5 h-3.5" />
                    <span>إرسال</span>
                  </button>
                </div>
              </form>

              {/* List of Analyst Notes */}
              {notes.length > 0 && (
                <div className="space-y-2">
                  {notes.map((note) => {
                    const isAuthor = note.user_id === user?.id;
                    const canDelete = isAuthor || can('team.manage');

                    const typeLabels: Record<string, { label: string; color: string }> = {
                      analysis: { label: 'استنتاج تحليلي', color: 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/40 dark:text-blue-300' },
                      classification_correction: { label: 'تصحيح تصنيف', color: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300' },
                      conflicting_info: { label: 'تعارض معلومات', color: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950/40 dark:text-red-300' },
                      missing_info: { label: 'معلومات ناقصة', color: 'bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-950/40 dark:text-purple-300' },
                    };
                    const typeInfo = typeLabels[note.note_type] || { label: note.note_type, color: 'bg-stone-100 text-stone-700' };

                    return (
                      <div
                        key={note.id}
                        className="p-3 rounded-xl bg-stone-50/70 dark:bg-slate-900/40 border border-stone-200/80 dark:border-slate-800 space-y-1.5"
                      >
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-stone-900 dark:text-white">
                              {note.author_name}
                            </span>
                            <span className={`px-1.5 py-0.5 rounded text-[10px] border font-medium ${typeInfo.color}`}>
                              {typeInfo.label}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-stone-400 font-mono">
                              {fmtDateTime(note.created_at)}
                            </span>
                            {canDelete && (
                              <button
                                onClick={() => handleDeleteNote(note.id)}
                                className="text-stone-400 hover:text-red-500 p-0.5 rounded"
                                title="حذف الملاحظة"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            )}
                          </div>
                        </div>
                        <p className="text-xs text-stone-700 dark:text-slate-300 leading-relaxed pr-2 border-r-2 border-blue-400">
                          {note.content}
                        </p>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer actions */}
        <div className="p-4 border-t border-stone-200 dark:border-slate-800 bg-stone-50/80 dark:bg-slate-900/60 flex items-center justify-between text-xs">
          <span className="text-stone-500 dark:text-slate-400 font-mono">
            المعرف: {incident.ref} • تدقيق آلي مدعوم بالوقائع
          </span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-stone-800 dark:bg-slate-200 text-white dark:text-stone-900 font-medium hover:bg-stone-900 dark:hover:bg-white transition-colors"
          >
            إغلاق
          </button>
        </div>
      </div>
    </div>
  );
};
