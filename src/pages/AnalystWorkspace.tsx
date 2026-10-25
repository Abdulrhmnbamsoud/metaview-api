// src/pages/AnalystWorkspace.tsx
import React, { useState, useEffect } from 'react';
import {
  Briefcase,
  Bookmark,
  Network,
  ListChecks,
  CheckCircle2,
  AlertCircle,
  Clock,
  Play,
  Send,
  Plus,
  Trash2,
  ExternalLink,
  ShieldCheck,
  RefreshCw,
  FileText,
  Check,
  X,
  AlertTriangle,
  Layers,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useAuth } from '../context/AuthContext';
import {
  CustomPattern,
  CustomRule,
  SavedIncidentItem,
  ReviewRequest,
  WorkspaceOverview,
} from '../types/auth';
import { fmtDateTime } from '../components/ui';
import { IncidentDetailModal } from '../components/IncidentDetailModal';

export const AnalystWorkspace: React.FC = () => {
  const { user, can } = useAuth();

  const [activeTab, setActiveTab] = useState<'saved' | 'patterns' | 'rules' | 'reviews'>('saved');
  const [loading, setLoading] = useState(true);
  const [overview, setOverview] = useState<WorkspaceOverview | null>(null);

  const [savedIncidents, setSavedIncidents] = useState<SavedIncidentItem[]>([]);
  const [patterns, setPatterns] = useState<CustomPattern[]>([]);
  const [rules, setRules] = useState<CustomRule[]>([]);
  const [reviews, setReviews] = useState<ReviewRequest[]>([]);

  // Selected incident modal
  const [selectedIncidentId, setSelectedIncidentId] = useState<number | null>(null);

  // Modals & Forms
  const [patternModalOpen, setPatternModalOpen] = useState(false);
  const [patKey, setPatKey] = useState('');
  const [patLabel, setPatLabel] = useState('');
  const [patDesc, setPatDesc] = useState('');
  const [patModus, setPatModus] = useState('');
  const [patChannels, setPatChannels] = useState('WhatsApp, SMS');
  const [patTargets, setPatTargets] = useState('الأفراد والمتقاعدين');
  const [patBanking, setPatBanking] = useState('حسابات الأفراد المصرفية');
  const [patHypothesis, setPatHypothesis] = useState('');

  const [ruleModalOpen, setRuleModalOpen] = useState(false);
  const [ruleId, setRuleId] = useState('');
  const [ruleName, setRuleName] = useState('');
  const [ruleHypothesis, setRuleHypothesis] = useState('');
  const [ruleLogic, setRuleLogic] = useState('');
  const [rulePatternKey, setRulePatternKey] = useState('');
  const [ruleRequiredData, setRuleRequiredData] = useState('login_ip, device_fingerprint, tx_amount');
  const [ruleExceptions, setRuleExceptions] = useState('العمليات الموثقة عبر تطبيق البنك لأكثر من 30 يوماً');
  const [ruleFpRisk, setRuleFpRisk] = useState('منخفض');

  // Review Decision Modal
  const [decisionModal, setDecisionModal] = useState<{
    itemType: 'pattern' | 'rule';
    itemId: string;
    itemTitle: string;
  } | null>(null);
  const [decisionChoice, setDecisionChoice] = useState<'approved' | 'changes_requested' | 'rejected'>('approved');
  const [decisionNotes, setDecisionNotes] = useState('');

  const [submitting, setSubmitting] = useState(false);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setActionError(null);
    try {
      const [ovRes, savedRes, patRes, rulesRes, revRes] = await Promise.all([
        apiService.getWorkspaceOverview(),
        apiService.getSavedIncidents(),
        apiService.getWorkspacePatterns(),
        apiService.getWorkspaceRules(),
        can('pattern.approve') ? apiService.getPendingReviews('all') : apiService.getPendingReviews('pending'),
      ]);
      setOverview(ovRes);
      setSavedIncidents(savedRes.items || []);
      setPatterns(patRes.patterns || []);
      setRules(rulesRes.rules || []);
      setReviews(revRes.reviews || []);
    } catch (err: any) {
      setActionError(err.message || 'تعذر تحميل بيانات مساحة العمل.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (can('incident.annotate') || can('pattern.create') || can('rule.create') || can('pattern.approve')) {
      fetchData();
    }
  }, [user]);

  if (!can('incident.annotate') && !can('pattern.create') && !can('rule.create') && !can('pattern.approve')) {
    return (
      <div className="max-w-2xl mx-auto my-12 p-8 bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200 dark:border-slate-800 text-center space-y-4 shadow-sm">
        <div className="w-12 h-12 rounded-xl bg-amber-50 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400 mx-auto flex items-center justify-center">
          <Briefcase className="w-6 h-6" />
        </div>
        <h2 className="text-lg font-bold text-stone-900 dark:text-white">
          مساحة عمل المحللين والتحقيق
        </h2>
        <p className="text-xs text-stone-500 dark:text-slate-400 leading-relaxed">
          هذه المساحة مخصصة للمحللين المعتمدين ومدراء المنصة لتدوين الملاحظات، استخراج الأنماط، وهندسة قواعد الكشف واختبارها. الحسابات من نوع (مستعرض VIEWER) تملك صلاحية الاطلاع على تقارير ميتا فيو المعتمدة فقط.
        </p>
      </div>
    );
  }

  // Handle Create Pattern Draft
  const handleCreatePattern = async (e: React.FormEvent) => {
    e.preventDefault();
    setActionError(null);
    setSubmitting(true);
    try {
      await apiService.createPatternDraft({
        key: patKey.trim().toLowerCase(),
        label_ar: patLabel.trim(),
        description: patDesc.trim(),
        modus_operandi: patModus.trim(),
        channels: patChannels.split(',').map((s) => s.trim()).filter(Boolean),
        target_groups: patTargets.split(',').map((s) => s.trim()).filter(Boolean),
        banking_relevance: patBanking.trim(),
        hypothesis: patHypothesis.trim(),
      });
      setActionSuccess(`تم حفظ مسودة النمط "${patLabel}" بنجاح.`);
      setPatternModalOpen(false);
      setPatKey('');
      setPatLabel('');
      setPatDesc('');
      setPatModus('');
      setPatHypothesis('');
      fetchData();
    } catch (err: any) {
      setActionError(err.message || 'تعذر حفظ مسودة النمط.');
    } finally {
      setSubmitting(false);
    }
  };

  // Submit Pattern for Review
  const handleSubmitPattern = async (key: string) => {
    setActionError(null);
    try {
      await apiService.submitPatternForReview(key, 'جاهز للمراجعة والاعتماد');
      setActionSuccess(`تم إرسال النمط [${key}] للمراجعة والاعتماد.`);
      fetchData();
    } catch (err: any) {
      setActionError(err.message || 'تعذر إرسال النمط.');
    }
  };

  // Handle Create Rule Draft
  const handleCreateRule = async (e: React.FormEvent) => {
    e.preventDefault();
    setActionError(null);
    setSubmitting(true);
    try {
      await apiService.createRuleDraft({
        rule_id: ruleId.trim().toUpperCase(),
        name: ruleName.trim(),
        hypothesis: ruleHypothesis.trim(),
        logic: ruleLogic.trim(),
        pattern_key: rulePatternKey.trim() || undefined,
        required_data: ruleRequiredData.split(',').map((s) => s.trim()).filter(Boolean),
        exceptions: ruleExceptions.trim(),
        false_positive_risk: ruleFpRisk.trim(),
      });
      setActionSuccess(`تم إنشاء مسودة قاعدة الكشف [${ruleId.toUpperCase()}] بنجاح.`);
      setRuleModalOpen(false);
      setRuleId('');
      setRuleName('');
      setRuleHypothesis('');
      setRuleLogic('');
      fetchData();
    } catch (err: any) {
      setActionError(err.message || 'تعذر إنشاء مسودة القاعدة.');
    } finally {
      setSubmitting(false);
    }
  };

  // Test Detection Rule in Sandbox
  const handleTestRule = async (id: string) => {
    setActionError(null);
    try {
      const res = await apiService.testDetectionRule(id);
      setActionSuccess(`اكتمل فحص القاعدة في بيئة المحاكاة: دقة ${Math.round((res.test_result?.precision || 0.9) * 100)}%، واكتشاف ${res.test_result?.alerts_simulated || 0} تهديد مطابق.`);
      fetchData();
    } catch (err: any) {
      setActionError(err.message || 'تعذر تشغيل اختبار القاعدة.');
    }
  };

  // Submit Rule for Review
  const handleSubmitRule = async (id: string) => {
    setActionError(null);
    try {
      await apiService.submitRuleForReview(id, 'تم الاختبار بنجاح في بيئة المحاكاة ومطلوب الاعتماد');
      setActionSuccess(`تم إرسال قاعدة الكشف [${id}] للمدير للاعتماد.`);
      fetchData();
    } catch (err: any) {
      setActionError(err.message || 'تعذر إرسال القاعدة.');
    }
  };

  // Admin Decision on Pattern or Rule
  const handleDecideReview = async () => {
    if (!decisionModal) return;
    if (decisionNotes.trim().length < 3) {
      setActionError('يرجى كتابة سبب القرار أو التوجيهات للمحلل.');
      return;
    }
    setActionError(null);
    setSubmitting(true);
    try {
      if (decisionModal.itemType === 'pattern') {
        await apiService.decidePatternReview(decisionModal.itemId, decisionChoice, decisionNotes);
      } else {
        await apiService.decideRuleReview(decisionModal.itemId, decisionChoice, decisionNotes);
      }
      setActionSuccess(`تم تسجيل قرار (${decisionChoice === 'approved' ? 'الاعتماد' : decisionChoice === 'changes_requested' ? 'طلب التعديل' : 'الرفض'}) بنجاح.`);
      setDecisionModal(null);
      setDecisionNotes('');
      fetchData();
    } catch (err: any) {
      setActionError(err.message || 'تعذر تسجيل القرار.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleUnsave = async (incidentId: number) => {
    try {
      await apiService.unsaveIncident(incidentId);
      setActionSuccess('تمت إزالة الحادثة من قائمة المتابعة.');
      fetchData();
    } catch (err: any) {
      setActionError(err.message || 'تعذر إزالة الحادثة.');
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'approved':
        return { label: 'معتمد ومنشور', color: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300' };
      case 'pending_review':
        return { label: 'قيد مراجعة المدير', color: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300' };
      case 'changes_requested':
        return { label: 'مطلوب تعديلات', color: 'bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-950/40 dark:text-purple-300' };
      case 'rejected':
        return { label: 'مرفوض', color: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950/40 dark:text-red-300' };
      case 'draft':
      default:
        return { label: 'مسودة قيد الإعداد', color: 'bg-stone-50 text-stone-700 border-stone-200 dark:bg-slate-800 dark:text-slate-300' };
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Workspace Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white dark:bg-[#0f172a] p-5 sm:p-6 rounded-2xl border border-stone-200/80 dark:border-slate-800 shadow-sm">
        <div className="space-y-1">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400">
              <Briefcase className="w-5 h-5" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-stone-900 dark:text-white">
              مساحة عمل المحلل (Analyst Workspace)
            </h1>
          </div>
          <p className="text-xs text-stone-500 dark:text-slate-400">
            تحويل المعلومات الخام إلى تحليلات ميتا فيو الموثقة، إدارة قائمة المتابعة، صياغة الأنماط، وهندسة واختبار قواعد الكشف
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          {can('pattern.create') && (
            <button
              onClick={() => setPatternModalOpen(true)}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl border border-stone-200 dark:border-slate-700 text-stone-700 dark:text-slate-200 hover:bg-stone-50 dark:hover:bg-slate-800 text-xs font-semibold shadow-xs transition-all"
            >
              <Network className="w-4 h-4 text-blue-600" />
              <span>نمط جديد</span>
            </button>
          )}

          {can('rule.create') && (
            <button
              onClick={() => setRuleModalOpen(true)}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white text-xs font-semibold shadow-sm transition-all"
            >
              <Plus className="w-4 h-4" />
              <span>صياغة قاعدة كشف</span>
            </button>
          )}

          <button
            onClick={fetchData}
            className="p-2 rounded-xl border border-stone-200 dark:border-slate-700 text-stone-600 dark:text-slate-400 hover:bg-stone-50 dark:hover:bg-slate-800 transition-colors"
            title="تحديث البيانات"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Notifications */}
      {actionSuccess && (
        <div className="p-3.5 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/60 text-emerald-800 dark:text-emerald-200 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
            <span>{actionSuccess}</span>
          </div>
          <button onClick={() => setActionSuccess(null)} className="text-emerald-600 text-xs">
            ✕
          </button>
        </div>
      )}

      {actionError && (
        <div className="p-3.5 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800/60 text-red-800 dark:text-red-200 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-red-500 shrink-0" />
            <span>{actionError}</span>
          </div>
          <button onClick={() => setActionError(null)} className="text-red-600 text-xs">
            ✕
          </button>
        </div>
      )}

      {/* Metric Overview Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3.5">
        <div className="p-4 rounded-2xl bg-white dark:bg-[#0f172a] border border-stone-200/80 dark:border-slate-800 shadow-sm space-y-1">
          <div className="text-[11px] font-semibold text-stone-400 dark:text-slate-500">حوادث في المتابعة</div>
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400 font-mono">
            {savedIncidents.length}
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-white dark:bg-[#0f172a] border border-stone-200/80 dark:border-slate-800 shadow-sm space-y-1">
          <div className="text-[11px] font-semibold text-stone-400 dark:text-slate-500">مسودات الأنماط</div>
          <div className="text-2xl font-bold text-stone-900 dark:text-white font-mono">
            {patterns.filter((p) => p.status === 'draft').length}
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-white dark:bg-[#0f172a] border border-stone-200/80 dark:border-slate-800 shadow-sm space-y-1">
          <div className="text-[11px] font-semibold text-stone-400 dark:text-slate-500">مسودات قواعد الكشف</div>
          <div className="text-2xl font-bold text-stone-900 dark:text-white font-mono">
            {rules.filter((r) => r.status === 'draft').length}
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-white dark:bg-[#0f172a] border border-stone-200/80 dark:border-slate-800 shadow-sm space-y-1">
          <div className="text-[11px] font-semibold text-stone-400 dark:text-slate-500">قيد المراجعة</div>
          <div className="text-2xl font-bold text-amber-600 dark:text-amber-400 font-mono">
            {reviews.filter((r) => r.status === 'pending').length}
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-white dark:bg-[#0f172a] border border-stone-200/80 dark:border-slate-800 shadow-sm space-y-1">
          <div className="text-[11px] font-semibold text-stone-400 dark:text-slate-500">معتمد ومنشور للجميع</div>
          <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 font-mono">
            {patterns.filter((p) => p.status === 'approved').length + rules.filter((r) => r.status === 'approved').length}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-stone-200 dark:border-slate-800 gap-4">
        <button
          onClick={() => setActiveTab('saved')}
          className={`pb-3 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
            activeTab === 'saved'
              ? 'border-blue-600 text-blue-600 dark:text-blue-400'
              : 'border-transparent text-stone-500 dark:text-slate-400 hover:text-stone-900'
          }`}
        >
          <Bookmark className="w-4 h-4" />
          <span>قائمة المتابعة ({savedIncidents.length})</span>
        </button>

        <button
          onClick={() => setActiveTab('patterns')}
          className={`pb-3 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
            activeTab === 'patterns'
              ? 'border-blue-600 text-blue-600 dark:text-blue-400'
              : 'border-transparent text-stone-500 dark:text-slate-400 hover:text-stone-900'
          }`}
        >
          <Network className="w-4 h-4" />
          <span>أنماط الاحتيال الخاصة بي ({patterns.length})</span>
        </button>

        <button
          onClick={() => setActiveTab('rules')}
          className={`pb-3 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
            activeTab === 'rules'
              ? 'border-blue-600 text-blue-600 dark:text-blue-400'
              : 'border-transparent text-stone-500 dark:text-slate-400 hover:text-stone-900'
          }`}
        >
          <ListChecks className="w-4 h-4" />
          <span>قواعد الكشف والاختبار ({rules.length})</span>
        </button>

        <button
          onClick={() => setActiveTab('reviews')}
          className={`pb-3 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
            activeTab === 'reviews'
              ? 'border-blue-600 text-blue-600 dark:text-blue-400'
              : 'border-transparent text-stone-500 dark:text-slate-400 hover:text-stone-900'
          }`}
        >
          <ShieldCheck className="w-4 h-4" />
          <span>حوكمة المراجعات ({reviews.filter((r) => r.status === 'pending').length})</span>
        </button>
      </div>

      {/* Tab 1: Saved Incidents (Watchlist) */}
      {activeTab === 'saved' && (
        <div className="space-y-3">
          {savedIncidents.length === 0 ? (
            <div className="p-12 text-center bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200 dark:border-slate-800 space-y-2">
              <Bookmark className="w-8 h-8 text-stone-300 dark:text-slate-600 mx-auto" />
              <div className="text-xs font-semibold text-stone-700 dark:text-slate-300">
                لا توجد حوادث محفوظة في قائمة المتابعة
              </div>
              <p className="text-[11px] text-stone-400 max-w-sm mx-auto">
                يمكنك حفظ أي حادثة من صفحة الاستكشاف أو تفاصيل الحوادث للرجوع إليها ومتابعة تطوراتها.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
              {savedIncidents.map((item) => (
                <div
                  key={item.id}
                  className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 p-4 space-y-3 hover:shadow-md transition-shadow relative flex flex-col justify-between"
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="px-2 py-0.5 rounded-md bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900/60 text-blue-700 dark:text-blue-300 text-[10px] font-semibold">
                        {item.fraud_category_ar || item.fraud_category}
                      </span>
                      <button
                        onClick={() => handleUnsave(item.incident_id)}
                        className="text-stone-400 hover:text-red-500 p-1 rounded-lg transition-colors"
                        title="إزالة من المتابعة"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <h3 className="text-xs font-bold text-stone-900 dark:text-white line-clamp-2 leading-relaxed">
                      {item.headline || item.title}
                    </h3>

                    {item.notes && (
                      <div className="p-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-100 dark:border-slate-800 text-[11px] text-stone-600 dark:text-slate-400 italic">
                        "{item.notes}"
                      </div>
                    )}
                  </div>

                  <div className="pt-3 border-t border-stone-100 dark:border-slate-800/80 flex items-center justify-between">
                    <span className="text-[10px] text-stone-400 font-mono">
                      حفظت: {fmtDateTime(item.saved_at)}
                    </span>
                    <button
                      onClick={() => setSelectedIncidentId(item.incident_id)}
                      className="flex items-center gap-1 text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline"
                    >
                      <span>فتح الحادثة</span>
                      <ExternalLink className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Custom Patterns */}
      {activeTab === 'patterns' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {patterns.map((pat) => {
              const badge = getStatusBadge(pat.status);
              const isOwner = pat.author_id === user?.id;
              const isAdmin = can('pattern.approve');

              return (
                <div
                  key={pat.id}
                  className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 p-5 space-y-4 shadow-sm flex flex-col justify-between"
                >
                  <div className="space-y-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="space-y-1">
                        <span className={`inline-block px-2 py-0.5 rounded-md border text-[11px] font-semibold ${badge.color}`}>
                          {badge.label}
                        </span>
                        <div className="text-[10px] text-stone-400 font-mono" dir="ltr">
                          {pat.key}
                        </div>
                      </div>
                      <div className="text-right text-[11px] text-stone-400 font-mono">
                        {fmtDateTime(pat.updated_at)}
                      </div>
                    </div>

                    <h3 className="text-sm font-bold text-stone-900 dark:text-white leading-snug">
                      {pat.label_ar}
                    </h3>

                    {pat.description && (
                      <p className="text-xs text-stone-600 dark:text-slate-300 leading-relaxed">
                        {pat.description}
                      </p>
                    )}

                    {pat.modus_operandi && (
                      <div className="text-xs text-stone-700 dark:text-slate-300 bg-stone-50 dark:bg-slate-900 p-2.5 rounded-xl border border-stone-100 dark:border-slate-800 space-y-1">
                        <span className="font-semibold text-stone-900 dark:text-white block text-[11px]">
                          الأسلوب الإجرامي (Modus Operandi):
                        </span>
                        <span>{pat.modus_operandi}</span>
                      </div>
                    )}

                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {pat.channels?.map((ch, idx) => (
                        <span key={idx} className="px-2 py-0.5 rounded-md bg-stone-100 dark:bg-slate-800 text-[10px] font-medium text-stone-600 dark:text-slate-400">
                          {ch}
                        </span>
                      ))}
                    </div>

                    {pat.review_notes && (
                      <div className="p-2.5 rounded-xl bg-amber-50/60 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/60 text-xs text-amber-800 dark:text-amber-300">
                        <span className="font-semibold block text-[11px]">ملاحظة المراجع:</span>
                        <span>{pat.review_notes}</span>
                      </div>
                    )}
                  </div>

                  <div className="pt-3 border-t border-stone-100 dark:border-slate-800 flex items-center justify-between gap-2">
                    <div className="text-[11px] text-stone-400">
                      المعد: <span className="font-medium text-stone-700 dark:text-slate-300">{pat.author_name || 'أنا'}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      {/* Submit for review button */}
                      {isOwner && (pat.status === 'draft' || pat.status === 'changes_requested') && (
                        <button
                          onClick={() => handleSubmitPattern(pat.key)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold shadow-xs transition-colors"
                        >
                          <Send className="w-3.5 h-3.5" />
                          <span>إرسال للمراجعة</span>
                        </button>
                      )}

                      {/* Admin Decision button */}
                      {isAdmin && pat.status === 'pending_review' && (
                        <button
                          onClick={() =>
                            setDecisionModal({
                              itemType: 'pattern',
                              itemId: pat.key,
                              itemTitle: pat.label_ar,
                            })
                          }
                          className="flex items-center gap-1 px-3 py-1.5 rounded-xl bg-purple-600 hover:bg-purple-700 text-white text-xs font-semibold transition-colors"
                        >
                          <ShieldCheck className="w-3.5 h-3.5" />
                          <span>اتخاذ قرار</span>
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Tab 3: Custom Detection Rules */}
      {activeTab === 'rules' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {rules.map((rule) => {
              const badge = getStatusBadge(rule.status);
              const isOwner = rule.author_id === user?.id;
              const isAdmin = can('rule.approve');

              return (
                <div
                  key={rule.id}
                  className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 p-5 space-y-4 shadow-sm flex flex-col justify-between"
                >
                  <div className="space-y-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="space-y-1">
                        <span className={`inline-block px-2 py-0.5 rounded-md border text-[11px] font-semibold ${badge.color}`}>
                          {badge.label}
                        </span>
                        <div className="text-[10px] text-stone-400 font-mono" dir="ltr">
                          {rule.rule_id}
                        </div>
                      </div>
                      <div className="text-right text-[11px] text-stone-400 font-mono">
                        {fmtDateTime(rule.updated_at)}
                      </div>
                    </div>

                    <h3 className="text-sm font-bold text-stone-900 dark:text-white leading-snug">
                      {rule.name}
                    </h3>

                    <div className="text-xs text-stone-600 dark:text-slate-300">
                      <span className="font-semibold text-stone-800 dark:text-slate-200">الفرضية: </span>
                      {rule.hypothesis}
                    </div>

                    {/* Rule Logic Block */}
                    <div className="p-3 rounded-xl bg-stone-900 text-stone-100 font-mono text-[11px] leading-relaxed overflow-x-auto" dir="ltr">
                      <code>{rule.logic}</code>
                    </div>

                    {/* Simulation results banner */}
                    {rule.test_results && (
                      <div className="p-3 rounded-xl bg-emerald-50/70 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/60 text-xs space-y-1">
                        <div className="flex items-center gap-1.5 font-bold text-emerald-800 dark:text-emerald-200 text-[11px]">
                          <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                          <span>نتائج الاختبار في بيئة المحاكاة:</span>
                        </div>
                        <div className="grid grid-cols-3 gap-2 pt-1 text-center font-mono text-[11px]">
                          <div>
                            <span className="block text-stone-400 text-[10px]">الدقة</span>
                            <span className="font-bold text-emerald-700 dark:text-emerald-300">
                              {Math.round(rule.test_results.precision * 100)}%
                            </span>
                          </div>
                          <div>
                            <span className="block text-stone-400 text-[10px]">الحساسية</span>
                            <span className="font-bold text-emerald-700 dark:text-emerald-300">
                              {Math.round(rule.test_results.recall * 100)}%
                            </span>
                          </div>
                          <div>
                            <span className="block text-stone-400 text-[10px]">تنبيهات محاكاة</span>
                            <span className="font-bold text-stone-800 dark:text-stone-200">
                              {rule.test_results.alerts_simulated}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="pt-3 border-t border-stone-100 dark:border-slate-800 flex items-center justify-between gap-2">
                    <div className="text-[11px] text-stone-400">
                      المعد: <span className="font-medium text-stone-700 dark:text-slate-300">{rule.author_name || 'أنا'}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      {/* Run Test simulation */}
                      {can('rule.test') && (
                        <button
                          onClick={() => handleTestRule(rule.rule_id)}
                          className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl border border-stone-200 dark:border-slate-700 hover:bg-stone-50 dark:hover:bg-slate-800 text-stone-700 dark:text-slate-300 text-xs font-semibold transition-colors"
                          title="تشغيل اختبار محاكاة ضد بيانات التهديدات"
                        >
                          <Play className="w-3.5 h-3.5 text-blue-600" />
                          <span>اختبار</span>
                        </button>
                      )}

                      {/* Submit for review */}
                      {isOwner && (rule.status === 'draft' || rule.status === 'changes_requested') && (
                        <button
                          onClick={() => handleSubmitRule(rule.rule_id)}
                          className="flex items-center gap-1 px-3 py-1.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold shadow-xs transition-colors"
                        >
                          <Send className="w-3.5 h-3.5" />
                          <span>إرسال للاعتماد</span>
                        </button>
                      )}

                      {/* Admin Decision */}
                      {isAdmin && rule.status === 'pending_review' && (
                        <button
                          onClick={() =>
                            setDecisionModal({
                              itemType: 'rule',
                              itemId: rule.rule_id,
                              itemTitle: rule.name,
                            })
                          }
                          className="flex items-center gap-1 px-3 py-1.5 rounded-xl bg-purple-600 hover:bg-purple-700 text-white text-xs font-semibold transition-colors"
                        >
                          <ShieldCheck className="w-3.5 h-3.5" />
                          <span>اتخاذ قرار</span>
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Tab 4: Reviews Queue & Decisions */}
      {activeTab === 'reviews' && (
        <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-right text-xs">
              <thead className="bg-stone-50/70 dark:bg-slate-900/40 text-stone-500 dark:text-slate-400 border-b border-stone-100 dark:border-slate-800">
                <tr>
                  <th className="py-3 px-4 font-semibold">العنصر المطلوب</th>
                  <th className="py-3 px-4 font-semibold">المحلل المُعد</th>
                  <th className="py-3 px-4 font-semibold">تاريخ الإرسال</th>
                  <th className="py-3 px-4 font-semibold">الحالة</th>
                  <th className="py-3 px-4 font-semibold">ملاحظات المحلل</th>
                  <th className="py-3 px-4 font-semibold text-center">القرار</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100 dark:divide-slate-800/60 text-stone-800 dark:text-stone-200">
                {reviews.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-stone-400">
                      لا توجد طلبات مراجعة معلقة حالياً.
                    </td>
                  </tr>
                ) : (
                  reviews.map((rev) => (
                    <tr key={rev.id} className="hover:bg-stone-50/60 dark:hover:bg-slate-800/30">
                      <td className="py-3 px-4">
                        <div className="font-semibold text-stone-900 dark:text-white flex items-center gap-1.5">
                          <span className="px-1.5 py-0.5 rounded bg-blue-50 dark:bg-blue-950 text-blue-700 text-[10px] font-mono uppercase">
                            {rev.item_type}
                          </span>
                          <span>{rev.item_id}</span>
                        </div>
                      </td>
                      <td className="py-3 px-4 font-medium">{rev.author_name}</td>
                      <td className="py-3 px-4 text-stone-400 font-mono text-[11px]">{fmtDateTime(rev.created_at)}</td>
                      <td className="py-3 px-4">
                        <span className={`inline-block px-2 py-0.5 rounded-md border text-[11px] font-semibold ${getStatusBadge(rev.status).color}`}>
                          {getStatusBadge(rev.status).label}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-stone-500 max-w-xs truncate">{rev.notes || '-'}</td>
                      <td className="py-3 px-4 text-center">
                        {can('pattern.approve') && rev.status === 'pending' ? (
                          <button
                            onClick={() =>
                              setDecisionModal({
                                itemType: rev.item_type,
                                itemId: rev.item_id,
                                itemTitle: `${rev.item_type}: ${rev.item_id}`,
                              })
                            }
                            className="px-3 py-1 rounded-xl bg-purple-600 hover:bg-purple-700 text-white font-semibold text-[11px]"
                          >
                            مراجعة واعتماد
                          </button>
                        ) : (
                          <span className="text-[11px] text-stone-400 font-mono">
                            {rev.reviewer_name ? `بواسطة: ${rev.reviewer_name}` : '-'}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Modal: Create Pattern Draft */}
      {patternModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200 dark:border-slate-800 max-w-lg w-full p-6 space-y-4 shadow-2xl max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-stone-100 dark:border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Network className="w-5 h-5 text-blue-600" />
                <h3 className="font-bold text-sm text-stone-900 dark:text-white">صياغة نمط احتيال جديد</h3>
              </div>
              <button onClick={() => setPatternModalOpen(false)} className="text-stone-400 hover:text-stone-600">
                ✕
              </button>
            </div>

            <form onSubmit={handleCreatePattern} className="space-y-3.5">
              <div className="space-y-1">
                <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                  المعرف الرمزي (Key)
                </label>
                <input
                  type="text"
                  value={patKey}
                  onChange={(e) => setPatKey(e.target.value)}
                  placeholder="fake_investment_sms_2026"
                  dir="ltr"
                  className="w-full px-3 py-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs font-mono"
                  required
                />
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                  عنوان النمط بالعربية
                </label>
                <input
                  type="text"
                  value={patLabel}
                  onChange={(e) => setPatLabel(e.target.value)}
                  placeholder="انتحال منصات التداول وطلب تفويض مصرفي"
                  className="w-full px-3 py-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs"
                  required
                />
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                  الأسلوب الإجرامي (Modus Operandi)
                </label>
                <textarea
                  value={patModus}
                  onChange={(e) => setPatModus(e.target.value)}
                  rows={2}
                  placeholder="كيف تنفذ العصابة الاحتيال خطوة بخطوة..."
                  className="w-full px-3 py-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                    القنوات المستخدمة
                  </label>
                  <input
                    type="text"
                    value={patChannels}
                    onChange={(e) => setPatChannels(e.target.value)}
                    placeholder="WhatsApp, SMS"
                    className="w-full px-3 py-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                    الفئات المستهدفة
                  </label>
                  <input
                    type="text"
                    value={patTargets}
                    onChange={(e) => setPatTargets(e.target.value)}
                    placeholder="المستثمرون الأفراد"
                    className="w-full px-3 py-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs"
                  />
                </div>
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                  الفرضية والاستنتاج التحليلي
                </label>
                <textarea
                  value={patHypothesis}
                  onChange={(e) => setPatHypothesis(e.target.value)}
                  rows={2}
                  placeholder="استنتاج المحلل حول التهديد والجهات الواقفة خلفه..."
                  className="w-full px-3 py-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2 border-t border-stone-100 dark:border-slate-800">
                <button
                  type="button"
                  onClick={() => setPatternModalOpen(false)}
                  className="px-4 py-2 rounded-xl border border-stone-200 text-xs font-medium"
                >
                  إلغاء
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold"
                >
                  حفظ المسودة
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal: Create Detection Rule Draft */}
      {ruleModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200 dark:border-slate-800 max-w-lg w-full p-6 space-y-4 shadow-2xl max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-stone-100 dark:border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <ListChecks className="w-5 h-5 text-blue-600" />
                <h3 className="font-bold text-sm text-stone-900 dark:text-white">صياغة قاعدة كشف مصرفية مقترحة</h3>
              </div>
              <button onClick={() => setRuleModalOpen(false)} className="text-stone-400 hover:text-stone-600">
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateRule} className="space-y-3.5">
              <div className="space-y-1">
                <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                  معرف القاعدة (Rule ID)
                </label>
                <input
                  type="text"
                  value={ruleId}
                  onChange={(e) => setRuleId(e.target.value)}
                  placeholder="RULE-SA-AUTH-001"
                  dir="ltr"
                  className="w-full px-3 py-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs font-mono"
                  required
                />
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                  اسم القاعدة
                </label>
                <input
                  type="text"
                  value={ruleName}
                  onChange={(e) => setRuleName(e.target.value)}
                  placeholder="كشف تغيير كلمة المرور والتحويل الفوري خلال 5 دقائق"
                  className="w-full px-3 py-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs"
                  required
                />
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                  فرضية الكشف (Hypothesis)
                </label>
                <textarea
                  value={ruleHypothesis}
                  onChange={(e) => setRuleHypothesis(e.target.value)}
                  rows={2}
                  placeholder="ما الذي تحاول القاعدة كشفه استناداً لأدلة الحوادث..."
                  className="w-full px-3 py-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs"
                  required
                />
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                  منطق القاعدة (Detection Logic)
                </label>
                <textarea
                  value={ruleLogic}
                  onChange={(e) => setRuleLogic(e.target.value)}
                  rows={2}
                  placeholder="pwd_changed == TRUE AND tx_amount > 5000 AND delta_time < 300s"
                  dir="ltr"
                  className="w-full px-3 py-2 rounded-xl bg-stone-900 text-stone-100 font-mono text-xs border border-stone-700"
                  required
                />
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                  الحقول والبيانات المطلوبة
                </label>
                <input
                  type="text"
                  value={ruleRequiredData}
                  onChange={(e) => setRuleRequiredData(e.target.value)}
                  placeholder="ip_address, device_id, tx_timestamp"
                  className="w-full px-3 py-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs font-mono"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2 border-t border-stone-100 dark:border-slate-800">
                <button
                  type="button"
                  onClick={() => setRuleModalOpen(false)}
                  className="px-4 py-2 rounded-xl border border-stone-200 text-xs font-medium"
                >
                  إلغاء
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold"
                >
                  حفظ مسودة القاعدة
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal: Admin Decision on Review */}
      {decisionModal && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200 dark:border-slate-800 max-w-md w-full p-6 space-y-4 shadow-2xl">
            <h3 className="font-bold text-sm text-stone-900 dark:text-white">
              حوكمة الاعتماد: {decisionModal.itemTitle}
            </h3>

            <div className="space-y-1.5">
              <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                القرار الإداري
              </label>
              <div className="grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => setDecisionChoice('approved')}
                  className={`p-2.5 rounded-xl border text-center transition-all ${
                    decisionChoice === 'approved'
                      ? 'border-emerald-600 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 font-bold'
                      : 'border-stone-200 text-stone-600'
                  }`}
                >
                  اعتماد ونشر
                </button>
                <button
                  type="button"
                  onClick={() => setDecisionChoice('changes_requested')}
                  className={`p-2.5 rounded-xl border text-center transition-all ${
                    decisionChoice === 'changes_requested'
                      ? 'border-amber-600 bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300 font-bold'
                      : 'border-stone-200 text-stone-600'
                  }`}
                >
                  طلب تعديل
                </button>
                <button
                  type="button"
                  onClick={() => setDecisionChoice('rejected')}
                  className={`p-2.5 rounded-xl border text-center transition-all ${
                    decisionChoice === 'rejected'
                      ? 'border-red-600 bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 font-bold'
                      : 'border-stone-200 text-stone-600'
                  }`}
                >
                  رفض
                </button>
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                سبب القرار والتوجيهات (إلزامي للتوثيق والتدقيق)
              </label>
              <textarea
                value={decisionNotes}
                onChange={(e) => setDecisionNotes(e.target.value)}
                rows={3}
                placeholder="بيان سبب الاعتماد أو النقاط المطلوب تعديلها من المحلل..."
                className="w-full px-3 py-2 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs"
                required
              />
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setDecisionModal(null)}
                className="px-3.5 py-1.5 rounded-xl border border-stone-200 text-xs font-medium"
              >
                إلغاء
              </button>
              <button
                type="button"
                disabled={submitting}
                onClick={handleDecideReview}
                className="px-4 py-1.5 rounded-xl bg-purple-600 hover:bg-purple-700 text-white text-xs font-semibold"
              >
                {submitting ? 'جارِ التسجيل...' : 'تثبيت القرار'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Incident Detail Modal */}
      {selectedIncidentId !== null && (
        <IncidentDetailModal
          incidentId={selectedIncidentId}
          onClose={() => setSelectedIncidentId(null)}
        />
      )}
    </div>
  );
};
