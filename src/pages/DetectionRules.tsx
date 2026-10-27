// src/pages/DetectionRules.tsx
// MetaView 3 — Unified Fraud Detection Rules Intelligence & Deployment Studio
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ShieldCheck,
  Search,
  Layers,
  FlaskConical,
  Play,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Info,
  ExternalLink,
  FileCode,
  History,
  Sparkles,
  Cpu,
  Building2,
  Users2,
  Network,
  ArrowRight,
  Check,
  RefreshCw,
  Plus,
  Send,
  FileText,
  X,
  SlidersHorizontal,
  ChevronLeft,
  ChevronRight,
  BookmarkCheck,
  AlertCircle,
  HelpCircle,
  Fingerprint,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useAuth } from '../context/AuthContext';
import {
  UnifiedRule,
  RulesStudioOverview,
  SystemCapabilityEnvironment,
  SimulationSuiteResult,
  RuleOrigin,
  RuleStatus,
  DeploymentEnvironmentKey,
} from '../types/rules_studio';
import { LoadingBlock, ErrorBlock, fmtDay, fmtDateTime } from '../components/ui';

// Environment Icon Helper
const getEnvIcon = (envKey: string) => {
  switch (envKey) {
    case 'sas':
      return <Cpu className="w-4 h-4 text-sky-500" />;
    case 'branch_group':
      return <Building2 className="w-4 h-4 text-emerald-500" />;
    case 'shared_room':
      return <Users2 className="w-4 h-4 text-amber-500" />;
    case 'hybrid':
      return <Network className="w-4 h-4 text-indigo-500" />;
    default:
      return <Cpu className="w-4 h-4 text-stone-400" />;
  }
};

const getEnvBadge = (envKey: string) => {
  switch (envKey) {
    case 'sas':
      return <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-sky-500/10 text-sky-600 dark:text-sky-400 border border-sky-500/20"><Cpu className="w-3 h-3" />نظام ساس (SAS)</span>;
    case 'branch_group':
      return <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20"><Building2 className="w-3 h-3" />مجموعة الفروع</span>;
    case 'shared_room':
      return <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20"><Users2 className="w-3 h-3" />غرفة العمليات المشتركة</span>;
    case 'hybrid':
      return <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20"><Network className="w-3 h-3" />بيئة هجينة (Hybrid)</span>;
    default:
      return <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-stone-500/10 text-stone-600 dark:text-stone-400 border border-stone-500/20">غير محدد</span>;
  }
};

const getStatusBadge = (status: RuleStatus) => {
  switch (status) {
    case 'ready_for_bank_testing':
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 flex items-center gap-1"><CheckCircle2 className="w-3 h-3" />جاهز للاختبار المصرفي</span>;
    case 'simulation':
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-500/15 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30 flex items-center gap-1"><FlaskConical className="w-3 h-3" />مكتمل المحاكاة</span>;
    case 'approved_externally':
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-teal-500/15 text-teal-600 dark:text-teal-400 border border-teal-500/30 flex items-center gap-1"><BookmarkCheck className="w-3 h-3" />معتمد رسمياً خارجي</span>;
    case 'logic_review':
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-sky-500/15 text-sky-600 dark:text-sky-400 border border-sky-500/30 flex items-center gap-1"><SlidersHorizontal className="w-3 h-3" />مراجعة المنطق</span>;
    case 'evidence_review':
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30 flex items-center gap-1"><Layers className="w-3 h-3" />تدقيق الأدلة</span>;
    case 'draft':
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-stone-500/15 text-stone-600 dark:text-stone-400 border border-stone-500/30 flex items-center gap-1"><FileText className="w-3 h-3" />مسودة أولية</span>;
    case 'needs_revision':
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-orange-500/15 text-orange-600 dark:text-orange-400 border border-orange-500/30 flex items-center gap-1"><AlertTriangle className="w-3 h-3" />تتطلب تعديلاً</span>;
    case 'rejected':
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-500/15 text-red-600 dark:text-red-400 border border-red-500/30 flex items-center gap-1"><XCircle className="w-3 h-3" />مرفوضة</span>;
    default:
      return <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-stone-500/15 text-stone-600 dark:text-stone-400 border border-stone-500/30">مسودة</span>;
  }
};

const getOriginBadge = (origin: RuleOrigin) => {
  if (origin === 'intelligence') {
    return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800">مستخرجة من الرصد (Intelligence)</span>;
  }
  if (origin === 'analyst') {
    return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-purple-50 dark:bg-purple-950/40 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800">صياغة المحلل (Analyst Scenario)</span>;
  }
  return <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">هجينة (Hybrid)</span>;
};

export const DetectionRules: React.FC = () => {
  const { user, can } = useAuth();
  const canCreateRule = can('rule.create');
  const canApproveRule = can('rule.approve');

  // Navigation
  const [activeTab, setActiveTab] = useState<'discovery' | 'builder' | 'advisor' | 'simulation'>('discovery');

  // Core Data
  const [overview, setOverview] = useState<RulesStudioOverview | null>(null);
  const [rules, setRules] = useState<UnifiedRule[]>([]);
  const [environments, setEnvironments] = useState<Record<string, SystemCapabilityEnvironment> | null>(null);
  const [selectedRule, setSelectedRule] = useState<UnifiedRule | null>(null);

  // Loading & Errors
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [originFilter, setOriginFilter] = useState<string>('all');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [envFilter, setEnvFilter] = useState<string>('all');

  // Scenario Builder State
  const [scenarioInput, setScenarioInput] = useState('');
  const [isParsing, setIsParsing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [parsedDraft, setParsedDraft] = useState<UnifiedRule | null>(null);
  const [saveSuccessMsg, setSaveSuccessMsg] = useState<string | null>(null);

  // Simulation Runner State
  const [activeSimulationRule, setActiveSimulationRule] = useState<UnifiedRule | null>(null);
  const [simulationResult, setSimulationResult] = useState<SimulationSuiteResult | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);

  // Governance Modal State
  const [isGovModalOpen, setIsGovModalOpen] = useState(false);
  const [govTargetRule, setGovTargetRule] = useState<UnifiedRule | null>(null);
  const [govNewStatus, setGovNewStatus] = useState<string>('logic_review');
  const [govReason, setGovReason] = useState<string>('');
  const [govReviewer, setGovReviewer] = useState<string>('محلل مكافحة الاحتيال الأول');
  const [govAuthority, setGovAuthority] = useState<string>('');
  const [govRefDoc, setGovRefDoc] = useState<string>('');
  const [govError, setGovError] = useState<string | null>(null);
  const [isSubmittingGov, setIsSubmittingGov] = useState(false);

  // Initial Data Fetch
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [ovRes, listRes, envRes] = await Promise.all([
        apiService.getRulesStudioOverview(),
        apiService.getRulesStudioList(),
        apiService.getRulesStudioEnvironments(),
      ]);
      setOverview(ovRes);
      setRules(listRes.rules);
      setEnvironments(envRes.environments);
      // Auto-select first rule for deep dive if none selected
      if (listRes.rules.length > 0 && !selectedRule) {
        setSelectedRule(listRes.rules[0]);
      }
    } catch (e: any) {
      setError(e?.message || 'تعذر تحميل استوديو قواعد الكشف');
    } finally {
      setLoading(false);
    }
  }, [selectedRule]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Categories list for filter
  const categories = useMemo(() => {
    const set = new Set(rules.map((r) => r.fraud_category_ar || r.fraud_category));
    return Array.from(set).filter(Boolean);
  }, [rules]);

  // Filtered Rules
  const filteredRules = useMemo(() => {
    return rules.filter((r) => {
      if (originFilter !== 'all' && r.origin !== originFilter) return false;
      if (categoryFilter !== 'all' && (r.fraud_category_ar !== categoryFilter && r.fraud_category !== categoryFilter)) return false;
      if (statusFilter !== 'all' && r.status !== statusFilter) return false;
      if (envFilter !== 'all' && r.deployment?.recommended_environment !== envFilter) return false;
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchTitle = r.name.toLowerCase().includes(q);
        const matchId = r.rule_id.toLowerCase().includes(q);
        const matchHyp = r.detection_hypothesis.toLowerCase().includes(q);
        const matchCat = (r.fraud_category_ar || '').toLowerCase().includes(q);
        if (!matchTitle && !matchId && !matchHyp && !matchCat) return false;
      }
      return true;
    });
  }, [rules, originFilter, categoryFilter, statusFilter, envFilter, searchQuery]);

  // Handle Scenario Parse
  const handleParseScenario = async (textToParse?: string) => {
    const text = textToParse || scenarioInput;
    if (!text.trim()) return;
    setIsParsing(true);
    setParseError(null);
    setSaveSuccessMsg(null);
    try {
      const res = await apiService.parseRulesStudioScenario(text);
      setParsedDraft(res.proposed_rule);
    } catch (e: any) {
      setParseError(e?.message || 'فشل استخراج عناصر السيناريو');
    } finally {
      setIsParsing(false);
    }
  };

  // Handle Save Draft Rule
  const handleSaveDraft = async () => {
    if (!parsedDraft) return;
    try {
      const res = await apiService.saveRulesStudioRule(parsedDraft, 'إنشاء مسودة عبر محرر السيناريوهات الذكي');
      setSaveSuccessMsg(`تم حفظ مسودة القاعدة بنجاح برمز [${res.rule_id}]`);
      // Reload list and switch to discovery tab
      await loadData();
      setSelectedRule(res.rule);
    } catch (e: any) {
      setParseError(e?.message || 'تعذر حفظ مسودة القاعدة');
    }
  };

  // Handle Simulation
  const handleRunSimulation = async (rule: UnifiedRule) => {
    setActiveSimulationRule(rule);
    setActiveTab('simulation');
    setIsSimulating(true);
    try {
      const res = await apiService.simulateRulesStudioRule(rule.rule_id);
      setSimulationResult(res);
    } catch (e: any) {
      console.error('Simulation error:', e);
    } finally {
      setIsSimulating(false);
    }
  };

  // Handle Governance Submit
  const handleGovernanceSubmit = async () => {
    if (!govTargetRule) return;
    setIsSubmittingGov(true);
    setGovError(null);
    try {
      await apiService.updateRulesStudioGovernance(govTargetRule.rule_id, {
        new_status: govNewStatus,
        reason: govReason,
        reviewer_name: govReviewer,
        external_approval:
          govNewStatus === 'approved_externally'
            ? { authority: govAuthority, reference_doc: govRefDoc, notes: 'اعتماد رسمي موثق' }
            : undefined,
      });
      setIsGovModalOpen(false);
      await loadData();
      // Update selectedRule if active
      if (selectedRule && selectedRule.rule_id === govTargetRule.rule_id) {
        const updated = await apiService.getRulesStudioDetail(govTargetRule.rule_id);
        setSelectedRule(updated.rule);
      }
    } catch (e: any) {
      setGovError(e?.message || 'فشل تحديث حالة الحوكمة');
    } finally {
      setIsSubmittingGov(false);
    }
  };

  // Presets for Scenario Builder
  const PRESET_SCENARIOS = [
    {
      title: 'مستفيد جديد وتحويل يفوق المعدل',
      text: 'عميل أضاف مستفيدًا جديدًا من جهاز غير معتاد، ثم نفذ تحويلًا أعلى من سلوكه المعتاد خلال فترة قصيرة',
    },
    {
      title: 'انتحال جهة حكومية وطلب OTP',
      text: 'اتصال هاتفي ينتحل منصة حكومية يطلب رمز تحقق OTP لتسوية مخالفة متبوعاً بمحاولة تسجيل دخول من جهاز جديد',
    },
    {
      title: 'احتيال سداد وفاتورة لمفوتر غير معتاد',
      text: 'دخول من رابط خارجي وسداد فاتورة لمفوتر جديد لم يسبق التعامل معه بمبلغ يتجاوز متوسط الفواتير المعتادة',
    },
    {
      title: 'حساب وسيط (Mule) وتسييل سريع',
      text: 'حساب بنكي يستقبل تحويلات متعددة من 3 أطراف جديدة خلال 24 ساعة يتبعها سحب نقدي أو تحويل فوري لـ 80% من الرصيد',
    },
    {
      title: 'استيلاء على الحساب وتغيير الهاتف',
      text: 'تسجيل دخول من جهاز غير موثوق وتعديل رقم الجوال المسجل ثم محاولة إجراء تحويل مالي خلال أقل من ساعة',
    },
    {
      title: 'احتيال بطاقات بعملية اختبارية زهيدة',
      text: 'عملية تفويض بطاقة بمبلغ رمزي أقل من 5 ريالات متبوعة بعمليات شراء متتالية بمبالغ كبرى خلال 10 دقائق',
    },
  ];

  if (loading && !overview) {
    return <LoadingBlock label="جارٍ تحميل استوديو قواعد الكشف وسجل إمكانات الأنظمة…" />;
  }
  if (error && !overview) {
    return <ErrorBlock message={error} onRetry={loadData} />;
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-16 font-sans">
      {/* Header & Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-stone-200 dark:border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-stone-900 dark:text-stone-100">
              استوديو قواعد الكشف ونشر السيناريوهات
            </h1>
            <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-sky-500/10 text-sky-600 dark:text-sky-400 border border-sky-500/20">
              MetaView 3 Studio
            </span>
          </div>
          <p className="text-sm text-stone-600 dark:text-slate-400 mt-1 leading-relaxed">
            منظومة موحدة لمحلل مكافحة الاحتيال: رصد الأنماط الجنائية، هندسة الفرضيات، مطابقة بيئات النشر المصرفية، ومحاكاة السيناريوهات بالأدلة الحقيقية.
          </p>
        </div>

        {/* Action Button: Create Scenario */}
        <div className="flex items-center gap-2.5">
          <button
            onClick={() => {
              setActiveTab('builder');
              setScenarioInput(PRESET_SCENARIOS[0].text);
            }}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-sky-600 hover:bg-sky-500 text-white shadow-sm transition-all"
          >
            <Sparkles className="w-4 h-4" />
            صياغة سيناريو جديد باللغة الطبيعية
          </button>
          <button
            onClick={loadData}
            title="تحديث البيانات"
            className="p-2 rounded-xl border border-stone-200 dark:border-slate-800 hover:bg-stone-100 dark:hover:bg-slate-800/60 text-stone-600 dark:text-slate-300 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Honest Scientific Caution Banner */}
      <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-xs text-amber-900 dark:text-amber-300 leading-relaxed shadow-sm">
        <FlaskConical className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p>
            <strong>تنبيه منهجي وتوثيقي:</strong> جميع القواعد الحالية تعتبر <strong>فرضيات كشف قابلة للاختبار (Detection Hypotheses)</strong> مستخلصة من وقائع حقيقية. المنصة تفصل بدقة بين الأدلة المستخرجة من الحوادث والافتراضات الفنية للمحلل، ولا تدّعي نسب دقة أو استدعاء حقيقية حتى اكتمال المعايرة والاختبار على سجلات معاملات بنكية موسومة.
          </p>
          <p className="text-amber-800/80 dark:text-amber-400/80">
            حالة الاعتماد الخارجي (Approved Externally) تتطلب توثيق قرار رسمي صادر من الجهة المخولة، ولا تُمنح آلياً داخل المنصة.
          </p>
        </div>
      </div>

      {/* Overview KPI Cards */}
      {overview && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
          <div className="p-4 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-stone-500 dark:text-slate-400">
              <span>إجمالي القواعد الموثقة</span>
              <Layers className="w-4 h-4 text-sky-500" />
            </div>
            <div className="text-2xl font-bold text-stone-900 dark:text-stone-100">
              {overview.total_rules}
            </div>
            <div className="text-[11px] text-stone-500 dark:text-slate-500 flex items-center gap-1.5 pt-1 border-t border-stone-100 dark:border-slate-800/60">
              <span>{overview.origin_breakdown.intelligence} رصد آلي</span>
              <span>•</span>
              <span>{overview.origin_breakdown.analyst} مسودة محلل</span>
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-stone-500 dark:text-slate-400">
              <span>جاهزة للاختبار والمحاكاة</span>
              <FlaskConical className="w-4 h-4 text-indigo-500" />
            </div>
            <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
              {overview.testable_rules}
            </div>
            <div className="text-[11px] text-stone-500 dark:text-slate-500 flex items-center gap-1.5 pt-1 border-t border-stone-100 dark:border-slate-800/60">
              <span>مهيأة بسيناريوهات منطقية متكاملة</span>
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-stone-500 dark:text-slate-400">
              <span>بحاجة لمعايرة بيانات وتكامل</span>
              <AlertTriangle className="w-4 h-4 text-amber-500" />
            </div>
            <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">
              {overview.needing_data_rules}
            </div>
            <div className="text-[11px] text-stone-500 dark:text-slate-500 flex items-center gap-1.5 pt-1 border-t border-stone-100 dark:border-slate-800/60">
              <span>تتطلب حقول تيليمتري غير مكتملة</span>
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs text-stone-500 dark:text-slate-400">
              <span>تنتظر مراجعة حوكمة</span>
              <ShieldCheck className="w-4 h-4 text-emerald-500" />
            </div>
            <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
              {overview.awaiting_review_rules}
            </div>
            <div className="text-[11px] text-stone-500 dark:text-slate-500 flex items-center gap-1.5 pt-1 border-t border-stone-100 dark:border-slate-800/60">
              <span>قيد تدقيق الأدلة ومطابقة المنطق</span>
            </div>
          </div>
        </div>
      )}

      {/* Main Workspace Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-stone-200 dark:border-slate-800 pb-1">
        <button
          onClick={() => setActiveTab('discovery')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold transition-colors ${
            activeTab === 'discovery'
              ? 'bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800/60'
              : 'text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-stone-100'
          }`}
        >
          <Layers className="w-4 h-4" />
          قواعد الكشف المرصودة ({filteredRules.length})
        </button>

        <button
          onClick={() => setActiveTab('builder')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold transition-colors ${
            activeTab === 'builder'
              ? 'bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800/60'
              : 'text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-stone-100'
          }`}
        >
          <Sparkles className="w-4 h-4" />
          محرر السيناريوهات الذكي
        </button>

        <button
          onClick={() => setActiveTab('advisor')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold transition-colors ${
            activeTab === 'advisor'
              ? 'bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800/60'
              : 'text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-stone-100'
          }`}
        >
          <Cpu className="w-4 h-4" />
          مستشار النشر وسجل الأنظمة
        </button>

        <button
          onClick={() => setActiveTab('simulation')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold transition-colors ${
            activeTab === 'simulation'
              ? 'bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800/60'
              : 'text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-stone-100'
          }`}
        >
          <FlaskConical className="w-4 h-4" />
          استوديو المحاكاة والاختبار
        </button>
      </div>

      {/* ========================================================================= */}
      {/* TAB 1: RULE DISCOVERY & WORKBENCH */}
      {/* ========================================================================= */}
      {activeTab === 'discovery' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Rules List Column */}
          <div className="lg:col-span-7 space-y-4">
            {/* Search and Filters Strip */}
            <div className="p-3.5 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm space-y-3">
              <div className="relative">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="ابحث بالاسم، المعرف، الفرضية، أو التصنيف الجنائي..."
                  className="w-full bg-stone-50 dark:bg-slate-900/80 border border-stone-200 dark:border-slate-800 rounded-xl py-2 pr-9 pl-3 text-xs text-stone-900 dark:text-stone-100 placeholder:text-stone-400 dark:placeholder:text-slate-500 outline-none focus:border-sky-500"
                />
                <Search className="w-4 h-4 text-stone-400 absolute top-1/2 -translate-y-1/2 right-3 pointer-events-none" />
              </div>

              <div className="flex flex-wrap items-center gap-2 text-xs">
                {/* Origin Filter */}
                <select
                  value={originFilter}
                  onChange={(e) => setOriginFilter(e.target.value)}
                  className="bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-800 rounded-lg px-2.5 py-1 text-xs text-stone-700 dark:text-slate-300 outline-none"
                >
                  <option value="all">كل المصادر (Origin)</option>
                  <option value="intelligence">رصد آلي (Intelligence)</option>
                  <option value="analyst">صياغة المحلل (Analyst)</option>
                </select>

                {/* Environment Filter */}
                <select
                  value={envFilter}
                  onChange={(e) => setEnvFilter(e.target.value)}
                  className="bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-800 rounded-lg px-2.5 py-1 text-xs text-stone-700 dark:text-slate-300 outline-none"
                >
                  <option value="all">كل البيئات الموصى بها</option>
                  <option value="sas">نظام ساس (SAS)</option>
                  <option value="branch_group">مجموعة الفروع</option>
                  <option value="shared_room">الغرفة المشتركة</option>
                  <option value="hybrid">بيئة هجينة (Hybrid)</option>
                </select>

                {/* Status Filter */}
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-800 rounded-lg px-2.5 py-1 text-xs text-stone-700 dark:text-slate-300 outline-none"
                >
                  <option value="all">كل الحالات (Status)</option>
                  <option value="ready_for_bank_testing">جاهز للاختبار المصرفي</option>
                  <option value="simulation">مكتمل المحاكاة</option>
                  <option value="draft">مسودة</option>
                  <option value="approved_externally">معتمد رسمياً خارجي</option>
                </select>
              </div>
            </div>

            {/* Rules Cards Feed */}
            {filteredRules.length === 0 ? (
              <div className="p-8 text-center rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 space-y-2">
                <Info className="w-6 h-6 text-stone-400 mx-auto" />
                <p className="text-xs font-semibold text-stone-700 dark:text-slate-300">
                  لا توجد قواعد تطابق الفلاتر المحددة
                </p>
                <p className="text-[11px] text-stone-500 dark:text-slate-500">
                  جرّب إزالة بعض الفلاتر أو استخدام محرر السيناريوهات لصياغة قاعدة جديدة.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {filteredRules.map((rule) => {
                  const isSelected = selectedRule?.rule_id === rule.rule_id;
                  return (
                    <div
                      key={rule.rule_id}
                      onClick={() => setSelectedRule(rule)}
                      className={`p-4 rounded-2xl cursor-pointer transition-all border ${
                        isSelected
                          ? 'bg-sky-50/50 dark:bg-sky-950/20 border-sky-500/50 shadow-sm ring-1 ring-sky-500/20'
                          : 'bg-white dark:bg-[#111726] border-stone-200 dark:border-slate-800 hover:border-stone-300 dark:hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-mono text-xs font-bold text-sky-600 dark:text-sky-400">
                              {rule.rule_id}
                            </span>
                            {getOriginBadge(rule.origin)}
                            {getStatusBadge(rule.status)}
                          </div>
                          <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100 leading-snug">
                            {rule.name}
                          </h3>
                        </div>

                        <div className="shrink-0">
                          {getEnvBadge(rule.deployment?.recommended_environment)}
                        </div>
                      </div>

                      <p className="text-xs text-stone-600 dark:text-slate-400 mt-2 line-clamp-2 leading-relaxed">
                        {rule.detection_hypothesis}
                      </p>

                      {/* 3D Scores Strip */}
                      <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-stone-100 dark:border-slate-800/60 text-[11px]">
                        <div className="space-y-0.5">
                          <span className="text-stone-500 dark:text-slate-500 text-[10px]">قوة الأدلة</span>
                          <div className="flex items-center gap-1 font-semibold text-stone-800 dark:text-stone-200">
                            <span className="text-sky-600 dark:text-sky-400">{rule.scoring?.evidence_confidence}%</span>
                            <span className="text-[10px] text-stone-400">({rule.evidence?.corroboration_count || 0} وقائع)</span>
                          </div>
                        </div>

                        <div className="space-y-0.5">
                          <span className="text-stone-500 dark:text-slate-500 text-[10px]">جدوى الكشف</span>
                          <div className="flex items-center gap-1 font-semibold text-stone-800 dark:text-stone-200">
                            <span className="text-emerald-600 dark:text-emerald-400">{rule.scoring?.detection_feasibility}%</span>
                            <span className="text-[10px] text-stone-400">(بيانات مصرفية)</span>
                          </div>
                        </div>

                        <div className="space-y-0.5">
                          <span className="text-stone-500 dark:text-slate-500 text-[10px]">الجاهزية التشغيلية</span>
                          <div className="flex items-center gap-1 font-semibold text-stone-800 dark:text-stone-200">
                            <span className="text-indigo-600 dark:text-indigo-400">{rule.scoring?.operational_readiness}%</span>
                          </div>
                        </div>
                      </div>

                      {/* Action buttons footer */}
                      <div className="flex items-center justify-between gap-2 mt-3 pt-2 text-xs">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRunSimulation(rule);
                          }}
                          className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-stone-100 dark:bg-slate-800 hover:bg-stone-200 dark:hover:bg-slate-700 text-stone-700 dark:text-slate-200 transition-colors"
                        >
                          <Play className="w-3.5 h-3.5 text-indigo-500" />
                          تشغيل المحاكاة
                        </button>

                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setGovTargetRule(rule);
                            setGovNewStatus(rule.status);
                            setGovReason('');
                            setIsGovModalOpen(true);
                          }}
                          className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-stone-100"
                        >
                          <ShieldCheck className="w-3.5 h-3.5" />
                          تحديث الحوكمة
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Right Detail Workspace Column */}
          <div className="lg:col-span-5 sticky top-6 space-y-4">
            {selectedRule ? (
              <div className="p-5 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm space-y-5">
                {/* Header */}
                <div className="space-y-1.5 pb-4 border-b border-stone-100 dark:border-slate-800/80">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-sm font-bold text-sky-600 dark:text-sky-400">
                      {selectedRule.rule_id}
                    </span>
                    {getStatusBadge(selectedRule.status)}
                  </div>
                  <h2 className="text-base font-bold text-stone-900 dark:text-stone-100">
                    {selectedRule.name}
                  </h2>
                  <div className="flex items-center gap-2 pt-1 text-xs text-stone-500 dark:text-slate-400">
                    <span>التصنيف: <strong>{selectedRule.fraud_category_ar}</strong></span>
                    <span>•</span>
                    <span>الإصدار: v{selectedRule.governance?.current_version || '1.0.0'}</span>
                  </div>
                </div>

                {/* 3D Scoring Cards */}
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold text-stone-700 dark:text-slate-300">
                    تقييم نضج القاعدة (3D Scoring):
                  </h4>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="p-2.5 rounded-xl bg-sky-500/10 border border-sky-500/20 text-center space-y-0.5">
                      <span className="text-[10px] text-stone-500 dark:text-slate-400 block">قوة الأدلة</span>
                      <span className="text-base font-bold text-sky-600 dark:text-sky-400">
                        {selectedRule.scoring?.evidence_confidence}%
                      </span>
                    </div>
                    <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-center space-y-0.5">
                      <span className="text-[10px] text-stone-500 dark:text-slate-400 block">جدوى الكشف</span>
                      <span className="text-base font-bold text-emerald-600 dark:text-emerald-400">
                        {selectedRule.scoring?.detection_feasibility}%
                      </span>
                    </div>
                    <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-center space-y-0.5">
                      <span className="text-[10px] text-stone-500 dark:text-slate-400 block">الجاهزية</span>
                      <span className="text-base font-bold text-indigo-600 dark:text-indigo-400">
                        {selectedRule.scoring?.operational_readiness}%
                      </span>
                    </div>
                  </div>
                  <p className="text-[11px] text-stone-500 dark:text-slate-400 leading-relaxed bg-stone-50 dark:bg-slate-900/60 p-2.5 rounded-lg border border-stone-100 dark:border-slate-800">
                    {selectedRule.scoring?.rationales?.evidence}
                  </p>
                </div>

                {/* Section 1: Evidence vs Assumptions */}
                <div className="space-y-2.5 text-xs">
                  <h4 className="font-semibold text-stone-800 dark:text-slate-200 flex items-center gap-1.5">
                    <Fingerprint className="w-4 h-4 text-sky-500" />
                    الأدلة الجنائية المرصودة (Evidence References):
                  </h4>
                  {selectedRule.evidence?.incident_ids?.length > 0 ? (
                    <div className="p-3 rounded-xl bg-stone-50 dark:bg-slate-900/80 border border-stone-200 dark:border-slate-800 space-y-2">
                      <div className="flex items-center justify-between text-[11px] text-stone-600 dark:text-slate-400">
                        <span>الحوادث المرتبطة في MetaView 3:</span>
                        <span className="font-semibold text-sky-600 dark:text-sky-400">
                          {selectedRule.evidence.incident_ids.map((id) => `#INC-${id}`).join(', ')}
                        </span>
                      </div>
                      {selectedRule.evidence.sources?.length > 0 && (
                        <div className="flex items-center gap-1 flex-wrap pt-1 border-t border-stone-200/60 dark:border-slate-800/60">
                          <span className="text-[10px] text-stone-500">المصادر الموثقة:</span>
                          {selectedRule.evidence.sources.map((s, idx) => (
                            <span key={idx} className="px-1.5 py-0.5 rounded bg-white dark:bg-slate-800 text-[10px] border border-stone-200 dark:border-slate-700">
                              {s}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="text-[11px] text-stone-500 dark:text-slate-400 italic">
                      مسودة مصممة يدوياً بواسطة المحلل؛ لا ترتبط تلقائياً بوقائع جنائية مثبتة حتى الآن.
                    </p>
                  )}

                  {/* Technical Assumptions Disclosed */}
                  <div className="space-y-1">
                    <span className="text-[11px] font-medium text-stone-600 dark:text-slate-400">
                      الافتراضات الفنية للمحلل (Technical Assumptions):
                    </span>
                    <ul className="list-disc list-inside space-y-1 text-[11px] text-stone-600 dark:text-slate-400 pr-1">
                      {selectedRule.technical_assumptions?.map((asm, idx) => (
                        <li key={idx}>{asm}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Section 2: Neutral Detection Logic */}
                <div className="space-y-2 text-xs">
                  <h4 className="font-semibold text-stone-800 dark:text-slate-200 flex items-center gap-1.5">
                    <FileCode className="w-4 h-4 text-indigo-500" />
                    منطق الكشف المحايد (WHEN / THEN):
                  </h4>
                  <div className="p-3 rounded-xl bg-slate-900 text-slate-100 font-mono text-[11px] space-y-1.5 leading-relaxed overflow-x-auto" dir="ltr">
                    <div className="text-sky-400 font-bold">WHEN:</div>
                    {selectedRule.logic?.conditions?.map((c, idx) => (
                      <div key={idx} className="pl-4 text-slate-300">
                        • {c.field} {c.operator} <span className="text-amber-400">{String(c.value)}</span>
                        <span className="text-slate-500 text-[10px] block font-sans" dir="rtl">
                          ({c.description_ar})
                        </span>
                      </div>
                    ))}
                    <div className="text-emerald-400 font-bold pt-1">THEN:</div>
                    <div className="pl-4 text-slate-300">• GENERATE_FRAUD_ALERT_AND_CHALLENGE</div>
                  </div>
                </div>

                {/* Section 3: Deployment Recommendation */}
                <div className="space-y-2 text-xs">
                  <h4 className="font-semibold text-stone-800 dark:text-slate-200 flex items-center gap-1.5">
                    <Cpu className="w-4 h-4 text-emerald-500" />
                    توصية بيئة النشر (Deployment Recommendation):
                  </h4>
                  <div className="p-3 rounded-xl bg-stone-50 dark:bg-slate-900/80 border border-stone-200 dark:border-slate-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-stone-500">البيئة الموصى بها:</span>
                      {getEnvBadge(selectedRule.deployment?.recommended_environment)}
                    </div>
                    <p className="text-[11px] text-stone-600 dark:text-slate-400 leading-relaxed">
                      {selectedRule.deployment?.rationale}
                    </p>

                    {selectedRule.deployment?.hybrid_breakdown && (
                      <div className="p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-[11px] text-indigo-900 dark:text-indigo-300 space-y-1">
                        <strong>التنسيق الهجين:</strong>
                        <p>• {selectedRule.deployment.hybrid_breakdown.detection_engine}</p>
                        <p>• {selectedRule.deployment.hybrid_breakdown.operational_response}</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Section 4: Required Data Fields */}
                <div className="space-y-2 text-xs">
                  <h4 className="font-semibold text-stone-800 dark:text-slate-200 flex items-center gap-1.5">
                    <SlidersHorizontal className="w-4 h-4 text-amber-500" />
                    متطلبات البيانات المصرفية:
                  </h4>
                  <div className="flex flex-wrap gap-1.5">
                    {selectedRule.data_requirements?.required_fields?.map((f, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 rounded-md text-[10px] bg-stone-100 dark:bg-slate-800 text-stone-700 dark:text-slate-300 border border-stone-200 dark:border-slate-700"
                        title={f.description}
                      >
                        {f.name}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Action CTA in Drawer */}
                <div className="pt-2 border-t border-stone-100 dark:border-slate-800 flex items-center gap-2">
                  <button
                    onClick={() => handleRunSimulation(selectedRule)}
                    className="flex-1 flex items-center justify-center gap-2 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white shadow-sm transition-all"
                  >
                    <Play className="w-3.5 h-3.5" />
                    اختبار المحاكاة (6 حالات)
                  </button>
                  <button
                    onClick={() => {
                      if (!canApproveRule) return;
                      setGovTargetRule(selectedRule);
                      setGovNewStatus(selectedRule.status);
                      setGovReason('');
                      setIsGovModalOpen(true);
                    }}
                    disabled={!canApproveRule}
                    title={!canApproveRule ? 'يتطلب اعتماد الحوكمة صلاحية مدير المنصة' : 'مراجعة واعتماد الحوكمة'}
                    className={`px-3 py-2 rounded-xl text-xs font-medium border transition-colors ${
                      canApproveRule
                        ? 'border-stone-200 dark:border-slate-800 hover:bg-stone-100 dark:hover:bg-slate-800 text-stone-700 dark:text-slate-300 cursor-pointer'
                        : 'border-stone-200 dark:border-slate-800/60 text-stone-400 dark:text-slate-600 cursor-not-allowed bg-stone-50 dark:bg-slate-900/40'
                    }`}
                  >
                    الحوكمة {!canApproveRule && '(مدير)'}
                  </button>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 text-stone-400 text-xs">
                اختر قاعدة من القائمة لعرض تفاصيلها ومنطقها وأدلتها
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: ANALYST NATURAL LANGUAGE SCENARIO BUILDER */}
      {/* ========================================================================= */}
      {activeTab === 'builder' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <div className="lg:col-span-6 space-y-4">
            <div className="p-5 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm space-y-4">
              <div>
                <h3 className="text-base font-bold text-stone-900 dark:text-stone-100 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-sky-500" />
                  محرر السيناريوهات الذكي باللغة الطبيعية
                </h3>
                <p className="text-xs text-stone-600 dark:text-slate-400 mt-1 leading-relaxed">
                  اكتب السيناريو الاحتيالي باللغة العربية أو الإنجليزية كما ورد في البلاغ أو الملاحظة؛ سيقوم المحرك باستخراج أحداث البداية، والشروط، والكيانات، والحقول المطلوبة، والفجوات المعيارية.
                </p>
              </div>

              {/* Quick Preset Buttons */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-medium text-stone-500 dark:text-slate-400">
                  نماذج وسيناريوهات استرشادية سريعة:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {PRESET_SCENARIOS.map((ps, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setScenarioInput(ps.text);
                        handleParseScenario(ps.text);
                      }}
                      className="px-2.5 py-1 rounded-lg text-[11px] bg-stone-100 dark:bg-slate-800 hover:bg-sky-500/10 hover:text-sky-600 dark:hover:text-sky-400 text-stone-700 dark:text-slate-300 border border-stone-200 dark:border-slate-700 transition-colors"
                    >
                      {ps.title}
                    </button>
                  ))}
                </div>
              </div>

              {/* Textarea */}
              <div className="space-y-2">
                <textarea
                  rows={5}
                  value={scenarioInput}
                  onChange={(e) => setScenarioInput(e.target.value)}
                  placeholder="مثال: عميل أضاف مستفيدًا جديدًا من جهاز غير معتاد، ثم نفذ تحويلًا أعلى من سلوكه المعتاد خلال فترة قصيرة..."
                  className="w-full p-3.5 rounded-xl bg-stone-50 dark:bg-slate-900/80 border border-stone-200 dark:border-slate-800 text-xs text-stone-900 dark:text-stone-100 placeholder:text-stone-400 dark:placeholder:text-slate-500 outline-none focus:border-sky-500 leading-relaxed resize-y"
                />

                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-stone-400">
                    يدعم المصطلحات المصرفية والجنائية الشائعة
                  </span>
                  <button
                    disabled={isParsing || !scenarioInput.trim()}
                    onClick={() => handleParseScenario()}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white shadow-sm transition-all"
                  >
                    {isParsing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                    تحليل واستخراج عناصر القاعدة
                  </button>
                </div>
              </div>

              {parseError && (
                <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-xs text-red-600 dark:text-red-400 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{parseError}</span>
                </div>
              )}

              {saveSuccessMsg && (
                <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-600 dark:text-emerald-400 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                  <span>{saveSuccessMsg}</span>
                </div>
              )}
            </div>
          </div>

          {/* Parsed Output Column */}
          <div className="lg:col-span-6 space-y-4">
            {parsedDraft ? (
              <div className="p-5 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm space-y-4">
                <div className="flex items-center justify-between border-b border-stone-100 dark:border-slate-800 pb-3">
                  <div>
                    <span className="font-mono text-xs font-bold text-purple-600 dark:text-purple-400">
                      {parsedDraft.rule_id}
                    </span>
                    <h4 className="text-sm font-bold text-stone-900 dark:text-stone-100 mt-0.5">
                      مسودة القاعدة المستخلصة
                    </h4>
                  </div>
                  {getEnvBadge(parsedDraft.deployment?.recommended_environment)}
                </div>

                {/* Triggers */}
                <div className="space-y-1.5">
                  <span className="text-[11px] font-semibold text-stone-700 dark:text-slate-300 flex items-center gap-1.5">
                    <Play className="w-3.5 h-3.5 text-sky-500" />
                    أحداث البداية (Trigger Events):
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {parsedDraft.scenario_extraction?.triggers?.map((t, idx) => (
                      <span key={idx} className="px-2 py-0.5 rounded text-[11px] bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Conditions */}
                <div className="space-y-1.5">
                  <span className="text-[11px] font-semibold text-stone-700 dark:text-slate-300 flex items-center gap-1.5">
                    <SlidersHorizontal className="w-3.5 h-3.5 text-indigo-500" />
                    شروط الكشف المستخرجة:
                  </span>
                  <ul className="space-y-1 text-xs text-stone-700 dark:text-slate-300">
                    {parsedDraft.logic?.conditions?.map((c, idx) => (
                      <li key={idx} className="p-2 rounded-lg bg-stone-50 dark:bg-slate-900 border border-stone-100 dark:border-slate-800/80 flex items-start gap-2">
                        <span className="font-mono text-sky-500 text-[10px] mt-0.5">{c.id}:</span>
                        <span>{c.description_ar}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Missing Info / Calibration Needs */}
                {parsedDraft.scenario_extraction?.missing_info && parsedDraft.scenario_extraction.missing_info.length > 0 && (
                  <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-900 dark:text-amber-300 space-y-1.5">
                    <div className="font-semibold flex items-center gap-1.5">
                      <HelpCircle className="w-4 h-4 text-amber-500" />
                      معلومات لازمة للمعايرة البنكية (Missing Information):
                    </div>
                    <ul className="list-disc list-inside space-y-1 text-[11px] pr-1">
                      {parsedDraft.scenario_extraction.missing_info.map((info, idx) => (
                        <li key={idx}>{info}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Save Draft Action */}
                <div className="pt-2 border-t border-stone-100 dark:border-slate-800 space-y-2">
                  {!canCreateRule && (
                    <div className="p-2.5 rounded-xl bg-stone-100 dark:bg-slate-800/80 text-[11px] text-stone-600 dark:text-slate-400 border border-stone-200 dark:border-slate-700">
                      * وضع المستعرض العام (قراءة فقط): لحفظ مسودات القواعد في مساحة العمل يتطلب تسجيل الدخول بصلاحية محلل مكافحة الاحتيال أو مدير.
                    </div>
                  )}
                  <button
                    onClick={handleSaveDraft}
                    disabled={!canCreateRule}
                    className={`w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-semibold shadow-sm transition-all ${
                      canCreateRule
                        ? 'bg-emerald-600 hover:bg-emerald-500 text-white cursor-pointer'
                        : 'bg-stone-200 dark:bg-slate-800 text-stone-400 dark:text-slate-500 cursor-not-allowed border border-stone-300 dark:border-slate-700'
                    }`}
                  >
                    <Check className="w-4 h-4" />
                    حفظ كمسودة قاعدة موحدة (Save to Workspace)
                  </button>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 text-stone-400 text-xs space-y-2">
                <Sparkles className="w-6 h-6 text-stone-400 mx-auto" />
                <p>اكتب سيناريو في المحرر واضغط «تحليل واستخراج» لمشاهدة تفكيك المنطق آلياً</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: DEPLOYMENT ADVISOR & CAPABILITY REGISTRY */}
      {/* ========================================================================= */}
      {activeTab === 'advisor' && environments && (
        <div className="space-y-6">
          <div className="p-5 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm space-y-2">
            <h3 className="text-base font-bold text-stone-900 dark:text-stone-100 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-sky-500" />
              سجل إمكانات الأنظمة المصرفية (System Capability Registry)
            </h3>
            <p className="text-xs text-stone-600 dark:text-slate-400 leading-relaxed">
              تعريف موثق للإمكانات الفعلية والقيود التقنية لكل بيئة تشغيل؛ يساعد المستشار الذكي في مطابقة متطلبات كل قاعدة مع البيئة الأنسب وتحديد الفجوات المعيارية.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.values(environments).map((env) => (
              <div
                key={env.id}
                className="p-5 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm space-y-4"
              >
                <div className="flex items-start justify-between gap-3 border-b border-stone-100 dark:border-slate-800 pb-3">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-xl bg-stone-100 dark:bg-slate-800">
                      {getEnvIcon(env.id)}
                    </div>
                    <div>
                      <h4 className="text-sm font-bold text-stone-900 dark:text-stone-100">
                        {env.name_ar}
                      </h4>
                      <span className="text-[11px] text-stone-500 dark:text-slate-400">
                        {env.type_ar}
                      </span>
                    </div>
                  </div>
                  {getEnvBadge(env.id)}
                </div>

                <div className="space-y-3 text-xs">
                  <div>
                    <span className="text-[11px] font-semibold text-stone-500 dark:text-slate-400 block mb-1">
                      نمط المعالجة والسرعة:
                    </span>
                    <span className="px-2 py-0.5 rounded bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 font-medium">
                      {env.processing_mode_ar}
                    </span>
                  </div>

                  <div>
                    <span className="text-[11px] font-semibold text-stone-500 dark:text-slate-400 block mb-1">
                      ربط الأحداث والنوافذ الزمنية:
                    </span>
                    <p className="text-stone-700 dark:text-slate-300 text-[11px]">
                      {env.event_correlation.description_ar} ({env.temporal_windows.range_ar})
                    </p>
                  </div>

                  <div>
                    <span className="text-[11px] font-semibold text-stone-500 dark:text-slate-400 block mb-1">
                      العمليات المدعومة:
                    </span>
                    <div className="flex flex-wrap gap-1">
                      {env.supported_operations.slice(0, 5).map((op, idx) => (
                        <span key={idx} className="px-2 py-0.5 rounded bg-stone-100 dark:bg-slate-800 text-[10px] text-stone-700 dark:text-slate-300">
                          {op}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div>
                    <span className="text-[11px] font-semibold text-amber-600 dark:text-amber-400 block mb-1">
                      القيود التقنية المثبتة (Technical Limitations):
                    </span>
                    <ul className="list-disc list-inside space-y-1 text-[11px] text-stone-600 dark:text-slate-400 pr-1">
                      {env.technical_limitations.map((lim, idx) => (
                        <li key={idx}>{lim}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 4: SIMULATION STUDIO */}
      {/* ========================================================================= */}
      {activeTab === 'simulation' && (
        <div className="space-y-6">
          <div className="p-5 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <FlaskConical className="w-5 h-5 text-indigo-500" />
                <h3 className="text-base font-bold text-stone-900 dark:text-stone-100">
                  استوديو المحاكاة المنطقية للقواعد (6 Mandatory Test Cases)
                </h3>
              </div>
              <p className="text-xs text-stone-600 dark:text-slate-400 mt-1 leading-relaxed">
                اختبار دقة منطق الكشف عبر 6 سيناريوهات قياسية موسومة ببيانات اصطناعية؛ للتحقق من عدم إطلاق إنذارات كاذبة والتصرف الآمن عند نقص البيانات أو تعارض الترتيب.
              </p>
            </div>

            {/* Rule Selector */}
            <div className="flex items-center gap-2.5">
              <select
                value={activeSimulationRule?.rule_id || selectedRule?.rule_id || ''}
                onChange={(e) => {
                  const r = rules.find((x) => x.rule_id === e.target.value);
                  if (r) {
                    setActiveSimulationRule(r);
                    handleRunSimulation(r);
                  }
                }}
                className="bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-800 rounded-xl px-3 py-2 text-xs font-semibold text-stone-800 dark:text-slate-200 outline-none"
              >
                {rules.map((r) => (
                  <option key={r.rule_id} value={r.rule_id}>
                    {r.rule_id}: {r.name}
                  </option>
                ))}
              </select>

              <button
                disabled={isSimulating}
                onClick={() => {
                  const target = activeSimulationRule || selectedRule;
                  if (target) handleRunSimulation(target);
                }}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white shadow-sm transition-all disabled:opacity-50"
              >
                {isSimulating ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                إعادة تشغيل المحاكاة
              </button>
            </div>
          </div>

          {/* Simulation Results Feed */}
          {simulationResult ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3.5 rounded-xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 text-xs">
                <span className="text-stone-600 dark:text-slate-400">
                  القاعدة المختبرة: <strong>{simulationResult.rule_id}</strong>
                </span>
                <span className="font-semibold text-indigo-600 dark:text-indigo-400">
                  اجتياز {simulationResult.passed_cases} من أصل {simulationResult.total_test_cases} سيناريوهات
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {simulationResult.results.map((tc) => (
                  <div
                    key={tc.test_case_id}
                    className="p-4 rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-sm space-y-3"
                  >
                    <div className="flex items-start justify-between gap-2 border-b border-stone-100 dark:border-slate-800/80 pb-2.5">
                      <div>
                        <span className="font-mono text-[10px] text-stone-400 block">
                          {tc.test_case_id}
                        </span>
                        <h4 className="text-xs font-bold text-stone-900 dark:text-stone-100">
                          {tc.type_ar}
                        </h4>
                      </div>

                      {tc.verdict === 'ALERT_TRIGGERED' ? (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-red-500/15 text-red-600 dark:text-red-400 border border-red-500/20">
                          إطلاق إنذار (Alert Triggered)
                        </span>
                      ) : tc.verdict === 'INSUFFICIENT_DATA' ? (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                          بيانات غير كافية (Safe Hold)
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                          عدم إطلاق إنذار (No Alert)
                        </span>
                      )}
                    </div>

                    <p className="text-[11px] text-stone-600 dark:text-slate-400 leading-relaxed">
                      {tc.description}
                    </p>

                    <div className="space-y-1.5 text-[11px]">
                      <span className="font-medium text-stone-500">حالة الشروط:</span>
                      <div className="space-y-1">
                        {tc.matched_conditions.map((m, idx) => (
                          <div key={idx} className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
                            <Check className="w-3.5 h-3.5" />
                            <span>{m}</span>
                          </div>
                        ))}
                        {tc.unmatched_conditions.map((u, idx) => (
                          <div key={idx} className="flex items-center gap-1.5 text-stone-400">
                            <X className="w-3.5 h-3.5" />
                            <span>{u}</span>
                          </div>
                        ))}
                        {tc.missing_fields.map((mf, idx) => (
                          <div key={idx} className="flex items-center gap-1.5 text-amber-500">
                            <AlertCircle className="w-3.5 h-3.5" />
                            <span>حقل مفقود: {mf}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="p-2.5 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-100 dark:border-slate-800 text-[11px] text-stone-600 dark:text-slate-400">
                      <strong>سبب القرار:</strong> {tc.decision_reason}
                    </div>

                    <div className="text-[10px] text-stone-400 italic">
                      {tc.synthetic_tag}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="p-12 text-center rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 text-stone-400 text-xs space-y-2">
              <FlaskConical className="w-8 h-8 text-stone-400 mx-auto" />
              <p>اختر قاعدة واضغط «تشغيل المحاكاة» لإجراء فحص الحالات الستة المنطقية</p>
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* GOVERNANCE MODAL */}
      {/* ========================================================================= */}
      {isGovModalOpen && govTargetRule && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-lg rounded-2xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 shadow-xl p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-stone-100 dark:border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-stone-900 dark:text-stone-100 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-500" />
                تحديث دورة حياة القاعدة: {govTargetRule.rule_id}
              </h3>
              <button
                onClick={() => setIsGovModalOpen(false)}
                className="text-stone-400 hover:text-stone-600 dark:hover:text-stone-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="font-medium text-stone-700 dark:text-slate-300 block mb-1">
                  الحالة الجديدة (Lifecycle Stage):
                </label>
                <select
                  value={govNewStatus}
                  onChange={(e) => setGovNewStatus(e.target.value)}
                  className="w-full p-2.5 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-800 text-xs text-stone-900 dark:text-stone-100 outline-none"
                >
                  <option value="draft">مسودة أولية (Draft)</option>
                  <option value="evidence_review">تدقيق الأدلة (Evidence Review)</option>
                  <option value="logic_review">مراجعة المنطق (Logic Review)</option>
                  <option value="simulation">مرحلة المحاكاة (Simulation)</option>
                  <option value="ready_for_bank_testing">جاهز للاختبار في البنك (Ready for Bank Testing)</option>
                  <option value="approved_externally">معتمد رسمياً خارجي (Approved Externally)</option>
                  <option value="needs_revision">تتطلب تعديلاً (Needs Revision)</option>
                  <option value="rejected">مرفوضة (Rejected)</option>
                </select>
              </div>

              <div>
                <label className="font-medium text-stone-700 dark:text-slate-300 block mb-1">
                  سبب التغيير والملاحظات (Mandatory Justification):
                </label>
                <textarea
                  rows={3}
                  value={govReason}
                  onChange={(e) => setGovReason(e.target.value)}
                  placeholder="بيّن سبب ترقية أو تغيير حالة القاعدة ونتائج الفحص..."
                  className="w-full p-2.5 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-800 text-xs text-stone-900 dark:text-stone-100 outline-none"
                />
              </div>

              {govNewStatus === 'approved_externally' && (
                <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 space-y-2 text-xs">
                  <div className="font-semibold text-amber-800 dark:text-amber-300 flex items-center gap-1.5">
                    <AlertTriangle className="w-4 h-4" />
                    بيانات الاعتماد الخارجي الإلزامية:
                  </div>
                  <div>
                    <label className="block text-[11px] mb-0.5 text-stone-600 dark:text-slate-400">الجهة المعتمدة (Authority):</label>
                    <input
                      type="text"
                      value={govAuthority}
                      onChange={(e) => setGovAuthority(e.target.value)}
                      placeholder="مثال: البنك المركزي السعودي / إدارة مخاطر الاحتيال"
                      className="w-full p-2 rounded-lg bg-white dark:bg-slate-800 border border-stone-200 dark:border-slate-700 text-xs"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] mb-0.5 text-stone-600 dark:text-slate-400">رقم الوثيقة أو التعميم (Reference Doc):</label>
                    <input
                      type="text"
                      value={govRefDoc}
                      onChange={(e) => setGovRefDoc(e.target.value)}
                      placeholder="مثال: REF-AUTH-2026-SAMA-092"
                      className="w-full p-2 rounded-lg bg-white dark:bg-slate-800 border border-stone-200 dark:border-slate-700 text-xs"
                    />
                  </div>
                </div>
              )}

              {govError && (
                <div className="p-2.5 rounded-xl bg-red-500/10 border border-red-500/30 text-[11px] text-red-600 dark:text-red-400">
                  {govError}
                </div>
              )}
            </div>

            <div className="flex items-center justify-end gap-2 pt-2 border-t border-stone-100 dark:border-slate-800">
              <button
                onClick={() => setIsGovModalOpen(false)}
                className="px-4 py-2 rounded-xl text-xs font-medium text-stone-600 dark:text-slate-400 hover:bg-stone-100 dark:hover:bg-slate-800"
              >
                إلغاء
              </button>
              <button
                disabled={isSubmittingGov || !govReason.trim()}
                onClick={handleGovernanceSubmit}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white shadow-sm transition-all"
              >
                {isSubmittingGov ? 'جارٍ الحفظ…' : 'اعتماد التحديث'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DetectionRules;
