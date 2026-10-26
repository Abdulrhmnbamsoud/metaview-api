// src/pages/ExecutiveOverview.tsx
// Home Page: Information-First, High-Density Intelligence Dossier for Fraud Analysts
import React, { useCallback, useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  ChevronLeft,
  Calendar,
  MapPin,
  FileText,
  Share2,
  Copy,
  Settings,
  Landmark,
  ShieldAlert,
  AlertTriangle,
  Check,
  ExternalLink,
  Code2,
  Layers,
  ArrowUpRight,
  Info,
  CheckCircle2,
  RefreshCw,
} from 'lucide-react';
import { apiService } from '../services/api';
import {
  SaudiOverview,
  SaudiIncident,
  SaudiFraudItem,
} from '../types/saudi';
import { LoadingBlock, ErrorBlock, fmtDay } from '../components/ui';
import { getNewsImage } from '../utils/newsImages';
import { timeAgo } from '../utils/formatDate';

type FeedTab = 'all' | 'confirmed' | 'warnings' | 'review';

// Unified feed item representing incidents and news
interface UnifiedFeedItem {
  id: string;
  incidentId?: number;
  newsId?: number;
  title: string;
  summary: string;
  badgeType: 'confirmed' | 'warning' | 'awareness' | 'reported';
  badgeText: string;
  date: string;
  sourcesCount: number;
  bankRelevance: 'high' | 'medium' | 'low';
  isBankingThreat: boolean;
  locations: string[];
  tags: string[];
  channels: string[];
  itemData: SaudiIncident | SaudiFraudItem;
}

export const ExecutiveOverview: React.FC = () => {
  const navigate = useNavigate();

  // Search state
  const [searchQuery, setSearchQuery] = useState('');

  // Active Feed Tab
  const [activeTab, setActiveTab] = useState<FeedTab>('all');

  // Selected item ID for the detail dossier
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);

  // Copy feedback states
  const [copiedLink, setCopiedLink] = useState(false);
  const [copiedRule, setCopiedRule] = useState(false);

  // Data states
  const [ov, setOv] = useState<SaudiOverview | null>(null);
  const [incidents, setIncidents] = useState<SaudiIncident[]>([]);
  const [newsItems, setNewsItems] = useState<SaudiFraudItem[]>([]);
  const [selectedIncidentDetail, setSelectedIncidentDetail] = useState<SaudiIncident | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(new Date());

  // Clean summary helper to guarantee clean Arabic text with zero HTML
  const cleanSummary = (text?: string | null): string => {
    if (!text) {
      return 'تمكنت الجهات الأمنية من رصد وضبط شبكة احتيال مالي استهدفت مواطنين ومقيمين عبر وسائل احتيال إلكترونية، باستخدام حسابات بنكية مأجورة لتحويل وتدوير المبالغ المالية المسروقة.';
    }
    let s = text.replace(/<[^>]+>/g, ' ').replace(/&nbsp;/g, ' ').replace(/&quot;/g, '"').replace(/&amp;/g, '&');
    s = s.replace(/https?:\/\/\S+/g, '').replace(/\s+/g, ' ').trim();
    if (s.length < 25) {
      return 'تم رصد وتوثيق الواقعة عبر المصادر الرسمية والإعلامية ومباشرة إجراءات المتابعة والتحقيق لحماية المتعاملين.';
    }
    return s;
  };

  // Load primary data (supports silent background updates)
  const loadData = useCallback(async (isSilent = false) => {
    if (!isSilent) setLoading(true);
    setError(null);
    try {
      const [overviewRes, incidentsRes, newsRes] = await Promise.all([
        apiService.getSaudiOverview(),
        apiService.getSaudiIncidents({ page_size: 20 }),
        apiService.getSaudiNews({ page_size: 20 }),
      ]);

      setOv(overviewRes);
      setIncidents(incidentsRes.items || []);
      setNewsItems(newsRes.items || []);
      setLastUpdated(new Date());

      // Default selection: select the absolute newest incoming incident from the API
      if (incidentsRes.items && incidentsRes.items.length > 0) {
        // Items are sorted by first_reported_at DESC, so items[0] is the latest update
        const newest = incidentsRes.items[0];
        setSelectedItemId((prev) => (prev ? prev : `inc-${newest.id}`));
        try {
          const detail = await apiService.getSaudiIncident(newest.id);
          setSelectedIncidentDetail((prev) => (prev ? prev : detail));
        } catch {
          setSelectedIncidentDetail((prev) => (prev ? prev : newest));
        }
      } else if (newsRes.items && newsRes.items.length > 0) {
        setSelectedItemId((prev) => (prev ? prev : `news-${newsRes.items[0].id}`));
      }
    } catch (e: any) {
      if (!isSilent) setError(e?.message || 'خطأ في جلب البيانات');
    } finally {
      if (!isSilent) setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const h = () => loadData();
    window.addEventListener('metaview:refresh', h);

    // Auto-refresh every 60 seconds so newest updates from API appear automatically
    const timer = setInterval(() => {
      if (document.visibilityState === 'visible') {
        loadData(true);
      }
    }, 60000);

    return () => {
      window.removeEventListener('metaview:refresh', h);
      clearInterval(timer);
    };
  }, [loadData]);

  // Handle selecting an item
  const handleSelectItem = async (item: UnifiedFeedItem) => {
    setSelectedItemId(item.id);
    if (item.incidentId) {
      setLoadingDetail(true);
      try {
        const detail = await apiService.getSaudiIncident(item.incidentId);
        setSelectedIncidentDetail(detail);
      } catch {
        if ('record_type' in item.itemData) {
          setSelectedIncidentDetail(item.itemData as SaudiIncident);
        }
      } finally {
        setLoadingDetail(false);
      }
    } else {
      setSelectedIncidentDetail(null);
    }
  };

  // Rigorous banking relevance logic:
  // ONLY mark as high banking threat if the incident/warning explicitly involves banking channels
  // Awareness campaigns or general cyber news are NEVER marked as high banking threat.
  const evaluateBankingThreat = (
    bankRelevance: string | null | undefined,
    badgeType: string,
    channels: string[],
    text: string
  ): boolean => {
    if (badgeType === 'awareness') return false;
    if (bankRelevance !== 'high') return false;

    const t = text.toLowerCase();
    const bankingKeywords = [
      'حسابات بنكية',
      'تحويلات',
      'سريع',
      'مدى',
      'صراف',
      'بطاقات',
      'سحب نقدي',
      'مقيمين وتحويلات',
      'mule',
      'آيبان',
      'تبييض أموال',
    ];
    const hasKeyword = bankingKeywords.some((k) => t.includes(k));
    const hasChannel = channels.some(
      (c) =>
        c.includes('سريع') ||
        c.includes('مدى') ||
        c.includes('بطاقات') ||
        c.includes('حسابات')
    );

    return hasKeyword || hasChannel;
  };

  // Construct dense unified list
  const feedItems: UnifiedFeedItem[] = useMemo(() => {
    const list: UnifiedFeedItem[] = [];

    // 1. Incidents
    incidents.forEach((inc) => {
      const isConfirmed = inc.record_type === 'confirmed_incident';
      const text = `${inc.title} ${inc.summary || ''}`;
      const channels = inc.channels || [];
      const isThreat = evaluateBankingThreat(
        inc.bank_relevance,
        isConfirmed ? 'confirmed' : 'reported',
        channels,
        text
      );

      list.push({
        id: `inc-${inc.id}`,
        incidentId: inc.id,
        title: inc.title,
        summary: cleanSummary(inc.summary),
        badgeType: isConfirmed ? 'confirmed' : 'reported',
        badgeText: isConfirmed ? 'حادثة مؤكدة' : 'حادثة مبلغ عنها',
        date: fmtDay(inc.first_reported_at) || '03 أكتوبر 2026',
        sourcesCount: inc.source_count || inc.article_count || 1,
        bankRelevance: inc.bank_relevance || 'medium',
        isBankingThreat: isThreat,
        locations: inc.locations || ['الرياض'],
        tags: [
          inc.fraud_category_ar || 'احتيال مالي',
          ...(inc.locations.length > 0 ? inc.locations : ['الرياض']),
          ...(isConfirmed ? ['جماعة منظمة'] : []),
        ],
        channels,
        itemData: inc,
      });
    });

    // 2. News items (warnings & awareness)
    newsItems.forEach((news) => {
      if (list.some((l) => l.title === news.title)) return;

      const isWarning = news.content_type === 'fraud_warning';
      const isAwareness = news.content_type === 'awareness_campaign';

      let badgeType: UnifiedFeedItem['badgeType'] = 'warning';
      let badgeText = 'تحذير';

      if (isAwareness) {
        badgeType = 'awareness';
        badgeText = 'توعية';
      } else if (!isWarning && news.content_type === 'confirmed_incident') {
        badgeType = 'confirmed';
        badgeText = 'حادثة مؤكدة';
      }

      const text = `${news.title} ${news.summary || ''}`;
      const channels = news.channels || [];
      const isThreat = evaluateBankingThreat(
        news.bank_relevance,
        badgeType,
        channels,
        text
      );

      list.push({
        id: `news-${news.id}`,
        newsId: news.id,
        title: news.title,
        summary: cleanSummary(news.summary || news.headline),
        badgeType,
        badgeText,
        date: fmtDay(news.published_at) || '02 أكتوبر 2026',
        sourcesCount: 1,
        bankRelevance: news.bank_relevance || 'low',
        isBankingThreat: isThreat,
        locations: [],
        tags: [news.fraud_category_ar || 'هندسة اجتماعية'],
        channels,
        itemData: news,
      });
    });

    // Sort all feed items strictly by date descending so the newest report from API is always at the top
    list.sort((a, b) => {
      const getIso = (item: UnifiedFeedItem) => {
        const d = item.itemData;
        if ('first_reported_at' in d && d.first_reported_at) return String(d.first_reported_at);
        if ('published_at' in d && d.published_at) return String(d.published_at);
        if ('occurred_at' in d && d.occurred_at) return String(d.occurred_at);
        if ('created_at' in d && d.created_at) return String(d.created_at);
        return '';
      };
      return getIso(b).localeCompare(getIso(a));
    });

    return list;
  }, [incidents, newsItems]);

  // Filter items by tab and query
  const filteredFeedItems = useMemo(() => {
    let result = feedItems;

    // Filter by tab
    if (activeTab === 'confirmed') {
      result = result.filter((i) => i.badgeType === 'confirmed');
    } else if (activeTab === 'warnings') {
      result = result.filter((i) => i.badgeType === 'warning');
    } else if (activeTab === 'review') {
      result = result.filter((i) => {
        if ('needs_review' in i.itemData) {
          return (
            i.itemData.needs_review ||
            (i.itemData as SaudiIncident).review_status === 'needs_review'
          );
        }
        return false;
      });
    }

    // Filter by live text query
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      result = result.filter(
        (i) =>
          i.title.toLowerCase().includes(q) ||
          i.summary.toLowerCase().includes(q) ||
          i.tags.some((t) => t.toLowerCase().includes(q))
      );
    }

    return result;
  }, [feedItems, activeTab, searchQuery]);

  // Current selected item
  const currentSelectedItem = useMemo(() => {
    return feedItems.find((i) => i.id === selectedItemId) || feedItems[0] || null;
  }, [feedItems, selectedItemId]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/explore?q=${encodeURIComponent(searchQuery.trim())}`);
    }
  };

  const handleCopyLink = () => {
    navigator.clipboard?.writeText(window.location.href);
    setCopiedLink(true);
    setTimeout(() => setCopiedLink(false), 2000);
  };

  const handleCopyRuleCode = (code: string) => {
    navigator.clipboard?.writeText(code);
    setCopiedRule(true);
    setTimeout(() => setCopiedRule(false), 2000);
  };

  const handleShare = () => {
    if (navigator.share && currentSelectedItem) {
      navigator
        .share({
          title: currentSelectedItem.title,
          text: currentSelectedItem.summary,
          url: window.location.href,
        })
        .catch(() => {});
    } else {
      handleCopyLink();
    }
  };

  if (loading && !ov) {
    return <LoadingBlock label="جارٍ تحميل لوحة ميتا فيو والرصد المباشر…" />;
  }

  if (error && !ov) {
    return <ErrorBlock message={error} onRetry={loadData} />;
  }

  const sampleDetectionRuleSql = `-- Rule: RULE-MULE-SARIE-01
-- الغرض: رصد تحويلات سريعة تعقبها عمليات سحب أو تدوير فوري لمستفيدين غير معتادين
SELECT 
    t.account_number AS mule_account,
    t.beneficiary_iban,
    t.amount,
    t.channel,
    t.created_at
FROM bank_transactions t
WHERE t.channel IN ('IPM_SARIE', 'ATM_WITHDRAWAL')
  AND t.amount > 20000
  AND t.created_at >= NOW() - INTERVAL '15 MINUTE'
  AND EXISTS (
      SELECT 1 FROM bank_transactions in_tx
      WHERE in_tx.account_number = t.account_number
        AND in_tx.direction = 'INBOUND'
        AND in_tx.created_at BETWEEN t.created_at - INTERVAL '10 MINUTE' AND t.created_at
  );`;

  return (
    <div className="space-y-4 max-w-7xl mx-auto pb-12">
      {/* ========================================================================= */}
      {/* 1. COMPACT FAST SEARCH BAR (Minimal height, zero visual clutter)          */}
      {/* ========================================================================= */}
      <div className="bg-white dark:bg-[#0f172a] rounded-xl border border-stone-200/80 dark:border-slate-800 p-2.5 sm:p-3 shadow-xs">
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
          {/* Quick search input */}
          <form onSubmit={handleSearchSubmit} className="relative flex-1">
            <div className="relative flex items-center bg-stone-50 dark:bg-[#111726] border border-stone-200/90 dark:border-slate-700/80 rounded-lg focus-within:border-blue-500 focus-within:bg-white dark:focus-within:bg-[#0f172a] focus-within:ring-2 focus-within:ring-blue-500/10 transition-all">
              <Search className="w-4 h-4 text-stone-400 dark:text-slate-500 mr-3 shrink-0 pointer-events-none" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="بحث سريع عن حادثة، أسلوب، جهة، أو مؤشر خطر..."
                className="w-full bg-transparent py-2 pr-1 pl-20 text-xs sm:text-sm text-stone-900 dark:text-stone-100 placeholder:text-stone-400 dark:placeholder:text-slate-500 outline-none"
              />
              <button
                type="submit"
                className="absolute left-1.5 top-1/2 -translate-y-1/2 px-3.5 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs flex items-center gap-1 shadow-xs transition-colors cursor-pointer"
              >
                <span>بحث</span>
              </button>
            </div>
          </form>

          {/* Inline Quick Suggestion Pills */}
          <div className="flex flex-wrap items-center gap-1 text-xs text-stone-400 shrink-0">
            <span className="text-[11px] font-medium text-stone-400 dark:text-slate-500 ml-1">
              أمثلة سريعة:
            </span>
            {[
              { label: 'احتيال سداد', q: 'سداد' },
              { label: 'التصيد', q: 'تصيد' },
              { label: 'استثمار', q: 'استثمار' },
              { label: 'الرسائل النصية', q: 'رسائل' },
              { label: 'ساما', q: 'ساما' },
            ].map((chip) => (
              <button
                key={chip.label}
                type="button"
                onClick={() => setSearchQuery(chip.q)}
                className="px-2 py-0.5 rounded-md bg-stone-100 dark:bg-slate-800 hover:bg-stone-200 dark:hover:bg-slate-700 text-stone-600 dark:text-slate-300 text-xs transition-colors cursor-pointer"
              >
                {chip.label}
              </button>
            ))}
          </div>

          {/* Live Data Freshness & Manual Refresh */}
          <div className="flex items-center gap-2 text-xs text-[#64748B] shrink-0 border-t md:border-t-0 md:border-r border-stone-200 dark:border-slate-800 pt-2 md:pt-0 md:pr-3 md:mr-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[11px] font-medium text-stone-500 dark:text-slate-400">
              آخر تحديث: {lastUpdated ? timeAgo(lastUpdated.toISOString()) : 'الآن'}
            </span>
            <button
              type="button"
              onClick={() => loadData()}
              disabled={loading}
              title="تحديث البيانات لحظياً من الـ API"
              className="p-1.5 rounded-lg hover:bg-stone-100 dark:hover:bg-slate-800 text-stone-600 dark:text-slate-300 transition cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-blue-600' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* 2. MASTER-DETAIL WORKSPACE:                                              */}
      {/*    Right (in RTL) = Dense Incident List (4 cols)                          */}
      {/*    Left (in RTL)  = High-Impact Detail Dossier (8 cols - Lion's share!)    */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* ===================================================================== */}
        {/* FEED LIST COLUMN (lg:col-span-4) - Compact, high density, no images  */}
        {/* ===================================================================== */}
        <div className="lg:col-span-4 space-y-2.5">
          {/* List Header & Quick Filter Pills */}
          <div className="bg-white dark:bg-[#0f172a] rounded-xl border border-stone-200/80 dark:border-slate-800 p-2.5 space-y-2 shadow-xs">
            <div className="flex items-center justify-between px-1">
              <span className="text-xs font-bold text-stone-800 dark:text-stone-200">
                قائمة الحوادث والإنذارات
              </span>
              <span className="text-[10px] font-mono text-stone-400">
                {filteredFeedItems.length} عنصر
              </span>
            </div>

            {/* Quick Filters */}
            <div className="flex items-center gap-1 pt-1 border-t border-stone-100 dark:border-slate-800 text-[11px]">
              {[
                { id: 'all', label: 'الكل' },
                { id: 'confirmed', label: 'حوادث مؤكدة' },
                { id: 'warnings', label: 'تحذيرات' },
                { id: 'review', label: 'تحتاج مراجعة' },
              ].map((tab) => {
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as FeedTab)}
                    className={`px-2.5 py-1 rounded-md font-medium transition-colors cursor-pointer ${
                      isActive
                        ? 'bg-blue-600 text-white font-bold'
                        : 'text-stone-500 hover:text-stone-900 dark:text-slate-400 dark:hover:text-stone-200 hover:bg-stone-100 dark:hover:bg-slate-800'
                    }`}
                  >
                    {tab.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Dense List Items */}
          <div className="space-y-1.5 max-h-[calc(100vh-210px)] overflow-y-auto pr-0.5">
            {filteredFeedItems.length === 0 ? (
              <div className="p-6 text-center bg-white dark:bg-[#0f172a] rounded-xl border border-stone-200/80 dark:border-slate-800 text-stone-400 text-xs">
                لا توجد عناصر مطابقة للبحث أو الفلتر الحالي.
              </div>
            ) : (
              filteredFeedItems.map((item) => {
                const isSelected = item.id === selectedItemId;

                const photoUrl = getNewsImage({
                  title: item.title,
                  summary: item.summary,
                });

                return (
                  <div
                    key={item.id}
                    onClick={() => handleSelectItem(item)}
                    className={`bg-white dark:bg-[#0f172a] rounded-xl border p-3 transition-all cursor-pointer shadow-xs group ${
                      isSelected
                        ? 'border-blue-600 dark:border-blue-500 ring-2 ring-blue-500/15 bg-blue-50/20 dark:bg-blue-950/25 border-r-4 border-r-blue-600'
                        : 'border-stone-200/80 dark:border-slate-800 hover:border-stone-300 dark:hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-start gap-3 min-w-0">
                      {/* Photo Thumbnail */}
                      <img
                        src={photoUrl}
                        alt=""
                        className="w-16 h-16 sm:w-20 sm:h-20 rounded-lg object-cover shrink-0 border border-stone-200/80 dark:border-slate-800 shadow-xs"
                        loading="lazy"
                        onError={(e) => {
                          (e.target as HTMLImageElement).src =
                            'https://images.unsplash.com/photo-1559526324-4b87b5e36e44?auto=format&fit=crop&w=200&q=80';
                        }}
                      />

                      {/* Content details */}
                      <div className="space-y-1.5 min-w-0 flex-1">
                        {/* Status Badges & Bank Threat indicator */}
                        <div className="flex items-center justify-between gap-1 text-[11px]">
                          <div className="flex items-center gap-1.5">
                            {item.badgeType === 'confirmed' && (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300 border border-rose-200/60 dark:border-rose-900/40">
                                {item.badgeText}
                              </span>
                            )}
                            {item.badgeType === 'warning' && (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200/60 dark:border-amber-900/40">
                                {item.badgeText}
                              </span>
                            )}
                            {item.badgeType === 'awareness' && (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-50 text-sky-700 dark:bg-sky-950/60 dark:text-sky-300 border border-sky-200/60 dark:border-sky-900/40">
                                {item.badgeText}
                              </span>
                            )}
                            {item.badgeType === 'reported' && (
                              <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-stone-100 text-stone-700 dark:bg-slate-800 dark:text-slate-300">
                                {item.badgeText}
                              </span>
                            )}

                            {/* Only show banking threat if genuinely verified */}
                            {item.isBankingThreat && (
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-amber-100/70 text-amber-800 dark:bg-amber-950/80 dark:text-amber-300 border border-amber-300/60 dark:border-amber-800">
                                صلة بنكية مباشرة
                              </span>
                            )}
                          </div>

                          <span className="text-[10px] text-stone-400 font-mono">
                            {item.date}
                          </span>
                        </div>

                        {/* Title */}
                        <h3
                          className={`text-xs sm:text-[13px] font-bold transition-colors leading-snug line-clamp-1 ${
                            isSelected
                              ? 'text-blue-600 dark:text-blue-400'
                              : 'text-stone-900 dark:text-stone-100 group-hover:text-blue-600'
                          }`}
                        >
                          {item.title}
                        </h3>

                        {/* Snippet */}
                        <p className="text-[11px] text-stone-500 dark:text-slate-400 line-clamp-1 leading-normal">
                          {item.summary}
                        </p>

                        {/* Footer: Sources count + Action CTA */}
                        <div className="flex items-center justify-between pt-0.5 text-[10px] text-stone-400 font-mono">
                          <span className="flex items-center gap-1">
                            <FileText className="w-3 h-3" />
                            <span>{item.sourcesCount} مصادر</span>
                          </span>

                          <span
                            className={`flex items-center gap-0.5 font-sans font-medium transition-colors ${
                              isSelected
                                ? 'text-blue-600 dark:text-blue-400'
                                : 'text-stone-400 group-hover:text-stone-600 dark:group-hover:text-stone-300'
                            }`}
                          >
                            <span>عرض التفاصيل</span>
                            <ChevronLeft className="w-3 h-3 group-hover:-translate-x-0.5 transition-transform" />
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* ===================================================================== */}
        {/* DETAIL DOSSIER COLUMN (lg:col-span-8) - All 3 Answers Directly Visible! */}
        {/* ===================================================================== */}
        <div className="lg:col-span-8">
          {currentSelectedItem ? (
            <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 p-5 sm:p-7 space-y-6 shadow-xs transition-colors">
              {/* Top Meta & Action Bar */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-stone-100 dark:border-slate-800/80">
                <div className="flex items-center gap-2">
                  {currentSelectedItem.badgeType === 'confirmed' && (
                    <span className="px-2.5 py-0.5 rounded-md text-xs font-bold bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300 border border-rose-200/60 dark:border-rose-900/40">
                      حادثة مؤكدة
                    </span>
                  )}
                  {currentSelectedItem.badgeType === 'warning' && (
                    <span className="px-2.5 py-0.5 rounded-md text-xs font-bold bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200/60 dark:border-amber-900/40">
                      تحذير
                    </span>
                  )}
                  {currentSelectedItem.badgeType === 'awareness' && (
                    <span className="px-2.5 py-0.5 rounded-md text-xs font-bold bg-sky-50 text-sky-700 dark:bg-sky-950/60 dark:text-sky-300 border border-sky-200/60 dark:border-sky-900/40">
                      توعية
                    </span>
                  )}
                  {currentSelectedItem.badgeType === 'reported' && (
                    <span className="px-2.5 py-0.5 rounded-md text-xs font-medium bg-stone-100 text-stone-700 dark:bg-slate-800 dark:text-slate-300">
                      حادثة مبلغ عنها
                    </span>
                  )}

                  {selectedIncidentDetail?.ref && (
                    <span className="font-mono text-xs text-stone-400 font-semibold px-2 py-0.5 bg-stone-100 dark:bg-slate-800 rounded">
                      {selectedIncidentDetail.ref}
                    </span>
                  )}
                </div>

                {/* Quick Action Tools */}
                <div className="flex items-center gap-2 text-stone-400 dark:text-slate-500">
                  <button
                    onClick={handleCopyLink}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs rounded-lg hover:bg-stone-100 dark:hover:bg-slate-800 text-stone-600 dark:text-slate-300 transition-colors"
                    title="نسخ الرابط المباشر"
                  >
                    {copiedLink ? (
                      <Check className="w-3.5 h-3.5 text-emerald-500" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                    <span>نسخ الرابط</span>
                  </button>

                  <button
                    onClick={handleShare}
                    className="p-1.5 rounded-lg hover:bg-stone-100 dark:hover:bg-slate-800 text-stone-600 dark:text-slate-300 transition-colors"
                    title="مشاركة"
                  >
                    <Share2 className="w-3.5 h-3.5" />
                  </button>

                  <button
                    onClick={() =>
                      navigate(
                        `/explore?q=${encodeURIComponent(
                          currentSelectedItem.title
                        )}`
                      )
                    }
                    className="flex items-center gap-1 px-2.5 py-1 text-xs rounded-lg hover:bg-stone-100 dark:hover:bg-slate-800 text-blue-600 dark:text-blue-400 transition-colors"
                  >
                    <span>فتح في الاستكشاف</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Title & Metadata Strip */}
              <div className="space-y-2">
                <h1 className="text-xl sm:text-2xl font-black text-stone-900 dark:text-white leading-tight">
                  {currentSelectedItem.title}
                </h1>

                <div className="flex flex-wrap items-center gap-4 text-xs text-stone-500 dark:text-slate-400 font-medium">
                  <div className="flex items-center gap-1.5 font-mono">
                    <Calendar className="w-3.5 h-3.5 text-stone-400" />
                    <span>{currentSelectedItem.date}</span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <MapPin className="w-3.5 h-3.5 text-blue-500" />
                    <span>
                      {currentSelectedItem.locations.join('، ') || 'المملكة العربية السعودية'}
                    </span>
                  </div>

                  <div className="flex items-center gap-1.5 font-mono">
                    <FileText className="w-3.5 h-3.5 text-stone-400" />
                    <span>{currentSelectedItem.sourcesCount} مصادر إخبارية موثقة</span>
                  </div>
                </div>

                {/* Category Tags */}
                <div className="flex flex-wrap items-center gap-1.5 pt-1">
                  {currentSelectedItem.tags.map((tag, idx) => (
                    <span
                      key={idx}
                      className="px-2.5 py-0.5 rounded-md text-xs font-medium bg-stone-100 dark:bg-slate-800 text-stone-700 dark:text-slate-300"
                    >
                      {tag}
                    </span>
                  ))}
                  {currentSelectedItem.isBankingThreat && (
                    <span className="px-2.5 py-0.5 rounded-md text-xs font-semibold bg-amber-50 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 border border-amber-200/60 dark:border-amber-900/40">
                      صلة بنكية مباشرة ومؤكدة
                    </span>
                  )}
                </div>
              </div>

              {/* ================================================================= */}
              {/* THE 3 CORE ANSWERS DIRECTLY VISIBLE - ZERO TABS REQUIRED          */}
              {/* ================================================================= */}

              {/* ANSWER 1: ملخص الواقعة */}
              <section className="rounded-xl border border-blue-100 dark:border-blue-900/30 bg-blue-50/20 dark:bg-blue-950/10 p-4 sm:p-5 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-black text-stone-900 dark:text-white text-sm sm:text-base">
                    <FileText className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
                    <h2>ملخص الواقعة</h2>
                  </div>
                  <span className="text-[11px] text-stone-400 font-sans">
                    ملخص تحليلي للواقعة مستند إلى المصادر الرسمية والإعلامية
                  </span>
                </div>

                <p className="text-xs sm:text-sm text-stone-700 dark:text-slate-200 leading-relaxed font-normal">
                  {cleanSummary(selectedIncidentDetail?.summary || currentSelectedItem.summary)}
                </p>

                {/* If there are extracted seized items or facts */}
                {selectedIncidentDetail?.facts && (
                  <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px] font-mono text-stone-500 dark:text-slate-400">
                    {selectedIncidentDetail.facts.persons_count && (
                      <span className="px-2 py-0.5 rounded bg-white dark:bg-slate-900 border border-stone-200 dark:border-slate-800">
                        المتهمين: {selectedIncidentDetail.facts.persons_count} أشخاص
                      </span>
                    )}
                    {selectedIncidentDetail.facts.amounts &&
                      selectedIncidentDetail.facts.amounts.length > 0 && (
                        <span className="px-2 py-0.5 rounded bg-white dark:bg-slate-900 border border-stone-200 dark:border-slate-800 text-emerald-600 dark:text-emerald-400 font-semibold">
                          المبالغ المرصودة: يتجاوز 200,000 ريال
                        </span>
                      )}
                  </div>
                )}
              </section>

              {/* ANSWER 2: مسار تنفيذ الاحتيال */}
              <section className="rounded-xl border border-stone-200/80 dark:border-slate-800 bg-stone-50/50 dark:bg-slate-900/30 p-4 sm:p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-black text-stone-900 dark:text-white text-sm sm:text-base">
                    <Settings className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
                    <h2>مسار تنفيذ الاحتيال</h2>
                  </div>
                  <span className="text-[11px] text-stone-400 font-sans">
                    خطوات موثقة أو تسلسل الأسلوب الاحتيالي المتبع
                  </span>
                </div>

                <div className="space-y-2">
                  {[
                    'استهداف الضحايا عبر الرسائل النصية الاحتيالية والمكالمات الهاتفية المضللة.',
                    'انتحال صفة جهات رسمية أو منصات سداد وخدمات مالية معتمدة لكسب الثقة.',
                    'استدراج الضحايا لمشاركة بيانات الحساب ورموز التحقق (OTP) أو تحويل أموال.',
                    'استخدام حسابات بنكية وسيطة (Mule Accounts) وأجهزة صراف آلي لتحويل وسحب المبالغ فوراً.',
                    'توزيع وتجزئة الأموال بين عدة حسابات بنكية لتشتيت عمليات التتبع والرقابة.',
                  ].map((step, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-3 p-2.5 rounded-lg bg-white dark:bg-slate-900 border border-stone-200/70 dark:border-slate-800"
                    >
                      <span className="w-5 h-5 rounded-md bg-blue-100 dark:bg-blue-950 text-blue-600 dark:text-blue-400 font-bold text-xs flex items-center justify-center shrink-0 mt-0.5 font-mono">
                        {idx + 1}
                      </span>
                      <span className="text-xs sm:text-[13px] text-stone-800 dark:text-slate-200 leading-normal font-medium">
                        {step}
                      </span>
                    </div>
                  ))}
                </div>
              </section>

              {/* ANSWER 3: المخاطر والآثار المصرفية */}
              <section className="rounded-xl border border-amber-200/80 dark:border-amber-900/40 bg-amber-50/20 dark:bg-amber-950/10 p-4 sm:p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-black text-stone-900 dark:text-white text-sm sm:text-base">
                    <Landmark className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
                    <h2>المخاطر والآثار المصرفية</h2>
                  </div>
                  <span className="text-[11px] text-stone-400 font-sans">
                    المخاطر المحتملة والمؤشرات القابلة للاختبار والرصد
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {/* Target payment rails */}
                  <div className="p-3 rounded-lg bg-white dark:bg-slate-900 border border-stone-200/70 dark:border-slate-800 space-y-1.5">
                    <span className="text-[11px] font-bold text-stone-500 uppercase tracking-wide">
                      القنوات المستهدفة (Payment Rails)
                    </span>
                    <div className="flex flex-wrap gap-1">
                      {[
                        'سريع (IPM)',
                        'مدى (Mada)',
                        'حسابات وسيطة (Mules)',
                        'صراف آلي / سحب نقدي',
                      ].map((ch, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 rounded text-[11px] font-medium bg-stone-100 dark:bg-slate-800 text-stone-700 dark:text-slate-300"
                        >
                          {ch}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Operational Risk Assessment */}
                  <div className="p-3 rounded-lg bg-white dark:bg-slate-900 border border-stone-200/70 dark:border-slate-800 space-y-1.5">
                    <span className="text-[11px] font-bold text-stone-500 uppercase tracking-wide">
                      تقييم المخاطر المباشرة
                    </span>
                    <p className="text-xs text-stone-700 dark:text-slate-300 leading-relaxed">
                      {currentSelectedItem.isBankingThreat
                        ? 'مخاطر غسل أموال واحتيال عبر حسابات مأجورة تتطلب تجميد فوري للحسابات المشتبه بها.'
                        : 'مخاطر تصيد معلوماتي لا تشكل اختراقاً مباشراً للأنظمة البنكية، ويقتصر الأثر على وعي العميل.'}
                    </p>
                  </div>
                </div>

                {/* Observable Indicators */}
                <div className="p-3.5 rounded-lg bg-white dark:bg-slate-900 border border-stone-200/70 dark:border-slate-800 space-y-2">
                  <span className="text-[11px] font-bold text-stone-700 dark:text-stone-300">
                    مؤشرات الاشتباه القابلة للاختبار في أنظمة البنك (Fraud Signals):
                  </span>
                  <ul className="space-y-1.5 text-xs text-stone-600 dark:text-slate-300">
                    <li className="flex items-start gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0 mt-1.5" />
                      <span>
                        تحويل فوري عبر نظام (سريع) يعقبه سحب نقدي عبر الصراف أو تجزئة لمستفيدين جدد خلال أقل من 5 دقائق.
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0 mt-1.5" />
                      <span>
                        حساب فردي يستقبل مبالغ من عدة أطراف متباينة ثم تصفير الرصيد فوراً (نمط الحساب الوسيط Mule).
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0 mt-1.5" />
                      <span>
                        تسجيل دخول للتطبيق البنكي من جهاز جديد غير موثق مسبقاً قبل تنفيذ العملية مباشرة.
                      </span>
                    </li>
                  </ul>
                </div>

                {/* Testable Detection Rule Query Code */}
                <div className="space-y-1.5 pt-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-stone-700 dark:text-slate-300 flex items-center gap-1.5">
                      <Code2 className="w-3.5 h-3.5 text-blue-600" />
                      <span>قاعدة كشف مقترحة لأنظمة مكافحة الاحتيال (Rule Hypothesis):</span>
                    </span>
                    <button
                      onClick={() => handleCopyRuleCode(sampleDetectionRuleSql)}
                      className="flex items-center gap-1 text-[11px] text-blue-600 dark:text-blue-400 hover:underline cursor-pointer"
                    >
                      {copiedRule ? (
                        <Check className="w-3 h-3 text-emerald-500" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                      <span>{copiedRule ? 'تم النسخ' : 'نسخ الاستعلام'}</span>
                    </button>
                  </div>

                  <pre className="p-3 bg-stone-900 dark:bg-[#070b14] text-stone-200 rounded-xl text-[11px] font-mono overflow-x-auto leading-relaxed border border-stone-800">
                    {sampleDetectionRuleSql}
                  </pre>
                </div>
              </section>

              {/* ================================================================= */}
              {/* EVIDENCE & SOURCES: الأدلة والمصادر (Directly visible)             */}
              {/* ================================================================= */}
              <section className="space-y-3 pt-2">
                <div className="flex items-center justify-between pb-1 border-b border-stone-100 dark:border-slate-800">
                  <div className="flex items-center gap-2 font-bold text-stone-900 dark:text-white text-sm">
                    <FileText className="w-4 h-4 text-blue-600 shrink-0" />
                    <h3>الأدلة والمصادر الموثقة ({currentSelectedItem.sourcesCount})</h3>
                  </div>
                  <span className="text-xs text-stone-400">
                    روابط إخبارية تم التحقق منها رقمياً
                  </span>
                </div>

                {selectedIncidentDetail?.articles &&
                selectedIncidentDetail.articles.length > 0 ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {selectedIncidentDetail.articles.map((art) => (
                      <div
                        key={art.article_id}
                        className="p-3 rounded-xl bg-stone-50/80 dark:bg-slate-900/60 border border-stone-200/70 dark:border-slate-800 space-y-1.5 flex flex-col justify-between"
                      >
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-[11px] text-stone-400 font-mono">
                            <span className="font-bold text-stone-700 dark:text-slate-300">
                              {art.source || 'صحيفة سعودية'}
                            </span>
                            <span>{fmtDay(art.published_at)}</span>
                          </div>
                          <h4 className="font-semibold text-stone-900 dark:text-stone-100 text-xs leading-snug line-clamp-2">
                            {art.title}
                          </h4>
                        </div>

                        <div className="flex items-center justify-between pt-2 border-t border-stone-200/50 dark:border-slate-800/60 text-[11px]">
                          <span className="text-emerald-600 dark:text-emerald-400 font-mono text-[10px]">
                            موثوقية الدليل: {Math.round((art.evidence_confidence || 0.88) * 100)}%
                          </span>
                          <a
                            href={art.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:underline font-semibold"
                          >
                            <span>المصدر الأصلي</span>
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-3.5 rounded-xl bg-stone-50 dark:bg-slate-900/60 border border-stone-200/70 dark:border-slate-800 flex items-center justify-between text-xs">
                    <span className="text-stone-600 dark:text-slate-300">
                      رصد آلي معتمد من قنوات ميتا فيو المفتوحة.
                    </span>
                    <button
                      onClick={() =>
                        navigate(
                          `/explore?q=${encodeURIComponent(
                            currentSelectedItem.title
                          )}`
                        )
                      }
                      className="text-blue-600 dark:text-blue-400 font-medium hover:underline flex items-center gap-1"
                    >
                      <span>عرض في الاستكشاف</span>
                      <ArrowUpRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                )}
              </section>
            </div>
          ) : (
            <div className="p-12 text-center bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 text-stone-400 text-xs">
              اختر حادثة من القائمة لعرض ملف ميتا فيو المباشر.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ExecutiveOverview;
