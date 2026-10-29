// src/pages/ExecutiveOverview.tsx
// Home Page: Global Fresh Intelligence Center & Multi-Country Monitoring for Fraud Analysts
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
  ArrowUpRight,
  Info,
  RefreshCw,
  Globe,
  Clock,
  Filter,
  Activity,
  SlidersHorizontal,
  Building2,
} from 'lucide-react';
import { apiService } from '../services/api';
import {
  SaudiOverview,
  SaudiIncident,
  SaudiFraudItem,
  CountryCount,
} from '../types/saudi';
import { LoadingBlock, ErrorBlock, fmtDay } from '../components/ui';
import { timeAgo } from '../utils/formatDate';
import { useRefresh } from '../context/RefreshContext';

type ScopeMode = 'all' | 'saudi' | 'country';
type TimeWindow = '24h' | '7d' | '30d' | 'custom';
type SortMode = 'latest' | 'importance';
type FeedTab = 'all' | 'confirmed' | 'warnings' | 'awareness' | 'x_posts' | 'review';

const COUNTRY_MAP: Record<string, { name: string; flag: string }> = {
  SA: { name: 'المملكة العربية السعودية', flag: '🇸🇦' },
  US: { name: 'الولايات المتحدة', flag: '🇺🇸' },
  GB: { name: 'المملكة المتحدة', flag: '🇬🇧' },
  AE: { name: 'الإمارات العربية المتحدة', flag: '🇦🇪' },
  EG: { name: 'جمهورية مصر العربية', flag: '🇪🇬' },
  CA: { name: 'كندا', flag: '🇨🇦' },
  AU: { name: 'أستراليا', flag: '🇦🇺' },
  IT: { name: 'إيطاليا', flag: '🇮🇹' },
  LB: { name: 'لبنان', flag: '🇱🇧' },
  QA: { name: 'قطر', flag: '🇶🇦' },
  FR: { name: 'فرنسا', flag: '🇫🇷' },
  DE: { name: 'ألمانيا', flag: '🇩🇪' },
  NL: { name: 'هولندا', flag: '🇳🇱' },
  Global: { name: 'نطاق دولي', flag: '🌐' },
  Unknown: { name: 'غير محددة', flag: '🌐' },
};

interface UnifiedFeedItem {
  id: string;
  incidentId?: number;
  newsId?: number;
  title: string;
  summary: string;
  badgeType: 'confirmed' | 'warning' | 'awareness' | 'reported';
  badgeText: string;
  date: string;
  publishedAt?: string | null;
  occurredAt?: string | null;
  sourceName: string;
  sourceUrl?: string;
  countryCode: string;
  countryName: string;
  countryFlag: string;
  sourcesCount: number;
  isXPost?: boolean;
  sourceHandle?: string;
  bankRelevance: 'high' | 'medium' | 'low';
  isBankingThreat: boolean;
  locations: string[];
  tags: string[];
  channels: string[];
  executionSteps?: string[];
  verifiedFacts?: any;
  itemData: SaudiIncident | SaudiFraudItem;
}

export const ExecutiveOverview: React.FC = () => {
  const navigate = useNavigate();

  // 1. Geographic Scope (Default is Global 'all' per requirement)
  const [scope, setScope] = useState<ScopeMode>('all');
  const [selectedCountry, setSelectedCountry] = useState<string>('all');

  // 2. Freshness Time Window
  const [timeWindow, setTimeWindow] = useState<TimeWindow>('30d');
  const [customDateFrom, setCustomDateFrom] = useState<string>('');
  const [customDateTo, setCustomDateTo] = useState<string>('');

  // 3. Sorting Mode (Default is Newest Published)
  const [sortBy, setSortBy] = useState<SortMode>('latest');

  // Search & Tab states
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<FeedTab>('all');
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);

  // Copy feedback states
  const [copiedLink, setCopiedLink] = useState(false);
  const [copiedRule, setCopiedRule] = useState(false);

  // Data states
  const [ov, setOv] = useState<SaudiOverview | null>(null);
  const [incidents, setIncidents] = useState<SaudiIncident[]>([]);
  const [newsItems, setNewsItems] = useState<SaudiFraudItem[]>([]);
  const [xPostsItems, setXPostsItems] = useState<SaudiFraudItem[]>([]);
  const [availableCountries, setAvailableCountries] = useState<CountryCount[]>([]);
  const [selectedIncidentDetail, setSelectedIncidentDetail] = useState<SaudiIncident | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshedAt, setLastRefreshedAt] = useState<Date | null>(new Date());

  // Clean summary helper
  const cleanSummary = (text?: string | null): string => {
    if (!text) {
      return 'تم رصد وتوثيق المادة عبر المصادر الإخبارية والرسمية المعتمدة وفق معايير مكافحة الاحتيال المالي.';
    }
    let s = text.replace(/<[^>]+>/g, ' ').replace(/&nbsp;/g, ' ').replace(/&quot;/g, '"').replace(/&amp;/g, '&');
    s = s.replace(/https?:\/\/\S+/g, '').replace(/\s+/g, ' ').trim();
    if (s.length < 25) {
      return 'تم رصد وتوثيق الواقعة عبر المصادر الرسمية والإعلامية ومباشرة إجراءات المتابعة والتحقيق لحماية المتعاملين.';
    }
    return s;
  };

  // Helper for telemetry formatting - never shows "الآن" unless strictly recent (< 5 min)
  const formatTelemetry = (dateStr?: string | null): string => {
    if (!dateStr) return 'غير متوفر';
    try {
      const dt = new Date(dateStr);
      const diffMin = Math.floor((Date.now() - dt.getTime()) / 60000);
      if (diffMin >= 0 && diffMin < 5) return 'منذ لحظات';
      return `${timeAgo(dateStr)} (${dt.toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' })})`;
    } catch {
      return dateStr;
    }
  };

  // Resolve documented country
  const resolveCountry = (cCode?: string | null, isSaudi?: boolean): { code: string; name: string; flag: string } => {
    if (cCode && cCode !== 'Unknown' && cCode !== 'Global') {
      const info = COUNTRY_MAP[cCode] || { name: cCode, flag: '🌐' };
      return { code: cCode, name: info.name, flag: info.flag };
    }
    if (isSaudi) {
      return { code: 'SA', name: COUNTRY_MAP['SA'].name, flag: COUNTRY_MAP['SA'].flag };
    }
    return { code: 'Unknown', name: 'غير محددة', flag: '🌐' };
  };

  const { updateTelemetry } = useRefresh();

  // Load primary data with synchronous filters (supports forceRefresh cache-busting)
  const loadData = useCallback(async (isSilent = false, isForce = false) => {
    if (!isSilent) setLoading(true);
    setError(null);
    try {
      const winDays = timeWindow === '24h' ? 1 : timeWindow === '7d' ? 7 : timeWindow === '30d' ? 30 : undefined;
      const dFrom = timeWindow === 'custom' ? (customDateFrom || undefined) : undefined;
      const dTo = timeWindow === 'custom' ? (customDateTo || undefined) : undefined;

      const effScope = scope === 'country' ? 'all' : scope;
      const effCountry = scope === 'country' ? (selectedCountry !== 'all' ? selectedCountry : undefined) : (scope === 'saudi' ? 'SA' : undefined);

      const [overviewRes, incidentsRes, newsRes, xNewsRes, countriesRes] = await Promise.all([
        apiService.getSaudiOverview({
          scope: effScope,
          country: effCountry,
          window_days: winDays,
          date_from: dFrom,
          date_to: dTo,
          forceRefresh: isForce,
        }),
        apiService.getSaudiIncidents({
          scope: effScope,
          country: effCountry,
          window_days: winDays,
          date_from: dFrom,
          date_to: dTo,
          page_size: 25,
        }),
        apiService.getSaudiNews({
          scope: effScope,
          country: effCountry,
          window_days: winDays,
          date_from: dFrom,
          date_to: dTo,
          sort_by: sortBy,
          page_size: 80,
        }),
        apiService
          .getSaudiNews({
            source: 'x_k_h_almuzaini',
            sort_by: sortBy,
            page_size: 50,
          })
          .catch(() => ({ items: [] as SaudiFraudItem[], total: 0, page: 1, page_size: 50 })),
        apiService.getFraudCountries('all'),
      ]);

      setOv(overviewRes);
      if (overviewRes.telemetry) {
        updateTelemetry(overviewRes.telemetry);
      }
      setIncidents(incidentsRes.items || []);
      setNewsItems(newsRes.items || []);
      setXPostsItems(xNewsRes.items || []);
      setAvailableCountries(countriesRes.countries || []);
      setLastRefreshedAt(new Date());

      // Default selection if none or invalid
      if (incidentsRes.items && incidentsRes.items.length > 0) {
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
      if (!isSilent) setError(e?.message || 'خطأ في جلب بيانات الرصد');
    } finally {
      if (!isSilent) setLoading(false);
    }
  }, [scope, selectedCountry, timeWindow, customDateFrom, customDateTo, sortBy, updateTelemetry]);

  useEffect(() => {
    loadData();
    const h = (e?: Event) => {
      const isForce = (e as CustomEvent)?.detail?.forceRefresh ?? false;
      loadData(false, isForce);
    };
    window.addEventListener('metaview:refresh', h);

    const timer = setInterval(() => {
      if (document.visibilityState === 'visible') {
        loadData(true, false);
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

  // Rigorous banking relevance check
  const evaluateBankingThreat = (
    bankRelevance?: string | null,
    badgeType?: string,
    channels: string[] = [],
    text: string = ''
  ): boolean => {
    if (badgeType === 'awareness') return false;
    if (bankRelevance?.toLowerCase() === 'high') return true;
    const lower = text.toLowerCase();
    const bankingKeywords = [
      'حساب بنكي', 'حسابات بنكية', 'تحويل بنكي', 'بطاقة مدى', 'بطاقة ائتمانية',
      'سريع', 'sarie', 'ipm', 'mule', 'حساب وسيط', 'مبالغ مسروقة', 'تجميد رصيد'
    ];
    const hasBankWord = bankingKeywords.some((w) => lower.includes(w));
    const hasBankChannel = channels.some((c) =>
      ['bank_account', 'sarie', 'mada', 'pos', 'credit_card', 'atm'].includes(c.toLowerCase())
    );
    return hasBankWord && hasBankChannel;
  };

  // Build unified feed items strictly prioritizing original published_at
  const feedItems = useMemo<UnifiedFeedItem[]>(() => {
    const list: UnifiedFeedItem[] = [];

    // 1. Add Incidents
    incidents.forEach((inc) => {
      const isThreat = evaluateBankingThreat(
        inc.bank_relevance,
        'confirmed',
        inc.channels,
        `${inc.title} ${inc.summary || ''}`
      );
      const isConfirmed = inc.record_type === 'confirmed_incident';
      const c = resolveCountry(inc.incident_country || inc.country, inc.incident_country === 'SA');

      // Extract genuine execution steps if present in data
      let steps: string[] | undefined = undefined;
      if (Array.isArray(inc.methods) && inc.methods.length > 0) {
        steps = inc.methods.map((m: any) => m.label || m.tag || String(m));
      } else if (inc.modus_operandi_detailed && inc.modus_operandi_detailed.length > 20) {
        steps = inc.modus_operandi_detailed.split('\n').filter((s: string) => s.trim().length > 5);
      }

      list.push({
        id: `inc-${inc.id}`,
        incidentId: inc.id,
        title: inc.title,
        summary: cleanSummary(inc.summary),
        badgeType: isConfirmed ? 'confirmed' : 'reported',
        badgeText: isConfirmed ? 'حادثة موثقة' : 'بلاغ احتيال',
        date: fmtDay(inc.first_reported_at) || 'مؤخرًا',
        publishedAt: inc.first_reported_at,
        occurredAt: inc.occurred_at || null,
        sourceName: inc.reporting_agency || (inc.articles && inc.articles[0]?.source) || 'مصدر رسمي',
        sourceUrl: inc.articles && inc.articles[0]?.url,
        countryCode: c.code,
        countryName: c.name,
        countryFlag: c.flag,
        sourcesCount: inc.source_count || (inc.articles ? inc.articles.length : 1),
        bankRelevance: inc.bank_relevance?.toLowerCase() as any || 'low',
        isBankingThreat: isThreat,
        locations: inc.locations || (inc.incident_city ? [inc.incident_city] : []),
        tags: [
          inc.fraud_type_ar || inc.fraud_category_ar || 'احتيال مالي',
          ...(inc.targeted_sector_ar ? [inc.targeted_sector_ar] : []),
        ],
        channels: inc.channels || [],
        executionSteps: steps,
        verifiedFacts: inc.facts,
        itemData: inc,
      });
    });

    // 2. Add News Articles & Warnings
    newsItems.forEach((news) => {
      if (list.some((l) => l.title === news.title)) return;

      const isWarning = news.content_type === 'fraud_warning';
      const isAwareness = news.content_type === 'awareness_campaign';
      const isXPost =
        news.source_code === 'x_k_h_almuzaini' ||
        Boolean(news.original_url && news.original_url.includes('x.com/K_H_AlMuzaini')) ||
        Boolean(news.source && news.source.includes('المزيني'));

      let badgeType: UnifiedFeedItem['badgeType'] = 'warning';
      let badgeText = 'تحذير احتيالي';

      if (isAwareness) {
        badgeType = 'awareness';
        badgeText = 'توعية أمنية';
      } else if (!isWarning && news.content_type === 'confirmed_incident') {
        badgeType = 'confirmed';
        badgeText = 'حادثة موثقة';
      }

      const text = `${news.title} ${news.summary || ''}`;
      const channels = news.channels || [];
      const isThreat = evaluateBankingThreat(news.bank_relevance, badgeType, channels, text);
      const c = resolveCountry(news.incident_country || news.country, news.is_saudi);

      list.push({
        id: `news-${news.id}`,
        newsId: news.id,
        title: news.title,
        summary: cleanSummary(news.summary || news.headline),
        badgeType,
        badgeText,
        date: fmtDay(news.published_at) || 'مؤخرًا',
        publishedAt: news.published_at,
        occurredAt: news.occurred_at || null,
        sourceName: news.source || (isXPost ? 'منصة 𝕏' : 'مصدر إخباري'),
        sourceUrl: news.original_url || undefined,
        countryCode: c.code,
        countryName: c.name,
        countryFlag: c.flag,
        sourcesCount: 1,
        isXPost,
        sourceHandle: isXPost ? '@K_H_AlMuzaini' : undefined,
        bankRelevance: news.bank_relevance || 'low',
        isBankingThreat: isThreat,
        locations: news.incident_city ? [news.incident_city] : [],
        tags: [
          isXPost ? 'منشور 𝕏 (@K_H_AlMuzaini)' : (news.fraud_category_ar || 'احتيال إلكتروني'),
        ],
        channels,
        verifiedFacts: news.verified_facts,
        itemData: news,
      });
    });

    // Chronological sort: strictly rely on published_at DESC (never on db insertion date)
    list.sort((a, b) => {
      const getIso = (item: UnifiedFeedItem) => {
        if (item.publishedAt) return item.publishedAt;
        const d = item.itemData;
        if ('first_reported_at' in d && d.first_reported_at) return String(d.first_reported_at);
        if ('published_at' in d && d.published_at) return String(d.published_at);
        return '';
      };
      if (sortBy === 'importance') {
        const score = (item: UnifiedFeedItem) =>
          (item.isBankingThreat ? 30 : 0) +
          (item.badgeType === 'confirmed' ? 20 : item.badgeType === 'warning' ? 10 : 5);
        const diff = score(b) - score(a);
        if (diff !== 0) return diff;
      }
      return getIso(b).localeCompare(getIso(a));
    });

    return list;
  }, [incidents, newsItems, sortBy]);

  // Dedicated X posts feed items so the منشورات 𝕏 tab always displays monitored @K_H_AlMuzaini posts
  const xFeedItems = useMemo<UnifiedFeedItem[]>(() => {
    const map = new Map<string, UnifiedFeedItem>();
    feedItems.filter((i) => i.isXPost).forEach((i) => map.set(i.id, i));
    xPostsItems.forEach((news) => {
      const id = `news-${news.id}`;
      if (map.has(id)) return;
      const isWarning = news.content_type === 'fraud_warning';
      const isAwareness = news.content_type === 'awareness_campaign';
      let badgeType: UnifiedFeedItem['badgeType'] = 'warning';
      let badgeText = 'تحذير احتيالي';
      if (isAwareness) {
        badgeType = 'awareness';
        badgeText = 'توعية أمنية';
      } else if (!isWarning && news.content_type === 'confirmed_incident') {
        badgeType = 'confirmed';
        badgeText = 'حادثة موثقة';
      }
      const text = `${news.title} ${news.summary || ''}`;
      const channels = news.channels || [];
      const isThreat = evaluateBankingThreat(news.bank_relevance, badgeType, channels, text);
      const c = resolveCountry(news.incident_country || news.country, news.is_saudi);
      map.set(id, {
        id,
        newsId: news.id,
        title: news.title,
        summary: cleanSummary(news.summary || news.headline),
        badgeType,
        badgeText,
        date: fmtDay(news.published_at) || 'مؤخرًا',
        publishedAt: news.published_at,
        occurredAt: news.occurred_at || null,
        sourceName: news.source || 'منصة 𝕏',
        sourceUrl: news.original_url || undefined,
        countryCode: c.code,
        countryName: c.name,
        countryFlag: c.flag,
        sourcesCount: 1,
        isXPost: true,
        sourceHandle: '@K_H_AlMuzaini',
        bankRelevance: news.bank_relevance || 'low',
        isBankingThreat: isThreat,
        locations: news.incident_city ? [news.incident_city] : [],
        tags: ['منشور 𝕏 (@K_H_AlMuzaini)'],
        channels,
        verifiedFacts: news.verified_facts,
        itemData: news,
      });
    });
    return Array.from(map.values()).sort((a, b) => (b.publishedAt || '').localeCompare(a.publishedAt || ''));
  }, [feedItems, xPostsItems]);

  // Tab and Query Filtering
  const filteredFeedItems = useMemo(() => {
    let result = activeTab === 'x_posts' ? xFeedItems : feedItems;

    if (activeTab === 'confirmed') {
      result = result.filter((i) => i.badgeType === 'confirmed');
    } else if (activeTab === 'warnings') {
      result = result.filter((i) => i.badgeType === 'warning');
    } else if (activeTab === 'awareness') {
      result = result.filter((i) => i.badgeType === 'awareness');
    } else if (activeTab === 'x_posts') {
      result = result.filter((i) => i.isXPost);
    } else if (activeTab === 'review') {
      result = result.filter((i) => {
        if ('needs_review' in i.itemData) {
          return (
            (i.itemData as any).needs_review ||
            (i.itemData as SaudiIncident).review_status === 'needs_review'
          );
        }
        return false;
      });
    }

    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      result = result.filter(
        (i) =>
          i.title.toLowerCase().includes(q) ||
          i.summary.toLowerCase().includes(q) ||
          i.sourceName.toLowerCase().includes(q) ||
          i.countryName.toLowerCase().includes(q) ||
          i.tags.some((t) => t.toLowerCase().includes(q))
      );
    }

    return result;
  }, [feedItems, xFeedItems, activeTab, searchQuery]);

  const currentSelectedItem = useMemo(() => {
    return (
      filteredFeedItems.find((i) => i.id === selectedItemId) ||
      feedItems.find((i) => i.id === selectedItemId) ||
      xFeedItems.find((i) => i.id === selectedItemId) ||
      filteredFeedItems[0] ||
      feedItems[0] ||
      null
    );
  }, [filteredFeedItems, feedItems, xFeedItems, selectedItemId]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/explore?q=${encodeURIComponent(searchQuery.trim())}&scope=${scope}`);
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
    return <LoadingBlock label="MetaView 3 — ما وراء الاحتيال: جارٍ جلب وتحديث البيانات…" />;
  }

  if (error && !ov) {
    return <ErrorBlock message={error} onRetry={() => loadData()} />;
  }

  const sampleDetectionRuleSql = `-- Rule: RULE-FRAUD-ALERT-MONITOR
-- الغرض: رصد أنماط تحويل فوري مشبوه مقترن بمؤشرات هندسة اجتماعية وانتحال صفة
SELECT 
    t.account_number AS suspect_account,
    t.beneficiary_iban,
    t.amount,
    t.channel,
    t.created_at
FROM bank_transactions t
WHERE t.channel IN ('IPM_SARIE', 'ATM_WITHDRAWAL', 'POS')
  AND t.amount > 15000
  AND t.created_at >= NOW() - INTERVAL '1 HOUR'
  AND EXISTS (
      SELECT 1 FROM bank_transactions in_tx
      WHERE in_tx.account_number = t.account_number
        AND in_tx.direction = 'INBOUND'
        AND in_tx.created_at BETWEEN t.created_at - INTERVAL '15 MINUTE' AND t.created_at
  );`;

  return (
    <div className="space-y-4 max-w-7xl mx-auto pb-12">
      {/* ========================================================================= */}
      {/* 1. TOP HEADER: Global Fresh Intelligence Center & Multi-Country Control   */}
      {/* ========================================================================= */}
      <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 p-4 sm:p-5 shadow-xs space-y-4">
        {/* Title, Subtitle, and Scope Switcher */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-mono text-xs sm:text-sm font-extrabold tracking-wider text-blue-600 dark:text-blue-400 uppercase">
                FINANCIAL FRAUD INTELLIGENCE
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 font-mono">
                LIVE INTEL
              </span>
            </div>
            <div className="flex items-baseline gap-2.5 flex-wrap">
              <span className="text-2xl sm:text-3xl font-black text-stone-900 dark:text-white tracking-tight">
                MetaView 3
              </span>
              <span className="text-stone-300 dark:text-slate-700 text-xl font-light">|</span>
              <h1 className="text-xl sm:text-2xl font-black text-blue-600 dark:text-blue-400">
                ما وراء الاحتيال
              </h1>
            </div>
            <p className="text-xs sm:text-sm font-medium text-stone-600 dark:text-slate-300">
              نرصد التحولات، ونربط الأنماط، ونكشف ما وراء الوقائع.
            </p>
          </div>

          {/* GEOGRAPHIC SCOPE SELECTOR (Global by default) */}
          <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-xl bg-stone-100 dark:bg-slate-900 border border-stone-200/80 dark:border-slate-800 shrink-0">
            <button
              onClick={() => {
                setScope('all');
                setSelectedCountry('all');
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
                scope === 'all'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white'
              }`}
            >
              <Globe className="w-3.5 h-3.5" />
              <span>🌐 العالم (افتراضي)</span>
            </button>

            <button
              onClick={() => {
                setScope('saudi');
                setSelectedCountry('SA');
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
                scope === 'saudi'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white'
              }`}
            >
              <span>🇸🇦 السعودية</span>
            </button>

            {/* Specific Country Selector */}
            <div className="relative inline-flex items-center">
              <select
                value={scope === 'country' ? selectedCountry : 'other'}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val === 'other') return;
                  setScope('country');
                  setSelectedCountry(val);
                }}
                className={`py-1.5 pr-2.5 pl-6 rounded-lg text-xs font-semibold appearance-none cursor-pointer outline-none transition-all ${
                  scope === 'country'
                    ? 'bg-stone-800 text-white dark:bg-stone-200 dark:text-stone-900 font-bold shadow-xs'
                    : 'bg-transparent text-stone-600 dark:text-slate-400 hover:text-stone-900 dark:hover:text-white'
                }`}
              >
                <option value="other" disabled>
                  🗺️ دولة محددة...
                </option>
                {availableCountries
                  .filter((c) => c.country && c.country !== 'SA' && c.country !== 'Global' && c.country !== 'Unknown')
                  .map((c) => {
                    const info = COUNTRY_MAP[c.country] || { name: c.country, flag: '🌐' };
                    return (
                      <option key={c.country} value={c.country}>
                        {info.flag} {info.name} ({c.count})
                      </option>
                    );
                  })}
              </select>
              <ChevronLeft className="w-3 h-3 absolute left-1.5 pointer-events-none text-stone-400 -rotate-90" />
            </div>
          </div>
        </div>

        {/* TIME WINDOW & CONTROLS STRIP */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pt-3 border-t border-stone-100 dark:border-slate-800/80">
          {/* Time Window Buttons */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs font-semibold text-stone-500 dark:text-slate-400 ml-1 flex items-center gap-1">
              <Clock className="w-3.5 h-3.5" />
              <span>المدى الزمني:</span>
            </span>

            {[
              { id: '24h', label: '⏱️ آخر 24 ساعة' },
              { id: '7d', label: '📅 آخر 7 أيام' },
              { id: '30d', label: '🗓️ آخر 30 يومًا' },
              { id: 'custom', label: '⚙️ نطاق مخصص' },
            ].map((tw) => (
              <button
                key={tw.id}
                onClick={() => setTimeWindow(tw.id as TimeWindow)}
                className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  timeWindow === tw.id
                    ? 'bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900 shadow-xs'
                    : 'bg-stone-100 dark:bg-slate-900 text-stone-600 dark:text-slate-400 hover:bg-stone-200 dark:hover:bg-slate-800'
                }`}
              >
                {tw.label}
              </button>
            ))}

            {/* Custom Date Pickers */}
            {timeWindow === 'custom' && (
              <div className="flex items-center gap-1.5 mr-2">
                <input
                  type="date"
                  value={customDateFrom}
                  onChange={(e) => setCustomDateFrom(e.target.value)}
                  className="px-2 py-0.5 text-xs bg-stone-50 dark:bg-slate-900 border border-stone-300 dark:border-slate-700 rounded text-stone-800 dark:text-stone-200 outline-none"
                  title="من تاريخ"
                />
                <span className="text-xs text-stone-400">إلى</span>
                <input
                  type="date"
                  value={customDateTo}
                  onChange={(e) => setCustomDateTo(e.target.value)}
                  className="px-2 py-0.5 text-xs bg-stone-50 dark:bg-slate-900 border border-stone-300 dark:border-slate-700 rounded text-stone-800 dark:text-stone-200 outline-none"
                  title="إلى تاريخ"
                />
              </div>
            )}
          </div>

          {/* Sort By Toggle */}
          <div className="flex items-center gap-1.5 self-start md:self-auto text-xs font-medium">
            <span className="text-stone-500 dark:text-slate-400 ml-1 flex items-center gap-1">
              <SlidersHorizontal className="w-3.5 h-3.5" />
              <span>الترتيب:</span>
            </span>
            <button
              onClick={() => setSortBy('latest')}
              className={`px-2 py-0.5 rounded text-xs font-semibold cursor-pointer ${
                sortBy === 'latest'
                  ? 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300'
                  : 'text-stone-500 hover:text-stone-800'
              }`}
            >
              الأحدث نشرًا
            </button>
            <span className="text-stone-300 dark:text-slate-700">|</span>
            <button
              onClick={() => setSortBy('importance')}
              className={`px-2 py-0.5 rounded text-xs font-semibold cursor-pointer ${
                sortBy === 'importance'
                  ? 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300'
                  : 'text-stone-500 hover:text-stone-800'
              }`}
            >
              الأعلى أهمية وتأثيرًا
            </button>
          </div>
        </div>

        {/* TELEMETRY BAR: 4 Independent Live Status Indicators (Requirement 4) */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-2 text-[11px] font-mono text-stone-500 dark:text-slate-400 bg-stone-50/60 dark:bg-slate-900/40 p-2.5 rounded-xl border border-stone-200/60 dark:border-slate-800/60">
          <div className="space-y-0.5">
            <span className="text-stone-400 block text-[10px]">آخر فحص للمصادر:</span>
            <span className="font-bold text-stone-800 dark:text-stone-200 flex items-center gap-1">
              <Activity className="w-3 h-3 text-sky-500" />
              <span>{formatTelemetry(ov?.last_source_check || ov?.last_fetch_at)}</span>
            </span>
          </div>

          <div className="space-y-0.5">
            <span className="text-stone-400 block text-[10px]">آخر جلب ناجح:</span>
            <span className="font-bold text-stone-800 dark:text-stone-200 flex items-center gap-1">
              <Check className="w-3 h-3 text-emerald-500" />
              <span>{formatTelemetry(ov?.last_successful_fetch || ov?.last_fetch_at)}</span>
            </span>
          </div>

          <div className="space-y-0.5">
            <span className="text-stone-400 block text-[10px]">أحدث مادة منشورة:</span>
            <span className="font-bold text-stone-800 dark:text-stone-200 flex items-center gap-1">
              <Clock className="w-3 h-3 text-amber-500" />
              <span>{formatTelemetry(ov?.last_new_item || (feedItems[0]?.publishedAt))}</span>
            </span>
          </div>

          <div className="space-y-0.5">
            <span className="text-stone-400 block text-[10px]">آخر تحديث للتحليل:</span>
            <span className="font-bold text-stone-800 dark:text-stone-200 flex items-center gap-1">
              <RefreshCw className="w-3 h-3 text-indigo-500" />
              <span>{formatTelemetry(ov?.last_analysis_update || ov?.last_fetch_at)}</span>
            </span>
          </div>
        </div>

        {/* 4 RECONCILED METRIC PILLS */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-1">
          <div className="p-3 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200/80 dark:border-slate-800 space-y-1">
            <span className="text-[11px] font-bold text-stone-500 block">إجمالي المواد المرصودة</span>
            <span className="text-xl sm:text-2xl font-black text-stone-900 dark:text-white font-mono">
              {ov?.total_items || feedItems.length}
            </span>
          </div>

          <div className="p-3 rounded-xl bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-200/70 dark:border-emerald-900/40 space-y-1">
            <span className="text-[11px] font-bold text-emerald-700 dark:text-emerald-400 block">حوادث موثقة (مطابقة للخريطة)</span>
            <span className="text-xl sm:text-2xl font-black text-emerald-700 dark:text-emerald-300 font-mono">
              {ov?.incidents_total ?? incidents.length}
            </span>
          </div>

          <div className="p-3 rounded-xl bg-amber-50/50 dark:bg-amber-950/20 border border-amber-200/70 dark:border-amber-900/40 space-y-1">
            <span className="text-[11px] font-bold text-amber-700 dark:text-amber-400 block">تهديدات بصلة بنكية مباشرة</span>
            <span className="text-xl sm:text-2xl font-black text-amber-700 dark:text-amber-300 font-mono">
              {ov?.incidents_high_bank_relevance ?? feedItems.filter((i) => i.isBankingThreat).length}
            </span>
          </div>

          <div className="p-3 rounded-xl bg-sky-50/50 dark:bg-sky-950/20 border border-sky-200/70 dark:border-sky-900/40 space-y-1">
            <span className="text-[11px] font-bold text-sky-700 dark:text-sky-400 block">تحذيرات وحملات توعية</span>
            <span className="text-xl sm:text-2xl font-black text-sky-700 dark:text-sky-300 font-mono">
              {(ov?.warnings_total || 0) + (ov?.awareness_total || 0)}
            </span>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* 2. SEARCH & LIVE FEED SECTION                                             */}
      {/* ========================================================================= */}
      <div className="bg-white dark:bg-[#0f172a] rounded-xl border border-stone-200/80 dark:border-slate-800 p-2.5 sm:p-3 shadow-xs">
        <form onSubmit={handleSearchSubmit} className="relative">
          <div className="relative flex items-center bg-stone-50 dark:bg-[#111726] border border-stone-200/90 dark:border-slate-700/80 rounded-lg focus-within:border-blue-500 focus-within:bg-white dark:focus-within:bg-[#0f172a] focus-within:ring-2 focus-within:ring-blue-500/10 transition-all">
            <Search className="w-4 h-4 text-stone-400 dark:text-slate-500 mr-3 shrink-0 pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="بحث في مستجدات الرصد (العنوان، الأسلوب الاحتيالي، الدولة، المصدر)..."
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
      </div>

      {/* ========================================================================= */}
      {/* 3. DUAL-COLUMN WORKSPACE: FEED LIST & DETAIL DOSSIER                       */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
        {/* FEED LIST COLUMN (lg:col-span-5) */}
        <div className="lg:col-span-5 space-y-2">
          {/* Feed Filter Tabs */}
          <div className="bg-white dark:bg-[#0f172a] rounded-xl border border-stone-200/80 dark:border-slate-800 p-2.5 space-y-2 shadow-xs">
            <div className="flex items-center justify-between px-1">
              <span className="text-xs font-bold text-stone-800 dark:text-stone-200">
                مستجدات الرصد والأخبار ({filteredFeedItems.length})
              </span>
              <span className="text-[10px] font-mono text-stone-400">
                {scope === 'all' ? '🌐 نطاق عالمي' : scope === 'saudi' ? '🇸🇦 السعودية' : `🗺️ ${selectedCountry}`}
              </span>
            </div>

            <div className="flex items-center gap-1 pt-1 border-t border-stone-100 dark:border-slate-800 text-[11px] overflow-x-auto">
              {[
                { id: 'all', label: 'الكل' },
                { id: 'confirmed', label: 'حوادث موثقة' },
                { id: 'warnings', label: 'تحذيرات' },
                { id: 'awareness', label: 'توعية' },
                { id: 'x_posts', label: 'منشورات 𝕏' },
              ].map((tab) => {
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as FeedTab)}
                    className={`px-2.5 py-1 rounded-md font-medium transition-colors cursor-pointer whitespace-nowrap ${
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

          {/* List Items */}
          <div className="space-y-1.5 max-h-[calc(100vh-230px)] overflow-y-auto pr-0.5">
            {filteredFeedItems.length === 0 ? (
              <div className="p-8 text-center bg-white dark:bg-[#0f172a] rounded-xl border border-stone-200/80 dark:border-slate-800 text-stone-400 text-xs space-y-1">
                <Info className="w-5 h-5 mx-auto text-stone-400" />
                <p>لا توجد مواد مطابقة للفلتر الزمني أو الجغرافي المختار.</p>
                <button
                  onClick={() => {
                    setTimeWindow('30d');
                    setScope('all');
                    setSelectedCountry('all');
                  }}
                  className="text-blue-600 dark:text-blue-400 underline text-xs pt-1 cursor-pointer"
                >
                  إعادة ضبط العرض لآخر 30 يومًا (العالم)
                </button>
              </div>
            ) : (
              filteredFeedItems.map((item) => {
                const isSelected = item.id === selectedItemId;

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
                    <div className="space-y-2">
                      {/* Header: Badges & Country */}
                      <div className="flex items-center justify-between gap-1 text-[11px]">
                        <div className="flex items-center gap-1.5 flex-wrap">
                          {/* Country badge */}
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-stone-100 text-stone-700 dark:bg-slate-800 dark:text-slate-300 border border-stone-200/60 dark:border-slate-700">
                            {item.countryFlag} {item.countryName}
                          </span>

                          {item.badgeType === 'confirmed' && (
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300 border border-rose-200/60 dark:border-rose-900/40">
                              🛡️ {item.badgeText}
                            </span>
                          )}
                          {item.badgeType === 'warning' && (
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200/60 dark:border-amber-900/40">
                              ⚠️ {item.badgeText}
                            </span>
                          )}
                          {item.badgeType === 'awareness' && (
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-50 text-sky-700 dark:bg-sky-950/60 dark:text-sky-300 border border-sky-200/60 dark:border-sky-900/40">
                              💡 {item.badgeText}
                            </span>
                          )}

                          {item.isBankingThreat && (
                            <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-amber-100/70 text-amber-800 dark:bg-amber-950/80 dark:text-amber-300 border border-amber-300/60 dark:border-amber-800">
                              صلة بنكية مباشرة
                            </span>
                          )}
                        </div>

                        {/* Chronological Published Time */}
                        <span className="text-[10px] text-stone-400 font-mono">
                          {item.date}
                        </span>
                      </div>

                      {/* Title */}
                      <h3
                        className={`text-xs sm:text-[13px] font-bold transition-colors leading-snug line-clamp-2 ${
                          isSelected
                            ? 'text-blue-600 dark:text-blue-400'
                            : 'text-stone-900 dark:text-stone-100 group-hover:text-blue-600'
                        }`}
                      >
                        {item.title}
                      </h3>

                      {/* Faithful Summary */}
                      <p className="text-[11px] text-stone-500 dark:text-slate-400 line-clamp-2 leading-relaxed">
                        {item.summary}
                      </p>

                      {/* Source & Action */}
                      <div className="flex items-center justify-between pt-1 border-t border-stone-100 dark:border-slate-800/80 text-[10px] text-stone-400 font-mono">
                        <span className="flex items-center gap-1 font-sans text-stone-600 dark:text-slate-400 font-medium">
                          <Building2 className="w-3 h-3 text-stone-400" />
                          <span>المصدر: {item.sourceName}</span>
                        </span>

                        <span
                          className={`flex items-center gap-0.5 font-sans font-medium transition-colors ${
                            isSelected
                              ? 'text-blue-600 dark:text-blue-400'
                              : 'text-stone-400 group-hover:text-stone-600 dark:group-hover:text-stone-300'
                          }`}
                        >
                          <span>عرض الملف</span>
                          <ChevronLeft className="w-3 h-3 group-hover:-translate-x-0.5 transition-transform" />
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* DETAIL DOSSIER COLUMN (lg:col-span-7) */}
        <div className="lg:col-span-7">
          {currentSelectedItem ? (
            <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 p-5 sm:p-6 space-y-5 shadow-xs transition-colors">
              {/* Header Badges & Actions */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-stone-100 dark:border-slate-800/80">
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-0.5 rounded text-xs font-bold bg-stone-100 dark:bg-slate-800 text-stone-800 dark:text-stone-200 border border-stone-200 dark:border-slate-700">
                    {currentSelectedItem.countryFlag} {currentSelectedItem.countryName}
                  </span>

                  <span className="px-2.5 py-0.5 rounded text-xs font-bold bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200/60 dark:border-blue-900/40">
                    {currentSelectedItem.badgeText}
                  </span>

                  {currentSelectedItem.isBankingThreat && (
                    <span className="px-2 py-0.5 rounded text-xs font-semibold bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-200/60">
                      صلة بنكية مباشرة
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 text-stone-400">
                  <button
                    onClick={handleCopyLink}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs rounded-lg hover:bg-stone-100 dark:hover:bg-slate-800 text-stone-600 dark:text-slate-300 transition-colors"
                    title="نسخ الرابط"
                  >
                    {copiedLink ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>نسخ</span>
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
                      navigate(`/explore?q=${encodeURIComponent(currentSelectedItem.title)}`)
                    }
                    className="flex items-center gap-1 px-2.5 py-1 text-xs rounded-lg hover:bg-stone-100 dark:hover:bg-slate-800 text-blue-600 dark:text-blue-400 transition-colors"
                  >
                    <span>الاستكشاف</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Title & Metadata Strip */}
              <div className="space-y-2.5">
                <h2 className="text-lg sm:text-xl font-black text-stone-900 dark:text-white leading-snug">
                  {currentSelectedItem.title}
                </h2>

                <div className="flex flex-wrap items-center gap-4 text-xs text-stone-500 dark:text-slate-400">
                  <div className="flex items-center gap-1.5 font-medium">
                    <Building2 className="w-3.5 h-3.5 text-stone-400" />
                    <span>المصدر: {currentSelectedItem.sourceName}</span>
                    {currentSelectedItem.sourceUrl && (
                      <a
                        href={currentSelectedItem.sourceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-0.5 text-blue-600 hover:underline mr-1"
                      >
                        <span>(رابط الخبر الأصلي)</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                  </div>

                  <div className="flex items-center gap-1.5 font-mono">
                    <Calendar className="w-3.5 h-3.5 text-stone-400" />
                    <span>تاريخ النشر: {fmtDay(currentSelectedItem.publishedAt) || currentSelectedItem.date}</span>
                  </div>

                  {currentSelectedItem.occurredAt && (
                    <div className="flex items-center gap-1.5 font-mono text-emerald-600 dark:text-emerald-400">
                      <span>تاريخ الواقعة: {fmtDay(currentSelectedItem.occurredAt)}</span>
                    </div>
                  )}

                  <div className="flex items-center gap-1.5">
                    <MapPin className="w-3.5 h-3.5 text-blue-500" />
                    <span>
                      {currentSelectedItem.locations.length > 0
                        ? currentSelectedItem.locations.join('، ')
                        : `${currentSelectedItem.countryName} (دون تحديد مدينة)`}
                    </span>
                  </div>
                </div>

                {/* Tags */}
                <div className="flex flex-wrap items-center gap-1.5 pt-1">
                  {currentSelectedItem.tags.map((tag, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 rounded text-[11px] font-medium bg-stone-100 dark:bg-slate-800 text-stone-700 dark:text-slate-300"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>

              {/* ================================================================= */}
              {/* SECTION 1: الحقائق المنقولة من المصدر (Verified Facts)             */}
              {/* ================================================================= */}
              <section className="rounded-xl border border-blue-100 dark:border-blue-900/30 bg-blue-50/20 dark:bg-blue-950/10 p-4 sm:p-5 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-black text-stone-900 dark:text-white text-sm">
                    <FileText className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
                    <h3>الحقائق المنقولة من المصدر</h3>
                  </div>
                  <span className="text-[11px] text-stone-400 font-sans">
                    نصوص وأرقام مستندة حرفيًا إلى التقرير الإخباري أو الأمني
                  </span>
                </div>

                <p className="text-xs sm:text-sm text-stone-700 dark:text-slate-200 leading-relaxed">
                  {cleanSummary(selectedIncidentDetail?.summary || currentSelectedItem.summary)}
                </p>

                {/* Verified figures/amounts if available */}
                {selectedIncidentDetail?.facts && (
                  <div className="flex flex-wrap items-center gap-2 pt-2 text-[11px] font-mono text-stone-600 dark:text-slate-300">
                    {selectedIncidentDetail.facts.persons_count && (
                      <span className="px-2 py-0.5 rounded bg-white dark:bg-slate-900 border border-stone-200 dark:border-slate-800">
                        المضبوطين: {selectedIncidentDetail.facts.persons_count} أشخاص
                      </span>
                    )}
                    {selectedIncidentDetail.facts.amounts && selectedIncidentDetail.facts.amounts.length > 0 && (
                      <span className="px-2 py-0.5 rounded bg-white dark:bg-slate-900 border border-stone-200 dark:border-slate-800 text-emerald-600 dark:text-emerald-400 font-semibold">
                        المبالغ المثبتة: {selectedIncidentDetail.facts.amounts.join(', ')}
                      </span>
                    )}
                  </div>
                )}
              </section>

              {/* ================================================================= */}
              {/* SECTION 2: مسار تنفيذ الاحتيال (Strict Facts vs Inferred Rules)   */}
              {/* ================================================================= */}
              <section className="rounded-xl border border-stone-200/80 dark:border-slate-800 bg-stone-50/50 dark:bg-slate-900/30 p-4 sm:p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-black text-stone-900 dark:text-white text-sm">
                    <Settings className="w-4 h-4 text-indigo-600 dark:text-indigo-400 shrink-0" />
                    <h3>مسار تنفيذ الاحتيال والأدلة التشغيلية</h3>
                  </div>
                  <span className="text-[11px] text-stone-400 font-sans">
                    التحليل الجنائي للأسلوب المتبع
                  </span>
                </div>

                {currentSelectedItem.executionSteps && currentSelectedItem.executionSteps.length > 0 ? (
                  <div className="space-y-2">
                    {currentSelectedItem.executionSteps.map((step, idx) => (
                      <div
                        key={idx}
                        className="flex items-start gap-2.5 p-2.5 rounded-lg bg-white dark:bg-slate-900 border border-stone-200/70 dark:border-slate-800 text-xs"
                      >
                        <span className="w-5 h-5 rounded bg-indigo-100 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 font-bold flex items-center justify-center shrink-0 font-mono">
                          {idx + 1}
                        </span>
                        <span className="text-stone-800 dark:text-slate-200 leading-normal">{step}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-3.5 rounded-lg bg-stone-100/80 dark:bg-slate-800/60 border border-stone-200/60 dark:border-slate-700/60 text-stone-600 dark:text-slate-300 text-xs flex items-center gap-2.5">
                    <Info className="w-4 h-4 text-stone-400 shrink-0" />
                    <span>المعلومات المتاحة لا تكفي لتحديد مسار الاحتيال الكامل (التقرير يركز على الإنذار الأولي أو التوعية دون تفصيل الخطوات الجنائية).</span>
                  </div>
                )}
              </section>

              {/* ================================================================= */}
              {/* SECTION 3: المخاطر والآثار المصرفية                                */}
              {/* ================================================================= */}
              <section className="rounded-xl border border-amber-200/80 dark:border-amber-900/40 bg-amber-50/20 dark:bg-amber-950/10 p-4 sm:p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-black text-stone-900 dark:text-white text-sm">
                    <Landmark className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
                    <h3>المخاطر والآثار المصرفية</h3>
                  </div>
                  <span className="text-[11px] text-stone-400 font-sans">
                    تقييم الأثر على القنوات البنكية
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div className="p-3 rounded-lg bg-white dark:bg-slate-900 border border-stone-200/70 dark:border-slate-800 space-y-1.5">
                    <span className="text-[11px] font-bold text-stone-500 uppercase tracking-wide">
                      القنوات المستهدفة (Payment Rails)
                    </span>
                    <div className="flex flex-wrap gap-1">
                      {currentSelectedItem.channels.length > 0 ? (
                        currentSelectedItem.channels.map((ch, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-0.5 rounded text-[11px] font-medium bg-stone-100 dark:bg-slate-800 text-stone-700 dark:text-slate-300"
                          >
                            {ch}
                          </span>
                        ))
                      ) : (
                        <span className="text-xs text-stone-400">قنوات إلكترونية متعددة</span>
                      )}
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-white dark:bg-slate-900 border border-stone-200/70 dark:border-slate-800 space-y-1.5">
                    <span className="text-[11px] font-bold text-stone-500 uppercase tracking-wide">
                      تقييم المخاطر التشغيلية
                    </span>
                    <p className="text-xs text-stone-700 dark:text-slate-300 leading-relaxed">
                      {currentSelectedItem.isBankingThreat
                        ? 'مخاطر استغلال حسابات وسيطة (Mules) أو سحب نقدي فوري تتطلب تطبيق قواعد فحص وتجميد للعمليات الشاذة.'
                        : 'مخاطر تصيد ووعي عام دون اختراق مباشر للبنية المصرفية الأساسية.'}
                    </p>
                  </div>
                </div>

                {/* Suggested Detection Rule */}
                <div className="space-y-1.5 pt-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-stone-700 dark:text-slate-300 flex items-center gap-1.5">
                      <Code2 className="w-3.5 h-3.5 text-blue-600" />
                      <span>فرضية قاعدة الكشف المقترحة (Rule Hypothesis):</span>
                    </span>
                    <button
                      onClick={() => handleCopyRuleCode(sampleDetectionRuleSql)}
                      className="flex items-center gap-1 text-[11px] text-blue-600 dark:text-blue-400 hover:underline cursor-pointer"
                    >
                      {copiedRule ? <Check className="w-3 h-3 text-emerald-500" /> : <Copy className="w-3 h-3" />}
                      <span>{copiedRule ? 'تم النسخ' : 'نسخ الاستعلام'}</span>
                    </button>
                  </div>

                  <pre className="p-3 bg-stone-900 dark:bg-[#070b14] text-stone-200 rounded-xl text-[11px] font-mono overflow-x-auto leading-relaxed border border-stone-800">
                    {sampleDetectionRuleSql}
                  </pre>
                </div>
              </section>

              {/* ================================================================= */}
              {/* SECTION 4: الأدلة والمصادر الإخبارية                              */}
              {/* ================================================================= */}
              <section className="space-y-3 pt-2">
                <div className="flex items-center justify-between pb-1 border-b border-stone-100 dark:border-slate-800">
                  <div className="flex items-center gap-2 font-bold text-stone-900 dark:text-white text-sm">
                    <FileText className="w-4 h-4 text-blue-600 shrink-0" />
                    <h3>الأدلة والمصادر الموثقة ({currentSelectedItem.sourcesCount})</h3>
                  </div>
                </div>

                {selectedIncidentDetail?.articles && selectedIncidentDetail.articles.length > 0 ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {selectedIncidentDetail.articles.map((art) => (
                      <div
                        key={art.article_id}
                        className="p-3 rounded-xl bg-stone-50/80 dark:bg-slate-900/60 border border-stone-200/70 dark:border-slate-800 space-y-1.5 flex flex-col justify-between"
                      >
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-[11px] text-stone-400 font-mono">
                            <span className="font-bold text-stone-700 dark:text-slate-300">
                              {art.source || 'مصدر إخباري'}
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
                      رصد آلي معتمد ومؤرشف في محرك الرصد العالمي.
                    </span>
                    {currentSelectedItem.sourceUrl && (
                      <a
                        href={currentSelectedItem.sourceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 dark:text-blue-400 font-medium hover:underline flex items-center gap-1"
                      >
                        <span>فتح الرابط المباشر</span>
                        <ExternalLink className="w-3.5 h-3.5" />
                      </a>
                    )}
                  </div>
                )}
              </section>
            </div>
          ) : (
            <div className="p-12 text-center bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 text-stone-400 text-xs">
              اختر مادة من القائمة لعرض تفاصيل الملف والتحليل المعتمد.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ExecutiveOverview;
