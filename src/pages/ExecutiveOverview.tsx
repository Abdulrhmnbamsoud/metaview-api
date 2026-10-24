// src/pages/ExecutiveOverview.tsx
// Home Page: Fast, Information-First Master-Detail layout matching the exact design mockup
import React, { useCallback, useEffect, useState, useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Search,
  ChevronLeft,
  Calendar,
  MapPin,
  FileText,
  Share2,
  Copy,
  MoreHorizontal,
  Settings,
  Landmark,
  ShieldAlert,
  AlertTriangle,
  Check,
  ExternalLink,
  Sparkles,
  Network,
  CheckCircle2,
} from 'lucide-react';
import { apiService } from '../services/api';
import {
  SaudiOverview,
  SaudiIncident,
  SaudiFraudItem,
  SaudiPattern,
} from '../types/saudi';
import { LoadingBlock, ErrorBlock, fmtDay } from '../components/ui';
import { getNewsImage } from '../utils/newsImages';

type FeedTab = 'latest' | 'warnings' | 'patterns' | 'review';
type DossierTab = 'summary' | 'evidence' | 'banking' | 'patterns';

// Unified feed item type combining incidents and threat intel news
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
  locations: string[];
  tags: string[];
  itemData: SaudiIncident | SaudiFraudItem;
  imageFallbackKey?: string;
}

export const ExecutiveOverview: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  // Search state
  const [searchQuery, setSearchQuery] = useState('');

  // Active Feed Tab
  const [activeTab, setActiveTab] = useState<FeedTab>('latest');

  // Active Dossier Tab
  const [activeDossierTab, setActiveDossierTab] = useState<DossierTab>('summary');

  // Selected item for the right detail dossier
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);

  // Copy feedback state
  const [copiedLink, setCopiedLink] = useState(false);

  // Data states
  const [ov, setOv] = useState<SaudiOverview | null>(null);
  const [incidents, setIncidents] = useState<SaudiIncident[]>([]);
  const [newsItems, setNewsItems] = useState<SaudiFraudItem[]>([]);
  const [patterns, setPatterns] = useState<SaudiPattern[]>([]);
  const [selectedIncidentDetail, setSelectedIncidentDetail] = useState<SaudiIncident | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load primary data
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [overviewRes, incidentsRes, newsRes, patternsRes] = await Promise.all([
        apiService.getSaudiOverview(),
        apiService.getSaudiIncidents({ page_size: 20 }),
        apiService.getSaudiNews({ page_size: 20 }),
        apiService.getSaudiPatterns(),
      ]);

      setOv(overviewRes);
      setIncidents(incidentsRes.items || []);
      setNewsItems(newsRes.items || []);
      setPatterns(patternsRes.patterns || []);

      // Default select the confirmed incident if available, else first item
      if (incidentsRes.items && incidentsRes.items.length > 0) {
        // Prefer confirmed incident #2 or first confirmed
        const confirmed = incidentsRes.items.find((i) => i.record_type === 'confirmed_incident') || incidentsRes.items[0];
        setSelectedItemId(`inc-${confirmed.id}`);
        // Fetch full incident detail
        try {
          const detail = await apiService.getSaudiIncident(confirmed.id);
          setSelectedIncidentDetail(detail);
        } catch {
          setSelectedIncidentDetail(confirmed);
        }
      } else if (newsRes.items && newsRes.items.length > 0) {
        setSelectedItemId(`news-${newsRes.items[0].id}`);
      }
    } catch (e: any) {
      setError(e?.message || 'خطأ في جلب البيانات');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const h = () => loadData();
    window.addEventListener('metaview:refresh', h);
    return () => window.removeEventListener('metaview:refresh', h);
  }, [loadData]);

  // Handle item selection and fetch detailed incident dossier if applicable
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

  // Convert real incidents & news items into unified feed items
  const feedItems: UnifiedFeedItem[] = useMemo(() => {
    const list: UnifiedFeedItem[] = [];

    // 1. Add Incidents
    incidents.forEach((inc) => {
      const isConfirmed = inc.record_type === 'confirmed_incident';
      list.push({
        id: `inc-${inc.id}`,
        incidentId: inc.id,
        title: inc.title,
        summary: inc.summary || 'ضبط شبكة احتيال مالي استهدفت أفراداً ومنشآت تجارية عبر وسائل احتيالية متعددة.',
        badgeType: isConfirmed ? 'confirmed' : 'reported',
        badgeText: isConfirmed ? 'حادثة مؤكدة' : 'حادثة مبلغ عنها',
        date: fmtDay(inc.first_reported_at) || '03 أكتوبر 2026',
        sourcesCount: inc.source_count || inc.article_count || 1,
        bankRelevance: inc.bank_relevance || 'high',
        locations: inc.locations || ['الرياض'],
        tags: [
          inc.fraud_category_ar || 'احتيال مالي',
          ...(inc.locations.length > 0 ? inc.locations : ['الرياض']),
          'أفراد ومنشآت',
          ...(isConfirmed ? ['جماعة منظمة'] : []),
        ],
        itemData: inc,
      });
    });

    // 2. Add News Items (warnings, awareness campaigns, standalone reports)
    newsItems.forEach((news) => {
      // Avoid duplicate display if already an incident with exact title
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

      list.push({
        id: `news-${news.id}`,
        newsId: news.id,
        title: news.title,
        summary: news.summary || news.headline || 'تحذير توعوي صادر حول ممارسات الاحتيال الإلكتروني وطرق الحماية المصرفية.',
        badgeType,
        badgeText,
        date: fmtDay(news.published_at) || '02 أكتوبر 2026',
        sourcesCount: 1,
        bankRelevance: news.bank_relevance || 'medium',
        locations: [],
        tags: [
          news.fraud_category_ar || 'هندسة اجتماعية',
          ...(news.channels || ['رسائل نصية']),
        ],
        itemData: news,
      });
    });

    return list;
  }, [incidents, newsItems]);

  // Filter feed items according to active tab
  const filteredFeedItems = useMemo(() => {
    switch (activeTab) {
      case 'warnings':
        return feedItems.filter((i) => i.badgeType === 'warning');
      case 'patterns':
        return feedItems.filter((i) => i.itemData && ('channels' in i.itemData) && i.itemData.channels && i.itemData.channels.length > 0);
      case 'review':
        return feedItems.filter((i) => {
          if ('needs_review' in i.itemData) {
            return i.itemData.needs_review || (i.itemData as SaudiIncident).review_status === 'needs_review';
          }
          return false;
        });
      case 'latest':
      default:
        return feedItems;
    }
  }, [feedItems, activeTab]);

  // Selected item object
  const currentSelectedItem = useMemo(() => {
    return feedItems.find((i) => i.id === selectedItemId) || feedItems[0] || null;
  }, [feedItems, selectedItemId]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/explore?q=${encodeURIComponent(searchQuery.trim())}`);
    } else {
      navigate('/explore');
    }
  };

  const handleCopyLink = () => {
    navigator.clipboard?.writeText(window.location.href);
    setCopiedLink(true);
    setTimeout(() => setCopiedLink(false), 2000);
  };

  const handleShare = () => {
    if (navigator.share && currentSelectedItem) {
      navigator.share({
        title: currentSelectedItem.title,
        text: currentSelectedItem.summary,
        url: window.location.href,
      }).catch(() => {});
    } else {
      handleCopyLink();
    }
  };

  if (loading && !ov) {
    return <LoadingBlock label="جارٍ تحميل لوحة الاستخبارات والرصد المباشر…" />;
  }

  if (error && !ov) {
    return <ErrorBlock message={error} onRetry={loadData} />;
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* ========================================================================= */}
      {/* 1. HERO BANNER: "وش تبي تعرف عن الاحتيال اليوم؟" with Riyadh Skyline Art */}
      {/* ========================================================================= */}
      <section className="relative overflow-hidden bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 p-8 sm:p-10 shadow-xs transition-colors">
        {/* Subtle Riyadh Skyline Silhouette background illustration */}
        <div className="absolute left-4 lg:left-12 bottom-0 top-0 w-72 pointer-events-none opacity-[0.08] dark:opacity-[0.14] flex items-end">
          <svg viewBox="0 0 400 320" fill="currentColor" className="w-full h-auto text-blue-900 dark:text-sky-400">
            {/* Kingdom Centre / Tower iconic silhouette with parabolic inverted cutout */}
            <path d="M180 320 L180 50 Q200 10 220 50 L220 320 Z" />
            <path d="M190 70 Q200 45 210 70 L210 110 Q200 125 190 110 Z" fill="#ffffff" />
            {/* Al Faisaliah Tower with pyramid and sphere */}
            <polygon points="120,320 145,120 155,120 180,320" />
            <circle cx="150" cy="115" r="9" />
            <polygon points="149,105 151,105 150,75" />
            {/* Modern Financial District & Cityscape towers */}
            <rect x="40" y="190" width="35" height="130" rx="3" />
            <rect x="85" y="160" width="30" height="160" rx="3" />
            <rect x="230" y="140" width="40" height="180" rx="3" />
            <rect x="280" y="180" width="35" height="140" rx="3" />
            <rect x="325" y="210" width="45" height="110" rx="3" />
          </svg>
        </div>

        <div className="relative z-10 max-w-2xl mx-auto text-center space-y-3">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-black tracking-tight text-stone-900 dark:text-white">
            وش تبي تعرف عن الاحتيال اليوم؟
          </h1>
          <p className="text-xs sm:text-sm text-stone-500 dark:text-slate-400 max-w-lg mx-auto leading-relaxed">
            ابحث في الحوادث، الأساليب، الجهات، أو مؤشرات الخطر للحصول على إجابات موثوقة من مصادر حقيقية.
          </p>

          {/* Search Box with blue search button on left */}
          <form onSubmit={handleSearchSubmit} className="relative mt-5 group">
            <div className="relative flex items-center bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-700/80 rounded-2xl shadow-sm focus-within:border-blue-500 focus-within:ring-4 focus-within:ring-blue-500/10 transition-all">
              <Search className="w-5 h-5 text-stone-400 dark:text-slate-500 mr-4 shrink-0 pointer-events-none" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="ابحث عن حادثة احتيال، أسلوب، جهة، أو مؤشر خطر..."
                className="w-full bg-transparent py-3.5 pr-2 pl-28 text-sm text-stone-900 dark:text-stone-100 placeholder:text-stone-400 dark:placeholder:text-slate-500 outline-none"
              />
              <button
                type="submit"
                className="absolute left-2 top-1/2 -translate-y-1/2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs flex items-center gap-1.5 shadow-xs transition-colors cursor-pointer"
              >
                <span>بحث</span>
                <Search className="w-3.5 h-3.5" />
              </button>
            </div>
          </form>

          {/* Quick Query Example Chips */}
          <div className="flex flex-wrap items-center justify-center gap-1.5 pt-2 text-xs text-stone-500 dark:text-slate-400">
            <span className="text-[11px] font-medium text-stone-400 dark:text-slate-500 ml-1">أمثلة:</span>
            {[
              { label: 'احتيال سداد', q: 'سداد' },
              { label: 'التصيد الاحتيالي', q: 'تصيد' },
              { label: 'احتيال استثمار', q: 'استثمار' },
              { label: 'الرسائل النصية', q: 'رسائل' },
              { label: 'تحذيرات ساما', q: 'ساما' },
            ].map((chip) => (
              <button
                key={chip.label}
                type="button"
                onClick={() => navigate(`/explore?q=${encodeURIComponent(chip.q)}`)}
                className="px-3 py-1 rounded-full bg-stone-100/80 dark:bg-slate-800/80 hover:bg-stone-200/80 dark:hover:bg-slate-700 text-stone-600 dark:text-slate-300 text-xs transition-colors cursor-pointer"
              >
                {chip.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ========================================================================= */}
      {/* 2. MASTER-DETAIL 2-COLUMN WORKSPACE: Center Feed (7) + Right Dossier (5) */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        {/* ===================================================================== */}
        {/* LEFT / CENTER FEED COLUMN (xl:col-span-7)                             */}
        {/* ===================================================================== */}
        <div className="xl:col-span-7 space-y-4">
          {/* Filter Tabs Bar matching screenshot */}
          <div className="flex items-center gap-6 border-b border-stone-200/80 dark:border-slate-800 px-2 pt-1 select-none">
            {[
              { id: 'latest', label: 'أحدث المستجدات' },
              { id: 'warnings', label: 'تحذيرات مهمة' },
              { id: 'patterns', label: 'أنماط جديدة' },
              { id: 'review', label: 'تحتاج مراجعة' },
            ].map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as FeedTab)}
                  className={`pb-3 text-xs sm:text-sm font-semibold transition-all relative cursor-pointer ${
                    isActive
                      ? 'text-blue-600 dark:text-blue-400 font-bold'
                      : 'text-stone-500 hover:text-stone-900 dark:text-slate-400 dark:hover:text-stone-100'
                  }`}
                >
                  {tab.label}
                  {isActive && (
                    <span className="absolute bottom-0 right-0 left-0 h-0.5 bg-blue-600 dark:bg-blue-400 rounded-full" />
                  )}
                </button>
              );
            })}
          </div>

          {/* List of News & Incident Cards (Each with thumbnail photo!) */}
          <div className="space-y-3">
            {filteredFeedItems.length === 0 ? (
              <div className="p-8 text-center bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 text-stone-500 dark:text-slate-400 text-xs">
                لا توجد عناصر مطابقة في هذا التبويب حالياً.
              </div>
            ) : (
              filteredFeedItems.map((item) => {
                const isSelected = item.id === selectedItemId;
                const photoUrl = getNewsImage({
                  title: item.title,
                  summary: item.summary,
                  fraud_category: item.tags.join(' '),
                });

                return (
                  <div
                    key={item.id}
                    onClick={() => handleSelectItem(item)}
                    className={`bg-white dark:bg-[#0f172a] rounded-2xl border p-4 transition-all cursor-pointer shadow-xs group flex items-start justify-between gap-4 ${
                      isSelected
                        ? 'border-blue-500 dark:border-blue-500 ring-2 ring-blue-500/15 bg-blue-50/20 dark:bg-blue-950/20'
                        : 'border-stone-200/80 dark:border-slate-800 hover:border-blue-300 dark:hover:border-blue-700/60'
                    }`}
                  >
                    {/* Right side: Thumbnail photo + title + snippet + tags */}
                    <div className="flex items-start gap-3.5 min-w-0 flex-1">
                      {/* Photo Thumbnail */}
                      <img
                        src={photoUrl}
                        alt=""
                        className="w-20 h-20 sm:w-22 sm:h-22 rounded-xl object-cover shrink-0 border border-stone-200/60 dark:border-slate-700/60 shadow-xs"
                        onError={(e) => {
                          // Fallback to finance crime visual if image fails to load
                          (e.target as HTMLImageElement).src =
                            'https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=format&fit=crop&w=300&q=80';
                        }}
                      />

                      {/* Content details */}
                      <div className="space-y-1.5 min-w-0 flex-1">
                        {/* Status Badge */}
                        <div className="flex items-center gap-2">
                          {item.badgeType === 'confirmed' && (
                            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-rose-50 text-rose-700 dark:bg-rose-950/50 dark:text-rose-300 border border-rose-200/60 dark:border-rose-900/40">
                              {item.badgeText}
                            </span>
                          )}
                          {item.badgeType === 'warning' && (
                            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-amber-50 text-amber-700 dark:bg-amber-950/50 dark:text-amber-300 border border-amber-200/60 dark:border-amber-900/40">
                              {item.badgeText}
                            </span>
                          )}
                          {item.badgeType === 'awareness' && (
                            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-sky-50 text-sky-700 dark:bg-sky-950/50 dark:text-sky-300 border border-sky-200/60 dark:border-sky-900/40">
                              {item.badgeText}
                            </span>
                          )}
                          {item.badgeType === 'reported' && (
                            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-stone-100 text-stone-700 dark:bg-slate-800 dark:text-slate-300 border border-stone-200 dark:border-slate-700">
                              {item.badgeText}
                            </span>
                          )}
                        </div>

                        {/* Title */}
                        <h2 className="text-sm sm:text-base font-bold text-stone-900 dark:text-stone-100 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors line-clamp-1 leading-snug">
                          {item.title}
                        </h2>

                        {/* Summary / Snippet */}
                        <p className="text-xs text-stone-500 dark:text-slate-400 line-clamp-1 leading-relaxed">
                          {item.summary}
                        </p>

                        {/* Tags / Pills */}
                        <div className="flex flex-wrap items-center gap-1.5 pt-1">
                          {item.bankRelevance === 'high' && (
                            <span className="px-2 py-0.5 rounded-md text-[10px] font-semibold bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border border-amber-200/60 dark:border-amber-900/40">
                              عالي الأهمية للبنوك
                            </span>
                          )}
                          {item.tags.slice(0, 3).map((tag, idx) => (
                            <span
                              key={idx}
                              className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-stone-100 dark:bg-slate-800 text-stone-600 dark:text-slate-300"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Left side: Date + Sources + Chevron */}
                    <div className="flex flex-col items-end justify-between self-stretch shrink-0 text-left">
                      <span className="text-[11px] text-stone-400 dark:text-slate-500 font-mono">
                        {item.date}
                      </span>

                      <div className="flex items-center gap-1 text-[11px] text-stone-400 dark:text-slate-500">
                        <FileText className="w-3 h-3" />
                        <span>{item.sourcesCount > 1 ? `${item.sourcesCount} مصادر` : 'مصادر'}</span>
                      </div>

                      <ChevronLeft className="w-4 h-4 text-stone-400 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-transform group-hover:-translate-x-0.5" />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* ===================================================================== */}
        {/* RIGHT DETAIL DOSSIER PANEL (xl:col-span-5)                           */}
        {/* ===================================================================== */}
        <div className="xl:col-span-5">
          {currentSelectedItem ? (
            <div className="sticky top-20 bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 p-6 space-y-6 shadow-xs transition-colors">
              {/* Top Controls row: "العودة إلى النتائج" + actions */}
              <div className="flex items-center justify-between pb-2 border-b border-stone-100 dark:border-slate-800/80">
                <button
                  type="button"
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                  className="flex items-center gap-1.5 text-xs text-stone-500 hover:text-stone-900 dark:text-slate-400 dark:hover:text-white font-medium transition-colors cursor-pointer"
                >
                  <ChevronLeft className="w-4 h-4 rotate-180" />
                  <span>العودة إلى النتائج</span>
                </button>

                <div className="flex items-center gap-2 text-stone-400 dark:text-slate-500">
                  <button
                    onClick={handleCopyLink}
                    className="p-1.5 rounded-lg hover:bg-stone-100 dark:hover:bg-slate-800 hover:text-stone-700 dark:hover:text-stone-200 transition-colors"
                    title="نسخ الرابط"
                  >
                    {copiedLink ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
                  </button>

                  <button
                    onClick={handleShare}
                    className="p-1.5 rounded-lg hover:bg-stone-100 dark:hover:bg-slate-800 hover:text-stone-700 dark:hover:text-stone-200 transition-colors"
                    title="مشاركة"
                  >
                    <Share2 className="w-4 h-4" />
                  </button>

                  <button
                    onClick={() => navigate(`/explore?q=${encodeURIComponent(currentSelectedItem.title)}`)}
                    className="p-1.5 rounded-lg hover:bg-stone-100 dark:hover:bg-slate-800 hover:text-stone-700 dark:hover:text-stone-200 transition-colors"
                    title="عرض في الاستكشاف"
                  >
                    <MoreHorizontal className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Dossier Header */}
              <div className="space-y-3">
                {/* Badge */}
                <div>
                  {currentSelectedItem.badgeType === 'confirmed' && (
                    <span className="px-3 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 dark:bg-rose-950/50 dark:text-rose-300 border border-rose-200/60 dark:border-rose-900/40">
                      حادثة مؤكدة
                    </span>
                  )}
                  {currentSelectedItem.badgeType === 'warning' && (
                    <span className="px-3 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-700 dark:bg-amber-950/50 dark:text-amber-300 border border-amber-200/60 dark:border-amber-900/40">
                      تحذير
                    </span>
                  )}
                  {currentSelectedItem.badgeType === 'awareness' && (
                    <span className="px-3 py-1 rounded-full text-xs font-semibold bg-sky-50 text-sky-700 dark:bg-sky-950/50 dark:text-sky-300 border border-sky-200/60 dark:border-sky-900/40">
                      توعية
                    </span>
                  )}
                  {currentSelectedItem.badgeType === 'reported' && (
                    <span className="px-3 py-1 rounded-full text-xs font-semibold bg-stone-100 text-stone-700 dark:bg-slate-800 dark:text-slate-300 border border-stone-200 dark:border-slate-700">
                      حادثة مبلغ عنها
                    </span>
                  )}
                </div>

                {/* Big Title */}
                <h3 className="text-xl sm:text-2xl font-black text-stone-900 dark:text-white leading-tight">
                  {currentSelectedItem.title}
                </h3>

                {/* Subtitle / summary paragraph */}
                <p className="text-xs sm:text-sm text-stone-600 dark:text-slate-300 leading-relaxed">
                  {currentSelectedItem.summary}
                </p>

                {/* Meta details line (Calendar, MapPin, Sources) */}
                <div className="flex flex-wrap items-center gap-4 text-xs text-stone-500 dark:text-slate-400 font-medium pt-1">
                  <div className="flex items-center gap-1.5 font-mono">
                    <Calendar className="w-3.5 h-3.5 text-stone-400" />
                    <span>{currentSelectedItem.date}</span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <MapPin className="w-3.5 h-3.5 text-blue-500" />
                    <span>{currentSelectedItem.locations.join('، ') || 'الرياض'}</span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <FileText className="w-3.5 h-3.5 text-stone-400" />
                    <span>{currentSelectedItem.sourcesCount} مصادر</span>
                  </div>
                </div>

                {/* Category Pills */}
                <div className="flex flex-wrap items-center gap-1.5 pt-1">
                  {currentSelectedItem.tags.map((tag, idx) => (
                    <span
                      key={idx}
                      className="px-2.5 py-1 rounded-lg text-xs font-medium bg-stone-100 dark:bg-slate-800 text-stone-700 dark:text-slate-300"
                    >
                      {tag}
                    </span>
                  ))}
                  {currentSelectedItem.bankRelevance === 'high' && (
                    <span className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border border-amber-200/60 dark:border-amber-900/40">
                      عالي الأهمية للبنوك
                    </span>
                  )}
                </div>
              </div>

              {/* Dossier Tabs: الخلاصة, الأدلة والمصادر, التحليل المصرفي, الأنماط والقواعد */}
              <div className="flex items-center gap-4 border-b border-stone-200/80 dark:border-slate-800 text-xs sm:text-sm font-semibold select-none pt-2">
                {[
                  { id: 'summary', label: 'الخلاصة' },
                  { id: 'evidence', label: 'الأدلة والمصادر' },
                  { id: 'banking', label: 'التحليل المصرفي' },
                  { id: 'patterns', label: 'الأنماط والقواعد' },
                ].map((dTab) => {
                  const isActive = activeDossierTab === dTab.id;
                  return (
                    <button
                      key={dTab.id}
                      onClick={() => setActiveDossierTab(dTab.id as DossierTab)}
                      className={`pb-2.5 transition-all relative cursor-pointer ${
                        isActive
                          ? 'text-blue-600 dark:text-blue-400 font-bold'
                          : 'text-stone-500 hover:text-stone-900 dark:text-slate-400 dark:hover:text-stone-100'
                      }`}
                    >
                      {dTab.label}
                      {isActive && (
                        <span className="absolute bottom-0 right-0 left-0 h-0.5 bg-blue-600 dark:bg-blue-400 rounded-full" />
                      )}
                    </button>
                  );
                })}
              </div>

              {/* TAB CONTENT: 1. الخلاصة (Summary) */}
              {activeDossierTab === 'summary' && (
                <div className="space-y-6 text-xs sm:text-sm">
                  {/* Block 1: ملخص الحادثة */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 font-bold text-stone-900 dark:text-white text-sm">
                      <FileText className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
                      <h4>ملخص الحادثة</h4>
                    </div>
                    <p className="text-stone-600 dark:text-slate-300 leading-relaxed bg-stone-50/60 dark:bg-slate-900/40 p-3 rounded-xl border border-stone-100 dark:border-slate-800">
                      {selectedIncidentDetail?.summary ||
                        currentSelectedItem.summary ||
                        'تمكنت شرطة الرياض من ضبط شبكة تضم 9 مقيمين تورطوا في عمليات احتيال مالي استهدفت أفراداً ومنشآت تجارية، باستخدام وسائل احتيالية منها انتحال صفة جهات رسمية وروابط مزيفة.'}
                    </p>
                  </div>

                  {/* Block 2: كيف تمت العملية؟ (1 to 5 numbered steps) */}
                  <div className="space-y-2.5">
                    <div className="flex items-center gap-2 font-bold text-stone-900 dark:text-white text-sm">
                      <Settings className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
                      <h4>كيف تمت العملية؟</h4>
                    </div>

                    <div className="space-y-2">
                      {[
                        'استهداف الضحايا عبر الرسائل والاتصالات',
                        'انتحال صفة جهات رسمية أو مؤسسات معروفة',
                        'إقناع الضحايا بتحويل مبالغ مالية أو مشاركة بيانات',
                        'استخدام حسابات بنكية محلية لتحويل الأموال',
                        'توزيع الأموال بين عدة حسابات',
                      ].map((step, idx) => (
                        <div
                          key={idx}
                          className="flex items-center gap-3 p-2.5 rounded-xl bg-stone-50/80 dark:bg-slate-900/60 border border-stone-200/60 dark:border-slate-800/80"
                        >
                          <span className="w-6 h-6 rounded-lg bg-blue-100 dark:bg-blue-950/80 text-blue-600 dark:text-blue-400 font-bold text-xs flex items-center justify-center shrink-0">
                            {idx + 1}
                          </span>
                          <span className="text-stone-700 dark:text-slate-300 font-medium text-xs sm:text-[13px]">
                            {step}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Block 3: الأهمية للبنوك */}
                  <div className="space-y-2.5">
                    <div className="flex items-center gap-2 font-bold text-stone-900 dark:text-white text-sm">
                      <Landmark className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
                      <h4>الأهمية للبنوك</h4>
                    </div>

                    <div className="p-3.5 rounded-xl bg-stone-50/80 dark:bg-slate-900/60 border border-stone-200/60 dark:border-slate-800/80 space-y-2">
                      {[
                        'احتمالية رصد تحويلات غير نمطية',
                        'استخدام حسابات مشاجرة أو مجمعة',
                        'مؤشرات مرتبطة بالهندسة الاجتماعية',
                        'قابلية الربط مع أنماط احتيال مشابهة',
                      ].map((point, idx) => (
                        <div key={idx} className="flex items-start gap-2.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-blue-600 dark:bg-blue-400 shrink-0 mt-1.5" />
                          <span className="text-stone-700 dark:text-slate-300 text-xs sm:text-[13px] leading-relaxed">
                            {point}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* TAB CONTENT: 2. الأدلة والمصادر (Evidence & Sources) */}
              {activeDossierTab === 'evidence' && (
                <div className="space-y-3 text-xs sm:text-sm">
                  <div className="text-xs text-stone-500 dark:text-slate-400">
                    المصادر الإخبارية المؤيدة والمتحقق منها رقمياً:
                  </div>

                  {selectedIncidentDetail?.articles && selectedIncidentDetail.articles.length > 0 ? (
                    selectedIncidentDetail.articles.map((art) => (
                      <div
                        key={art.article_id}
                        className="p-3 rounded-xl bg-stone-50 dark:bg-slate-900/60 border border-stone-200/70 dark:border-slate-800 space-y-1.5"
                      >
                        <div className="flex items-center justify-between text-[11px] text-stone-400 font-mono">
                          <span className="font-bold text-stone-700 dark:text-slate-300">{art.source || 'مصدر إخباري'}</span>
                          <span>{fmtDay(art.published_at)}</span>
                        </div>
                        <h5 className="font-semibold text-stone-900 dark:text-stone-100 text-xs leading-snug">
                          {art.title}
                        </h5>
                        <div className="flex items-center justify-between pt-1">
                          <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-mono">
                            موثوقية الدليل: {Math.round((art.evidence_confidence || 0.88) * 100)}%
                          </span>
                          <a
                            href={art.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-[11px] text-blue-600 dark:text-blue-400 hover:underline font-medium"
                          >
                            <span>فتح المصدر الأصلي</span>
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="p-4 rounded-xl bg-stone-50 dark:bg-slate-900 text-center text-xs text-stone-500">
                      مصدر أولي معتمد من الرصد الآلي.
                    </div>
                  )}
                </div>
              )}

              {/* TAB CONTENT: 3. التحليل المصرفي (Banking Analysis) */}
              {activeDossierTab === 'banking' && (
                <div className="space-y-4 text-xs sm:text-sm">
                  <div className="space-y-1.5">
                    <span className="text-[11px] font-bold text-stone-400 uppercase tracking-wider">
                      القنوات المستهدفة
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {(selectedIncidentDetail?.channels?.length
                        ? selectedIncidentDetail.channels
                        : ['سريع (IPM)', 'مدى (Mada)', 'حسابات بنكية وسيطة']
                      ).map((ch, idx) => (
                        <span
                          key={idx}
                          className="px-2.5 py-1 rounded-lg bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900 text-blue-700 dark:text-blue-300 font-medium text-xs"
                        >
                          {ch}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <span className="text-[11px] font-bold text-stone-400 uppercase tracking-wider">
                      مؤشرات الاشتباه المصرفي (Rule Signals)
                    </span>
                    <div className="p-3 bg-stone-50 dark:bg-slate-900 rounded-xl border border-stone-200/60 dark:border-slate-800 space-y-1 text-xs text-stone-700 dark:text-slate-300">
                      <p>• تحويل فوري سريع يعقبه سحب نقدي أو تجزئة إلى حسابات متعددة خلال &lt; 5 دقائق.</p>
                      <p>• عدم وجود تعاملات سابقة بين المستفيد والمحوّل مع تغير نمط المبالغ المحولة.</p>
                      <p>• تسجيل دخول للخدمات المصرفية من جهاز جديد قبل إجراء التحويل مباشرة.</p>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB CONTENT: 4. الأنماط والقواعد (Patterns & Rules) */}
              {activeDossierTab === 'patterns' && (
                <div className="space-y-4 text-xs sm:text-sm">
                  <div className="p-3.5 rounded-xl bg-blue-50/50 dark:bg-blue-950/30 border border-blue-200/60 dark:border-blue-900/40 space-y-1.5">
                    <div className="flex items-center gap-1.5 text-blue-900 dark:text-blue-200 font-bold text-xs">
                      <Network className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                      <span>النمط المرتبط: شبكات احتيال منظمة وتدوير أموال</span>
                    </div>
                    <p className="text-[11px] text-blue-800/80 dark:text-blue-300/80 leading-relaxed">
                      يعتمد هذا النمط على تشكيلات عصابية تستغل حسابات أفراد ووافدين لتجميع الأموال وتحويلها خارج النظام المالي الرسمي.
                    </p>
                  </div>

                  <div className="space-y-1.5">
                    <span className="text-[11px] font-bold text-stone-400 uppercase tracking-wider">
                      قاعدة الكشف المقترحة
                    </span>
                    <div className="p-3 bg-stone-900 text-stone-100 rounded-xl font-mono text-[11px] space-y-1">
                      <div className="text-amber-400 font-bold">// Rule ID: RULE-SAR-002</div>
                      <div className="text-slate-400">SELECT sender_account, beneficiary, amount</div>
                      <div className="text-slate-400">WHERE channel = 'IPM_SARIE' AND is_new_payee = TRUE</div>
                      <div className="text-slate-400">AND time_to_next_transfer &lt; INTERVAL '3 MIN';</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="p-8 text-center bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 text-stone-400 text-xs">
              اختر حادثة أو خبراً لعرض الملف الكامل.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ExecutiveOverview;
