// src/pages/UnifiedExplore.tsx
// الاستكشاف الموحد: رصد ميتا فيو الموحد والسريع للاحتيال المالي والإلكتروني (السعودية / العالمي / الكل)
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Search,
  X,
  ArrowUpDown,
  ExternalLink,
  MapPin,
  Globe2,
  ChevronLeft,
  ChevronRight,
  Filter,
  RotateCcw,
} from 'lucide-react';
import { apiService } from '../services/api';
import {
  SaudiIncident,
  SaudiFraudItem,
  SaudiPattern,
  CountryCount,
} from '../types/saudi';
import { Chip, Unavailable, LoadingBlock, fmtDay } from '../components/ui';
import { IncidentDetailModal } from '../components/IncidentDetailModal';
import { FraudNewsDetailModal } from '../components/FraudNewsDetailModal';
import { PatternDetailModal } from '../components/PatternDetailModal';
import { getNewsImage } from '../utils/newsImages';

type Scope = 'saudi' | 'global' | 'all';
type ItemType = 'all' | 'incident' | 'warning' | 'awareness' | 'pattern';
type BankRel = 'all' | 'high' | 'medium' | 'low';
type SortOption = 'newest' | 'relevance' | 'review_priority';

const COUNTRY_MAP: Record<string, { name: string; flag: string }> = {
  SA: { name: 'المملكة العربية السعودية', flag: '🇸🇦' },
  US: { name: 'الولايات المتحدة', flag: '🇺🇸' },
  GB: { name: 'المملكة المتحدة', flag: '🇬🇧' },
  CA: { name: 'كندا', flag: '🇨🇦' },
  AU: { name: 'أستراليا', flag: '🇦🇺' },
  EG: { name: 'جمهورية مصر العربية', flag: '🇪🇬' },
  AE: { name: 'الإمارات العربية المتحدة', flag: '🇦🇪' },
  NL: { name: 'هولندا', flag: '🇳🇱' },
  Global: { name: 'دولي / عالمي', flag: '🌐' },
};

interface UnifiedItem {
  id: string;
  type: 'incident' | 'warning' | 'awareness' | 'pattern' | 'fraud_news';
  typeLabel: string;
  title: string;
  summary: string;
  date: string | null;
  location?: string | null;
  country?: string | null;
  sourceCount: number;
  sourceName?: string | null;
  sourceCode?: string | null;
  domain?: string | null;
  url?: string | null;
  fraudCategory?: string | null;
  fraudCategoryAr?: string | null;
  bankRelevance?: 'high' | 'medium' | 'low' | null;
  needsReview?: boolean;
  rawIncident?: SaudiIncident;
  rawNews?: SaudiFraudItem;
  rawPattern?: SaudiPattern;
}

const ITEMS_PER_PAGE = 15;

export const UnifiedExplore: React.FC = () => {
  const [params, setParams] = useSearchParams();

  // Search & Filters state
  const [searchQuery, setSearchQuery] = useState(params.get('q') || '');
  const [scope, setScope] = useState<Scope>((params.get('scope') as Scope) || 'saudi');
  const [selectedCountry, setSelectedCountry] = useState<string>(params.get('country') || 'all');
  const [selectedSource, setSelectedSource] = useState<string>(params.get('source') || 'all');
  const [itemType, setItemType] = useState<ItemType>((params.get('type') as ItemType) || 'all');
  const [bankRel, setBankRel] = useState<BankRel>((params.get('bank') as BankRel) || 'all');
  const [windowDays, setWindowDays] = useState<number>(Number(params.get('window')) || 30);
  const [sortBy, setSortBy] = useState<SortOption>('newest');
  const [currentPage, setCurrentPage] = useState<number>(Number(params.get('p')) || 1);

  // Loaded data state
  const [incidents, setIncidents] = useState<SaudiIncident[]>([]);
  const [news, setNews] = useState<SaudiFraudItem[]>([]);
  const [patterns, setPatterns] = useState<SaudiPattern[]>([]);
  const [availableCountries, setAvailableCountries] = useState<CountryCount[]>([]);
  const [loading, setLoading] = useState(true);

  // Detail Modals state
  const [selectedIncident, setSelectedIncident] = useState<SaudiIncident | null>(null);
  const [selectedNews, setSelectedNews] = useState<SaudiFraudItem | null>(null);
  const [selectedPattern, setSelectedPattern] = useState<SaudiPattern | null>(null);

  // Sync state when URL params change
  useEffect(() => {
    const q = params.get('q') || '';
    const sc = (params.get('scope') as Scope) || 'saudi';
    const c = params.get('country') || 'all';
    const s = params.get('source') || 'all';
    const ty = (params.get('type') as ItemType) || 'all';
    const bk = (params.get('bank') as BankRel) || 'all';
    const w = Number(params.get('window')) || 30;
    const p = Number(params.get('p')) || 1;
    const incId = params.get('incident');
    const newsId = params.get('news');

    if (q !== searchQuery) setSearchQuery(q);
    if (sc !== scope) setScope(sc);
    if (c !== selectedCountry) setSelectedCountry(c);
    if (s !== selectedSource) setSelectedSource(s);
    if (ty !== itemType) setItemType(ty);
    if (bk !== bankRel) setBankRel(bk);
    if (w !== windowDays) setWindowDays(w);
    if (p !== currentPage) setCurrentPage(p);

    if (incId) {
      const numId = Number(incId);
      if (!isNaN(numId) && (!selectedIncident || selectedIncident.id !== numId)) {
        apiService.getSaudiIncident(numId).then(setSelectedIncident).catch(() => {});
      }
    } else if (selectedIncident && !incId) {
      setSelectedIncident(null);
    }

    if (newsId) {
      const numNewsId = Number(newsId);
      if (!isNaN(numNewsId) && (!selectedNews || selectedNews.id !== numNewsId)) {
        apiService.getSaudiNewsItem(numNewsId).then(setSelectedNews).catch(() => {});
      }
    } else if (selectedNews && !newsId) {
      setSelectedNews(null);
    }

    const patKey = params.get('pattern');
    if (patKey) {
      if (!selectedPattern || selectedPattern.key !== patKey) {
        apiService.getSaudiPattern(patKey).then((res) => setSelectedPattern(res.pattern)).catch(() => {});
      }
    } else if (selectedPattern && !patKey) {
      setSelectedPattern(null);
    }
  }, [params]);

  // Helper to update filters and push URL params predictably
  const updateFilter = (
    updates: Partial<{
      q: string;
      scope: Scope;
      country: string;
      source: string;
      type: ItemType;
      bank: BankRel;
      window: number;
      p: number;
    }>
  ) => {
    const nextQ = updates.q !== undefined ? updates.q : searchQuery;
    const nextScope = updates.scope !== undefined ? updates.scope : scope;
    const nextCountry = updates.country !== undefined ? updates.country : selectedCountry;
    const nextSource = updates.source !== undefined ? updates.source : selectedSource;
    const nextType = updates.type !== undefined ? updates.type : itemType;
    const nextBank = updates.bank !== undefined ? updates.bank : bankRel;
    const nextWindow = updates.window !== undefined ? updates.window : windowDays;
    const nextP = updates.p !== undefined ? updates.p : 1;

    const p = new URLSearchParams();
    if (nextQ.trim()) p.set('q', nextQ.trim());
    if (nextScope && nextScope !== 'saudi') p.set('scope', nextScope);
    if (nextCountry && nextCountry !== 'all') p.set('country', nextCountry);
    if (nextSource && nextSource !== 'all') p.set('source', nextSource);
    if (nextType && nextType !== 'all') p.set('type', nextType);
    if (nextBank && nextBank !== 'all') p.set('bank', nextBank);
    if (nextWindow !== 30) p.set('window', String(nextWindow));
    if (nextP > 1) p.set('p', String(nextP));
    setParams(p);
  };

  // Load data based on scope, search, and country
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      // 1. Fetch available countries for the scope
      const countryRes = await apiService.getFraudCountries(scope);
      setAvailableCountries(countryRes.countries || []);

      const incParams = {
        scope,
        country: selectedCountry !== 'all' ? selectedCountry : undefined,
        window_days: windowDays > 0 ? windowDays : undefined,
        page_size: 50,
        q: searchQuery || undefined,
      };

      const mergeNewsWithX = (primary: SaudiFraudItem[], xList: SaudiFraudItem[]) => {
        const seen = new Set(primary.map((i) => i.id));
        const merged = [...primary];
        xList.forEach((x) => {
          if (!seen.has(x.id)) {
            seen.add(x.id);
            merged.push(x);
          }
        });
        return merged;
      };

      if (scope === 'saudi') {
        const [incRes, newsRes, xNewsRes, patRes] = await Promise.all([
          apiService.getSaudiIncidents(incParams),
          apiService.getSaudiNews({ scope: 'saudi', page_size: 100, q: searchQuery || undefined }),
          apiService.getSaudiNews({ source: 'x_k_h_almuzaini', page_size: 50, q: searchQuery || undefined }).catch(() => ({ items: [] as SaudiFraudItem[] })),
          apiService.getSaudiPatterns(),
        ]);
        setIncidents(incRes.items || []);
        setNews(mergeNewsWithX(newsRes.items || [], xNewsRes.items || []));
        setPatterns(patRes.patterns || []);
      } else if (scope === 'global') {
        const [incRes, newsRes] = await Promise.all([
          apiService.getSaudiIncidents(incParams),
          apiService.getSaudiNews({
            scope: 'global',
            country: selectedCountry !== 'all' ? selectedCountry : undefined,
            page_size: 100,
            q: searchQuery || undefined,
          }),
        ]);
        setIncidents(incRes.items || []);
        setNews(newsRes.items || []);
        setPatterns([]);
      } else {
        // scope === 'all'
        const [incRes, newsRes, xNewsRes, patRes] = await Promise.all([
          apiService.getSaudiIncidents(incParams),
          apiService.getSaudiNews({
            scope: 'all',
            country: selectedCountry !== 'all' ? selectedCountry : undefined,
            page_size: 100,
            q: searchQuery || undefined,
          }),
          apiService.getSaudiNews({ source: 'x_k_h_almuzaini', page_size: 50, q: searchQuery || undefined }).catch(() => ({ items: [] as SaudiFraudItem[] })),
          apiService.getSaudiPatterns(),
        ]);
        setIncidents(incRes.items || []);
        setNews(
          selectedCountry === 'all' || selectedCountry === 'SA'
            ? mergeNewsWithX(newsRes.items || [], xNewsRes.items || [])
            : newsRes.items || []
        );
        setPatterns(patRes.patterns || []);
      }
    } catch (e) {
      console.error('Failed to load explore items:', e);
    } finally {
      setLoading(false);
    }
  }, [scope, searchQuery, selectedCountry, windowDays]);

  useEffect(() => {
    loadData();
    const handleRefresh = () => loadData();
    window.addEventListener('metaview:refresh', handleRefresh);
    return () => {
      window.removeEventListener('metaview:refresh', handleRefresh);
    };
  }, [loadData]);

  // 1. Gather all items in current scope/country/bank/source without filtering by itemType
  const baseItems: UnifiedItem[] = useMemo(() => {
    const list: UnifiedItem[] = [];
    const seenUrls = new Set<string>();
    const seenIncidentArticleIds = new Set<number>();

    // 1. Incidents
    incidents.forEach((inc) => {
      const incCountry = (inc as any).incident_country;
      const locsStr = Array.isArray(inc.locations) ? inc.locations.join(' ') : (inc.locations || '');
      const isSaudi = incCountry === 'SA' || locsStr.includes('السعودية') || locsStr.includes('الرياض') || locsStr.includes('جدة') || (inc.title && (inc.title.includes('السعودية') || inc.title.includes('الرياض')));

      if (scope === 'saudi' && !isSaudi) return;
      if (scope === 'global' && isSaudi) return;
      if (selectedCountry !== 'all' && incCountry && incCountry !== selectedCountry) return;

      inc.articles?.forEach((art) => {
        if (art.article_id) seenIncidentArticleIds.add(art.article_id);
        if (art.url) seenUrls.add(art.url);
      });

      const normalizedBankRel = inc.bank_relevance ? (inc.bank_relevance.toLowerCase() as any) : null;

      list.push({
        id: `inc-${inc.id}`,
        type: 'incident',
        typeLabel: inc.record_type_ar || 'حادثة مستقلة',
        title: inc.title,
        summary: inc.summary || `حادثة احتيال مرصودة وموثقة عبر ${inc.article_count} مصادر إخبارية ورسمية.`,
        date: inc.first_reported_at,
        location: inc.locations && inc.locations.length > 0 ? inc.locations.join('، ') : null,
        country: incCountry,
        sourceCount: inc.article_count,
        sourceName: inc.agencies?.[0] || 'جهات رسمية وقضائية',
        fraudCategory: inc.fraud_category,
        fraudCategoryAr: inc.fraud_category_ar,
        bankRelevance: normalizedBankRel,
        needsReview: inc.needs_review,
        rawIncident: inc,
      });
    });

    // 2. Patterns (Saudi or All)
    if (scope === 'saudi' || scope === 'all') {
      if (scope !== 'all' || selectedCountry === 'all' || selectedCountry === 'SA') {
        patterns.forEach((p) => {
          list.push({
            id: `pat-${p.key}`,
            type: 'pattern',
            typeLabel: p.pattern_tier_ar || 'تحذير من أسلوب احتيالي',
            title: p.label_ar,
            summary: p.summary || `تحذير من أسلوب احتيالي مستخرج من الأدلة المرصودة.`,
            date: p.last_seen || p.first_seen,
            country: 'SA',
            sourceCount: p.source_count || Math.max(1, (p.incident_count || 0) + (p.warning_article_count || 0)),
            sourceName: p.channels.join('، ') || (p.warning_articles?.[0]?.source || 'مصادر الرصد المعتمدة'),
            bankRelevance: p.bank_relevance || 'medium',
            rawPattern: p,
          });
        });
      }
    }

    // 3. News Items (Saudi warnings/awareness or Global fraud news or All)
    news.forEach((n) => {
      // Deduplication: skip if article already represented in an incident
      if (n.article_id && seenIncidentArticleIds.has(n.article_id)) return;
      if (n.original_url && seenUrls.has(n.original_url)) return;
      if (n.original_url) seenUrls.add(n.original_url);

      const isWarning = n.content_type === 'fraud_warning';
      const isAwareness = n.content_type === 'awareness_campaign';

      if (scope === 'saudi' && !isWarning && !isAwareness) return;

      list.push({
        id: `news-${n.id}`,
        type: isWarning ? 'warning' : isAwareness ? 'awareness' : 'fraud_news',
        typeLabel: n.content_type_ar || (isWarning ? 'تحذير احتيالي' : isAwareness ? 'حملة توعية' : 'خبر احتيال'),
        title: n.headline || n.title,
        summary: n.summary || n.content_excerpt || '',
        date: n.published_at || n.fetched_at,
        country: n.country || (n.is_saudi ? 'SA' : 'Global'),
        sourceCount: 1,
        sourceName: n.source,
        sourceCode: n.source_code,
        domain: n.domain,
        url: n.original_url,
        fraudCategory: n.fraud_category,
        fraudCategoryAr: n.fraud_category_ar,
        bankRelevance: n.bank_relevance,
        rawNews: n,
      });
    });

    // Apply Bank Relevance filter
    let filtered = list;
    if (bankRel !== 'all') {
      filtered = filtered.filter((i) => i.bankRelevance?.toLowerCase() === bankRel.toLowerCase());
    }

    // Apply Source filter
    if (selectedSource !== 'all') {
      filtered = filtered.filter((i) => {
        if (i.sourceCode === selectedSource) return true;
        if (selectedSource === 'x_thehackersnews') {
          return (
            i.domain === 'thehackernews.com' ||
            (i.sourceName && i.sourceName.toLowerCase().includes('hacker news')) ||
            (i.url && i.url.toLowerCase().includes('thehackernews'))
          );
        }
        if (selectedSource === 'x_k_h_almuzaini') {
          return (
            (i.sourceName && i.sourceName.includes('المزيني')) ||
            (i.url && i.url.includes('AlMuzaini'))
          );
        }
        return (
          (i.sourceName && i.sourceName.toLowerCase().includes(selectedSource.toLowerCase())) ||
          (i.rawIncident?.articles?.some(
            (a) =>
              a.source_code === selectedSource ||
              (a.source && a.source.toLowerCase().includes(selectedSource.toLowerCase()))
          ))
        );
      });
    }

    return filtered;
  }, [scope, selectedCountry, bankRel, selectedSource, incidents, patterns, news]);

  // Pre-calculate exact counts for all tabs so users always see the breakdown
  const itemCounts = useMemo(() => {
    let incCount = 0;
    let warnCount = 0;
    let awareCount = 0;
    let patCount = 0;
    let newsCount = 0;
    baseItems.forEach((i) => {
      if (i.type === 'incident') incCount++;
      else if (i.type === 'warning') warnCount++;
      else if (i.type === 'awareness') awareCount++;
      else if (i.type === 'pattern') patCount++;
      else newsCount++;
    });
    const xMuzainiCount = news.filter(
      (n) => n.source_code === 'x_k_h_almuzaini' || Boolean(n.original_url && n.original_url.includes('AlMuzaini'))
    ).length;
    return {
      allCount: baseItems.length,
      incCount,
      warnCount,
      awareCount,
      patCount,
      newsCount,
      xCount: xMuzainiCount,
    };
  }, [baseItems, news]);

  // Filter items by selected ItemType and apply sorting
  const unifiedItems: UnifiedItem[] = useMemo(() => {
    let filtered = baseItems;
    if (itemType === 'incident') {
      filtered = filtered.filter((i) => i.type === 'incident');
    } else if (itemType === 'warning') {
      filtered = filtered.filter((i) => i.type === 'warning');
    } else if (itemType === 'awareness') {
      filtered = filtered.filter((i) => i.type === 'awareness');
    } else if (itemType === 'pattern') {
      filtered = filtered.filter((i) => i.type === 'pattern');
    }

    // Sorting
    return [...filtered].sort((a, b) => {
      if (sortBy === 'review_priority') {
        const aScore = a.needsReview ? 10 : 0;
        const bScore = b.needsReview ? 10 : 0;
        return bScore - aScore;
      }
      if (sortBy === 'relevance') {
        const relScore = (r?: 'high' | 'medium' | 'low' | null) =>
          r === 'high' ? 3 : r === 'medium' ? 2 : 1;
        return relScore(b.bankRelevance) - relScore(a.bankRelevance);
      }
      // default: newest
      return (b.date || '').localeCompare(a.date || '');
    });
  }, [baseItems, itemType, sortBy]);

  // Pagination calculation
  const totalPages = Math.max(1, Math.ceil(unifiedItems.length / ITEMS_PER_PAGE));
  const paginatedItems = useMemo(() => {
    const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
    return unifiedItems.slice(startIndex, startIndex + ITEMS_PER_PAGE);
  }, [unifiedItems, currentPage]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updateFilter({ q: searchQuery, p: 1 });
  };

  const handleOpenIncident = (inc: SaudiIncident) => {
    setSelectedIncident(inc);
    setParams((prev) => {
      const p = new URLSearchParams(prev);
      p.set('incident', String(inc.id));
      return p;
    });
  };

  const handleCloseIncident = () => {
    setSelectedIncident(null);
    setParams((prev) => {
      const p = new URLSearchParams(prev);
      p.delete('incident');
      return p;
    });
  };

  const handleOpenNews = (item: SaudiFraudItem) => {
    setSelectedNews(item);
    setParams((prev) => {
      const p = new URLSearchParams(prev);
      p.set('news', String(item.id));
      return p;
    });
  };

  const handleCloseNews = () => {
    setSelectedNews(null);
    setParams((prev) => {
      const p = new URLSearchParams(prev);
      p.delete('news');
      return p;
    });
  };

  const handleOpenPattern = (pat: SaudiPattern) => {
    setSelectedPattern(pat);
    setParams((prev) => {
      const p = new URLSearchParams(prev);
      p.set('pattern', pat.key);
      return p;
    });
  };

  const handleClosePattern = () => {
    setSelectedPattern(null);
    setParams((prev) => {
      const p = new URLSearchParams(prev);
      p.delete('pattern');
      return p;
    });
  };

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      updateFilter({ p: newPage });
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const isFilterActive =
    searchQuery.trim() !== '' ||
    scope !== 'saudi' ||
    selectedCountry !== 'all' ||
    selectedSource !== 'all' ||
    itemType !== 'all' ||
    bankRel !== 'all' ||
    windowDays !== 30;

  const resetAllFilters = () => {
    setSearchQuery('');
    setScope('saudi');
    setSelectedCountry('all');
    setSelectedSource('all');
    setItemType('all');
    setBankRel('all');
    setWindowDays(30);
    setSortBy('newest');
    setCurrentPage(1);
    setParams(new URLSearchParams());
  };

  return (
    <div className="space-y-5 max-w-5xl mx-auto">
      {/* Search Header */}
      <div className="space-y-3">
        <form onSubmit={handleSearchSubmit} className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={
              scope === 'global'
                ? 'ابحث في أخبار الاحتيال العالمي: بنوك، تصيد، بطاقات، انتحال، نصب...'
                : scope === 'all'
                ? 'ابحث في جميع أخبار وحوادث الاحتيال محلياً وعالمياً...'
                : 'ابحث في الأخبار، الحوادث، التحذيرات، الأنماط، أو مؤشرات الخطر...'
            }
            className="w-full bg-white dark:bg-[#111726] border border-stone-300 dark:border-slate-800 rounded-xl py-3 pr-11 pl-24 text-sm text-stone-900 dark:text-stone-100 placeholder:text-stone-400 dark:placeholder:text-slate-500 shadow-sm focus:border-sky-500 dark:focus:border-sky-500 outline-none transition-all"
          />
          <Search className="w-4 h-4 text-stone-400 dark:text-slate-500 absolute top-1/2 -translate-y-1/2 right-4 pointer-events-none" />
          <div className="absolute top-1/2 -translate-y-1/2 left-3 flex items-center gap-1.5">
            {searchQuery && (
              <button
                type="button"
                onClick={() => {
                  setSearchQuery('');
                  updateFilter({ q: '', p: 1 });
                }}
                className="text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 text-xs p-1"
                title="مسح"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
            <button
              type="submit"
              className="px-3 py-1.5 rounded-lg bg-sky-600 dark:bg-sky-500 text-white font-medium text-xs hover:bg-sky-700 transition-colors"
            >
              بحث
            </button>
          </div>
        </form>

        {/* Filters Multi-bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 text-xs shadow-sm dark:shadow-none">
          <div className="flex flex-wrap items-center gap-2">
            {/* Scope: Saudi vs Global vs All */}
            <div className="inline-flex rounded-lg bg-stone-100 dark:bg-slate-800 p-0.5 border border-stone-200 dark:border-slate-700">
              <button
                type="button"
                onClick={() => {
                  setScope('saudi');
                  setSelectedCountry('all');
                  updateFilter({ scope: 'saudi', country: 'all', p: 1 });
                }}
                className={`px-3 py-1 rounded-md font-medium transition-all ${
                  scope === 'saudi'
                    ? 'bg-white dark:bg-slate-900 text-stone-900 dark:text-stone-100 shadow-sm'
                    : 'text-stone-500 dark:text-slate-400 hover:text-stone-800'
                }`}
              >
                🇸🇦 السعودية
              </button>
              <button
                type="button"
                onClick={() => {
                  setScope('global');
                  updateFilter({ scope: 'global', p: 1 });
                }}
                className={`px-3 py-1 rounded-md font-medium transition-all ${
                  scope === 'global'
                    ? 'bg-white dark:bg-slate-900 text-stone-900 dark:text-stone-100 shadow-sm'
                    : 'text-stone-500 dark:text-slate-400 hover:text-stone-800'
                }`}
              >
                🌐 العالمي
              </button>
              <button
                type="button"
                onClick={() => {
                  setScope('all');
                  updateFilter({ scope: 'all', p: 1 });
                }}
                className={`px-3 py-1 rounded-md font-medium transition-all ${
                  scope === 'all'
                    ? 'bg-white dark:bg-slate-900 text-stone-900 dark:text-stone-100 shadow-sm'
                    : 'text-stone-500 dark:text-slate-400 hover:text-stone-800'
                }`}
              >
                الكل
              </button>
            </div>

            {/* Type selector */}
            <div className="flex items-center gap-1 overflow-x-auto">
              {[
                { key: 'all', label: 'كافة المواد', count: itemCounts.allCount },
                { key: 'incident', label: 'الحوادث الموثقة', count: itemCounts.incCount, isIncident: true },
                { key: 'warning', label: 'التحذيرات', count: itemCounts.warnCount },
                { key: 'awareness', label: 'التوعية', count: itemCounts.awareCount },
                ...(scope !== 'global' ? [{ key: 'pattern', label: 'الأنماط', count: itemCounts.patCount }] : []),
              ].map((t) => (
                <button
                  key={t.key}
                  type="button"
                  id={`btn-type-${t.key}`}
                  onClick={() => {
                    setItemType(t.key as ItemType);
                    updateFilter({ type: t.key as ItemType, p: 1 });
                  }}
                  className={`px-2.5 py-1 rounded-lg border transition-all flex items-center gap-1.5 text-xs ${
                    itemType === t.key
                      ? t.isIncident
                        ? 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-800 dark:text-emerald-300 border-emerald-400 dark:border-emerald-700 font-bold shadow-xs'
                        : 'bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border-sky-300 dark:border-sky-800 font-semibold'
                      : t.isIncident
                      ? 'border-emerald-200 dark:border-emerald-900/40 text-emerald-700 dark:text-emerald-400 bg-emerald-50/40 hover:bg-emerald-50 font-medium'
                      : 'border-transparent text-stone-500 dark:text-slate-400 hover:text-stone-800 dark:hover:text-stone-200'
                  }`}
                >
                  <span>{t.label}</span>
                  <span
                    className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono font-bold ${
                      itemType === t.key
                        ? t.isIncident
                          ? 'bg-emerald-600 text-white'
                          : 'bg-sky-600 text-white'
                        : t.isIncident
                        ? 'bg-emerald-100 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-200'
                        : 'bg-stone-200 dark:bg-slate-800 text-stone-600 dark:text-slate-400'
                    }`}
                  >
                    {t.count}
                  </span>
                </button>
              ))}

              {/* Dedicated X / Dr. Al-Muzaini Source Chip */}
              {itemCounts.xCount > 0 && (
                <button
                  type="button"
                  id="btn-source-x-muzaini"
                  onClick={() => {
                    const next = selectedSource === 'x_k_h_almuzaini' ? 'all' : 'x_k_h_almuzaini';
                    setSelectedSource(next);
                    updateFilter({ source: next, p: 1 });
                  }}
                  className={`px-2.5 py-1 rounded-lg border transition-all flex items-center gap-1.5 text-xs whitespace-nowrap cursor-pointer ${
                    selectedSource === 'x_k_h_almuzaini'
                      ? 'bg-black text-white dark:bg-stone-800 dark:text-stone-100 border-black font-bold shadow-xs'
                      : 'border-stone-200 dark:border-slate-800 text-stone-700 dark:text-slate-300 hover:border-stone-400 bg-white dark:bg-slate-900 font-medium'
                  }`}
                  title="تصفية وعرض منشورات X العامة التابعة لحساب د. خالد بن حمد المزيني"
                >
                  <span className="font-bold">𝕏</span>
                  <span>رصد المزيني</span>
                  <span
                    className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono font-bold ${
                      selectedSource === 'x_k_h_almuzaini'
                        ? 'bg-stone-700 text-white dark:bg-stone-300 dark:text-stone-900'
                        : 'bg-stone-200 dark:bg-slate-800 text-stone-600 dark:text-slate-400'
                    }`}
                  >
                    {itemCounts.xCount}
                  </span>
                </button>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Country filter (Visible when Global or All is selected) */}
            {(scope === 'global' || scope === 'all') && (
              <div className="flex items-center gap-1">
                <Globe2 className="w-3.5 h-3.5 text-stone-400" />
                <select
                  value={selectedCountry}
                  onChange={(e) => {
                    const val = e.target.value;
                    setSelectedCountry(val);
                    updateFilter({ country: val, p: 1 });
                  }}
                  className="bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 rounded-lg px-2.5 py-1 text-xs text-stone-700 dark:text-slate-300 outline-none"
                >
                  <option value="all">كل الدول ({availableCountries.reduce((acc, c) => acc + (c.count || 0), 0)} حوادث موثقة)</option>
                  {availableCountries.map((c) => {
                    const info = COUNTRY_MAP[c.country] || { name: c.country, flag: '🌐' };
                    return (
                      <option key={c.country} value={c.country}>
                        {info.flag} {info.name} ({c.count} حوادث موثقة)
                      </option>
                    );
                  })}
                </select>
              </div>
            )}

            {/* Source filter */}
            <div className="flex items-center gap-1">
              <Filter className="w-3.5 h-3.5 text-stone-400" />
              <select
                value={selectedSource}
                onChange={(e) => {
                  const val = e.target.value;
                  setSelectedSource(val);
                  updateFilter({ source: val, p: 1 });
                }}
                className="bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 rounded-lg px-2.5 py-1 text-xs text-stone-700 dark:text-slate-300 outline-none"
              >
                <option value="all">كافة المصادر</option>
                <option value="x_thehackersnews">The Hacker News (@thehackersnews)</option>
                <option value="x_k_h_almuzaini">د. خالد بن حمد المزيني (@K_H_AlMuzaini)</option>
                <option value="spa">واس (SPA)</option>
                <option value="sabq">صحيفة سبق</option>
                <option value="okaz">صحيفة عكاظ</option>
                <option value="saudigazette">Saudi Gazette</option>
              </select>
            </div>

            {/* Banking relevance filter */}
            <select
              value={bankRel}
              onChange={(e) => {
                const val = e.target.value as BankRel;
                setBankRel(val);
                updateFilter({ bank: val, p: 1 });
              }}
              className="bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 rounded-lg px-2.5 py-1 text-xs text-stone-700 dark:text-slate-300 outline-none"
            >
              <option value="all">كل الصلات البنكية</option>
              <option value="high">صلة بنكية عالية</option>
              <option value="medium">صلة متوسطة</option>
              <option value="low">صلة منخفضة</option>
            </select>

            {/* Time Window filter */}
            <select
              value={windowDays}
              onChange={(e) => {
                const val = Number(e.target.value);
                setWindowDays(val);
                updateFilter({ window: val, p: 1 });
              }}
              className="bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 rounded-lg px-2.5 py-1 text-xs text-stone-700 dark:text-slate-300 outline-none"
            >
              <option value={7}>آخر 7 أيام</option>
              <option value={30}>آخر 30 يوماً (الخريطة والداشبورد)</option>
              <option value={90}>آخر 90 يوماً</option>
              <option value={0}>كافة الفترات</option>
            </select>

            {/* Sort order */}
            <div className="flex items-center gap-1 text-stone-500 dark:text-slate-400">
              <ArrowUpDown className="w-3.5 h-3.5" />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as SortOption)}
                className="bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 rounded-lg px-2.5 py-1 text-xs text-stone-700 dark:text-slate-300 outline-none"
              >
                <option value="newest">الأحدث تاريخاً</option>
                <option value="relevance">حسب الصلة البنكية</option>
                <option value="review_priority">أولوية المراجعة</option>
              </select>
            </div>
            {/* Reset Filters button */}
            {isFilterActive && (
              <button
                type="button"
                id="reset-filters-btn"
                onClick={resetAllFilters}
                className="px-2.5 py-1 rounded-lg text-xs font-semibold text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900/50 hover:bg-rose-100 transition-colors flex items-center gap-1 shadow-xs"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                إعادة ضبط الفلاتر
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Results Count Bar & Scope indicator */}
      <div className="flex flex-wrap items-center justify-between text-xs text-stone-500 dark:text-slate-400 px-1 font-mono gap-2">
        <div className="flex flex-wrap items-center gap-2">
          {itemType === 'all' ? (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="font-semibold text-stone-800 dark:text-stone-200">
                إجمالي المواد المرصودة: {unifiedItems.length} عنصر
              </span>
              <span className="text-stone-300 dark:text-slate-600">|</span>
              <span className="text-[11px] text-stone-600 dark:text-slate-400 flex flex-wrap items-center gap-1">
                منها:
                <button
                  type="button"
                  onClick={() => {
                    setItemType('incident');
                    updateFilter({ type: 'incident', p: 1 });
                  }}
                  className="px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 font-bold hover:bg-emerald-200 transition-colors inline-flex items-center gap-1 cursor-pointer"
                  title="عرض الحوادث الموثقة المطابقة للخريطة فقط"
                >
                  <span>🛡️ {itemCounts.incCount} حوادث موثقة (مطابقة للخريطة 100%)</span>
                </button>
                <span>• {itemCounts.warnCount} تحذيرات</span>
                <span>• {itemCounts.awareCount} توعية</span>
                {itemCounts.patCount > 0 && <span>• {itemCounts.patCount} أنماط</span>}
              </span>
            </div>
          ) : itemType === 'incident' ? (
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 font-bold inline-flex items-center gap-1 text-xs">
                ✓ الحوادث الموثقة: {unifiedItems.length} حادثة (مطابقة تماماً للخريطة والداشبورد)
              </span>
              <button
                type="button"
                onClick={() => {
                  setItemType('all');
                  updateFilter({ type: 'all', p: 1 });
                }}
                className="text-[11px] text-sky-600 dark:text-sky-400 hover:underline cursor-pointer"
              >
                عرض كافة المواد ({itemCounts.allCount}) ↩
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <span className="font-semibold text-stone-800 dark:text-stone-200">
                نتائج التصفية ({itemType === 'warning' ? 'التحذيرات' : itemType === 'awareness' ? 'التوعية' : 'الأنماط'}): {unifiedItems.length} عنصر
              </span>
              <button
                type="button"
                onClick={() => {
                  setItemType('all');
                  updateFilter({ type: 'all', p: 1 });
                }}
                className="text-[11px] text-sky-600 dark:text-sky-400 hover:underline cursor-pointer"
              >
                عرض كافة المواد ({itemCounts.allCount}) ↩
              </button>
            </div>
          )}

          {scope === 'global' && (
            <span className="text-sky-600 dark:text-sky-400">
              (منظومة الرصد العالمي للاحتيال)
            </span>
          )}
          {scope === 'all' && (
            <span className="text-emerald-600 dark:text-emerald-400">
              (دمج الرصد المحلي والعالمي بدون تكرار)
            </span>
          )}
        </div>
        {searchQuery && (
          <span className="text-sky-600 dark:text-sky-400">
            مطابقة لعبارة: «{searchQuery}»
          </span>
        )}
      </div>

      {/* Results List */}
      {loading ? (
        <LoadingBlock label="جارٍ جلب وتحديث نتائج ميتا فيو…" />
      ) : paginatedItems.length === 0 ? (
        <div className="space-y-4 text-center">
          <Unavailable
            title="لم يتم العثور على نتائج"
            note="جرب تغيير كلمات البحث أو اختيار دولة أخرى أو إعادة ضبط خيارات التصفية."
          />
          {isFilterActive && (
            <button
              type="button"
              id="reset-filters-empty-btn"
              onClick={resetAllFilters}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold text-sky-700 dark:text-sky-300 bg-sky-50 dark:bg-sky-950/50 border border-sky-200 dark:border-sky-800 hover:bg-sky-100 transition-all shadow-xs"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              إعادة ضبط الفلاتر والبحث
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {paginatedItems.map((item) => {
            const isInc = item.type === 'incident';
            const countryInfo = (item.country && COUNTRY_MAP[item.country]) || {
              name: item.country || 'دولي',
              flag: '🌐',
            };

            const photoUrl = getNewsImage({
              title: item.title,
              summary: item.summary,
              fraud_category: item.fraudCategory || '',
            });

            return (
              <div
                key={item.id}
                onClick={() => {
                  if (isInc && item.rawIncident) {
                    handleOpenIncident(item.rawIncident);
                  } else if (item.type === 'pattern' && item.rawPattern) {
                    handleOpenPattern(item.rawPattern);
                  } else if (item.rawNews) {
                    handleOpenNews(item.rawNews);
                  }
                }}
                className="panel p-4 transition-all cursor-pointer hover:border-sky-300 dark:hover:border-sky-800"
              >
                <div className="flex items-start gap-3.5 sm:gap-4">
                  {/* Photo Thumbnail */}
                  <img
                    src={photoUrl}
                    alt=""
                    className="w-20 h-20 sm:w-24 sm:h-24 rounded-xl object-cover shrink-0 border border-stone-200/80 dark:border-slate-800 shadow-xs"
                    loading="lazy"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src =
                        'https://images.unsplash.com/photo-1559526324-4b87b5e36e44?auto=format&fit=crop&w=300&q=80';
                    }}
                  />

                  {/* Card Content */}
                  <div className="space-y-2 min-w-0 flex-1">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <Chip
                          tone={
                            item.type === 'incident'
                              ? 'critical'
                              : item.type === 'warning'
                              ? 'high'
                              : item.type === 'pattern'
                              ? 'brand'
                              : 'plain'
                          }
                        >
                          {item.typeLabel}
                        </Chip>

                        {/* Country Badge */}
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-stone-100 dark:bg-slate-800 text-stone-700 dark:text-slate-300 border border-stone-200 dark:border-slate-700">
                          <span>{countryInfo.flag}</span>
                          <span>{countryInfo.name}</span>
                        </span>

                        {/* Fraud Category */}
                        {item.fraudCategoryAr && (
                          <Chip tone="brand">{item.fraudCategoryAr}</Chip>
                        )}

                        {/* Bank Relevance */}
                        {item.bankRelevance && (
                          <Chip tone={item.bankRelevance === 'high' ? 'critical' : 'medium'}>
                            صلة بنكية {item.bankRelevance === 'high' ? 'عالية' : 'متوسطة'}
                          </Chip>
                        )}

                        {item.needsReview && (
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-50 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-800">
                            تنتظر مراجعة
                          </span>
                        )}

                        {/* X Platform Badge */}
                        {(item.sourceCode?.startsWith('x_') || item.sourceName?.includes('@')) && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono bg-black text-white dark:bg-stone-800 dark:text-stone-200">
                            <span className="font-bold">𝕏</span>
                            <span>{item.sourceCode === 'x_k_h_almuzaini' ? '@K_H_AlMuzaini' : '@thehackersnews'}</span>
                          </span>
                        )}
                      </div>

                      <span className="text-xs font-mono text-stone-500 dark:text-slate-400">
                        {fmtDay(item.date)}
                      </span>
                    </div>

                    <div className="flex items-start justify-between gap-3">
                      <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100 leading-snug">
                        {item.title}
                      </h3>
                      {item.url && (
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          className="text-stone-400 hover:text-sky-600 dark:hover:text-sky-400 p-1 shrink-0"
                          title="فتح الرابط الأصلي"
                        >
                          <ExternalLink className="w-3.5 h-3.5" />
                        </a>
                      )}
                    </div>

                    {item.summary && (
                      <p className="text-xs text-stone-600 dark:text-slate-400 leading-relaxed line-clamp-2">
                        {item.summary}
                      </p>
                    )}

                    <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-xs text-stone-500 dark:text-slate-400">
                      <div className="flex items-center gap-3">
                        {item.sourceName && (
                          <span className="font-medium text-stone-700 dark:text-slate-300">
                            {item.sourceName}
                          </span>
                        )}
                        {item.location && (
                          <span className="flex items-center gap-1">
                            <MapPin className="w-3 h-3 text-sky-600 dark:text-sky-400" />
                            {item.location}
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-2">
                        {item.type === 'pattern' && item.rawPattern ? (
                          <div className="flex flex-wrap items-center gap-1.5">
                            <span className="font-mono text-[11px] bg-stone-100 dark:bg-slate-800 px-2 py-0.5 rounded text-stone-600 dark:text-slate-300">
                              حوادث: {item.rawPattern.incident_count}
                            </span>
                            <span className="font-mono text-[11px] bg-stone-100 dark:bg-slate-800 px-2 py-0.5 rounded text-stone-600 dark:text-slate-300">
                              تحذيرات: {item.rawPattern.warning_article_count}
                            </span>
                            <span className="font-mono text-[11px] bg-sky-50 dark:bg-sky-950/60 text-sky-700 dark:text-sky-300 px-2 py-0.5 rounded border border-sky-200 dark:border-sky-800 font-medium">
                              مصادر: {item.sourceCount}
                            </span>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleOpenPattern(item.rawPattern!);
                              }}
                              className="text-sky-600 dark:text-sky-400 text-xs font-semibold hover:underline cursor-pointer mr-1"
                            >
                              عرض التفاصيل والتحليل ←
                            </button>
                          </div>
                        ) : (
                          <>
                            <span className="font-mono text-[11px] bg-stone-100 dark:bg-slate-800 px-2 py-0.5 rounded text-stone-600 dark:text-slate-300">
                              {item.sourceCount} {isInc ? 'مصادر إخبارية' : 'مصدر أصلي'}
                            </span>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                if (isInc && item.rawIncident) handleOpenIncident(item.rawIncident);
                                else if (item.rawNews) handleOpenNews(item.rawNews);
                              }}
                              className="text-sky-600 dark:text-sky-400 text-xs font-medium hover:underline cursor-pointer"
                            >
                              {isInc ? 'عرض تفاصيل الحادثة والمؤشرات ←' : 'عرض التفاصيل والتحليل ←'}
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between p-3 rounded-xl bg-white dark:bg-[#111726] border border-stone-200 dark:border-slate-800 text-xs">
          <div className="text-stone-500 dark:text-slate-400 font-mono">
            صفحة {currentPage} من {totalPages} ({unifiedItems.length} عنصر إجمالي)
          </div>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              disabled={currentPage <= 1}
              onClick={() => handlePageChange(currentPage - 1)}
              className="p-1.5 rounded-lg border border-stone-200 dark:border-slate-700 text-stone-600 dark:text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-stone-50 dark:hover:bg-slate-800 transition-colors"
              title="الصفحة السابقة"
            >
              <ChevronRight className="w-4 h-4" />
            </button>

            {Array.from({ length: totalPages }, (_, i) => i + 1)
              .filter((p) => p === 1 || p === totalPages || Math.abs(p - currentPage) <= 1)
              .map((p, idx, arr) => {
                const showEllipsis = idx > 0 && p - arr[idx - 1] > 1;
                return (
                  <React.Fragment key={p}>
                    {showEllipsis && <span className="px-1 text-stone-400">…</span>}
                    <button
                      type="button"
                      onClick={() => handlePageChange(p)}
                      className={`min-w-7 h-7 rounded-lg font-mono font-medium transition-colors ${
                        p === currentPage
                          ? 'bg-sky-600 text-white'
                          : 'text-stone-600 dark:text-slate-400 hover:bg-stone-100 dark:hover:bg-slate-800'
                      }`}
                    >
                      {p}
                    </button>
                  </React.Fragment>
                );
              })}

            <button
              type="button"
              disabled={currentPage >= totalPages}
              onClick={() => handlePageChange(currentPage + 1)}
              className="p-1.5 rounded-lg border border-stone-200 dark:border-slate-700 text-stone-600 dark:text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-stone-50 dark:hover:bg-slate-800 transition-colors"
              title="الصفحة التالية"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Incident Detail Modal */}
      <IncidentDetailModal
        incident={selectedIncident}
        onClose={handleCloseIncident}
      />

      {/* Fraud News Detail Modal */}
      <FraudNewsDetailModal
        item={selectedNews}
        onClose={handleCloseNews}
      />

      {/* Pattern Detail Modal */}
      <PatternDetailModal
        pattern={selectedPattern}
        onClose={handleClosePattern}
        onOpenIncident={(incId) => {
          handleClosePattern();
          const inc = incidents.find((i) => i.id === incId);
          if (inc) {
            handleOpenIncident(inc);
          } else {
            apiService.getSaudiIncident(incId).then(handleOpenIncident).catch(() => {});
          }
        }}
      />
    </div>
  );
};

export default UnifiedExplore;
