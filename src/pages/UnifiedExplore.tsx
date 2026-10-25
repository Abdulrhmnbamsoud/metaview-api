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
  const [itemType, setItemType] = useState<ItemType>((params.get('type') as ItemType) || 'all');
  const [bankRel, setBankRel] = useState<BankRel>((params.get('bank') as BankRel) || 'all');
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

  // Sync state when URL params change
  useEffect(() => {
    const q = params.get('q') || '';
    const sc = (params.get('scope') as Scope) || 'saudi';
    const c = params.get('country') || 'all';
    const ty = (params.get('type') as ItemType) || 'all';
    const bk = (params.get('bank') as BankRel) || 'all';
    const p = Number(params.get('p')) || 1;
    const incId = params.get('incident');
    const newsId = params.get('news');

    if (q !== searchQuery) setSearchQuery(q);
    if (sc !== scope) setScope(sc);
    if (c !== selectedCountry) setSelectedCountry(c);
    if (ty !== itemType) setItemType(ty);
    if (bk !== bankRel) setBankRel(bk);
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
  }, [params]);

  // Helper to update filters and push URL params predictably
  const updateFilter = (
    updates: Partial<{
      q: string;
      scope: Scope;
      country: string;
      type: ItemType;
      bank: BankRel;
      p: number;
    }>
  ) => {
    const nextQ = updates.q !== undefined ? updates.q : searchQuery;
    const nextScope = updates.scope !== undefined ? updates.scope : scope;
    const nextCountry = updates.country !== undefined ? updates.country : selectedCountry;
    const nextType = updates.type !== undefined ? updates.type : itemType;
    const nextBank = updates.bank !== undefined ? updates.bank : bankRel;
    const nextP = updates.p !== undefined ? updates.p : 1;

    const p = new URLSearchParams();
    if (nextQ.trim()) p.set('q', nextQ.trim());
    if (nextScope && nextScope !== 'saudi') p.set('scope', nextScope);
    if (nextCountry && nextCountry !== 'all') p.set('country', nextCountry);
    if (nextType && nextType !== 'all') p.set('type', nextType);
    if (nextBank && nextBank !== 'all') p.set('bank', nextBank);
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

      if (scope === 'saudi') {
        const [incRes, newsRes, patRes] = await Promise.all([
          apiService.getSaudiIncidents({ page_size: 50, q: searchQuery || undefined }),
          apiService.getSaudiNews({ scope: 'saudi', page_size: 100, q: searchQuery || undefined }),
          apiService.getSaudiPatterns(),
        ]);
        setIncidents(incRes.items || []);
        setNews(newsRes.items || []);
        setPatterns(patRes.patterns || []);
      } else if (scope === 'global') {
        const newsRes = await apiService.getSaudiNews({
          scope: 'global',
          country: selectedCountry !== 'all' ? selectedCountry : undefined,
          page_size: 100,
          q: searchQuery || undefined,
        });
        setIncidents([]);
        setNews(newsRes.items || []);
        setPatterns([]);
      } else {
        // scope === 'all'
        const [incRes, newsRes, patRes] = await Promise.all([
          apiService.getSaudiIncidents({ page_size: 50, q: searchQuery || undefined }),
          apiService.getSaudiNews({
            scope: 'all',
            country: selectedCountry !== 'all' ? selectedCountry : undefined,
            page_size: 100,
            q: searchQuery || undefined,
          }),
          apiService.getSaudiPatterns(),
        ]);
        setIncidents(incRes.items || []);
        setNews(newsRes.items || []);
        setPatterns(patRes.patterns || []);
      }
    } catch (e) {
      console.error('Failed to load explore items:', e);
    } finally {
      setLoading(false);
    }
  }, [scope, searchQuery, selectedCountry]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Unify, deduplicate, and filter items into a single coherent list
  const unifiedItems: UnifiedItem[] = useMemo(() => {
    const list: UnifiedItem[] = [];
    const seenUrls = new Set<string>();
    const seenIncidentArticleIds = new Set<number>();

    // 1. Incidents (Saudi or All)
    if (scope === 'saudi' || scope === 'all') {
      if (itemType === 'all' || itemType === 'incident') {
        incidents.forEach((inc) => {
          // If country filter is applied on 'all', check if SA is chosen
          if (scope === 'all' && selectedCountry !== 'all' && selectedCountry !== 'SA') {
            return;
          }

          inc.articles?.forEach((art) => {
            if (art.article_id) seenIncidentArticleIds.add(art.article_id);
            if (art.url) seenUrls.add(art.url);
          });

          list.push({
            id: `inc-${inc.id}`,
            type: 'incident',
            typeLabel: inc.record_type_ar || 'حادثة مستقلة',
            title: inc.title,
            summary: inc.summary || `حادثة احتيال مرصودة وموثقة عبر ${inc.article_count} مصادر إخبارية ورسمية.`,
            date: inc.first_reported_at,
            location: inc.locations.length > 0 ? inc.locations.join('، ') : null,
            country: 'SA',
            sourceCount: inc.article_count,
            sourceName: inc.agencies?.[0] || 'جهات رسمية وقضائية',
            fraudCategory: inc.fraud_category,
            fraudCategoryAr: inc.fraud_category_ar,
            bankRelevance: inc.bank_relevance,
            needsReview: inc.needs_review,
            rawIncident: inc,
          });
        });
      }
    }

    // 2. Patterns (Saudi or All)
    if (scope === 'saudi' || scope === 'all') {
      if (itemType === 'all' || itemType === 'pattern') {
        if (scope !== 'all' || selectedCountry === 'all' || selectedCountry === 'SA') {
          patterns.forEach((p) => {
            list.push({
              id: `pat-${p.key}`,
              type: 'pattern',
              typeLabel: 'نمط احتيال',
              title: p.label_ar,
              summary: `نمط احتيالي متكرر مستخرج من ${p.incident_count} حوادث و${p.warning_article_count} تحذيرات.`,
              date: p.last_seen || p.first_seen,
              country: 'SA',
              sourceCount: p.incident_count + p.warning_article_count,
              sourceName: p.channels.join('، ') || 'قنوات متعددة',
              bankRelevance: 'high',
              rawPattern: p,
            });
          });
        }
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

      if (scope === 'saudi') {
        // In Saudi view, incident articles are grouped under incidents.
        // Only show separate warnings or awareness campaigns if requested
        if (itemType === 'warning' && !isWarning) return;
        if (itemType === 'awareness' && !isAwareness) return;
        if (itemType === 'incident') return; // incidents handled above
        if (itemType === 'pattern') return; // patterns handled above
        if (itemType === 'all' && !isWarning && !isAwareness) return;
      } else if (scope === 'global') {
        if (itemType === 'warning' && !isWarning) return;
        if (itemType === 'awareness' && !isAwareness) return;
        if (itemType === 'incident' && n.content_type !== 'confirmed_incident' && n.content_type !== 'reported_incident') return;
      } else {
        // scope === 'all'
        if (itemType === 'warning' && !isWarning) return;
        if (itemType === 'awareness' && !isAwareness) return;
        if (itemType === 'incident' && n.content_type !== 'confirmed_incident' && n.content_type !== 'reported_incident') return;
      }

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
      filtered = filtered.filter((i) => i.bankRelevance === bankRel);
    }

    // Sorting
    return filtered.sort((a, b) => {
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
  }, [scope, itemType, bankRel, sortBy, selectedCountry, incidents, news, patterns]);

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

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      updateFilter({ p: newPage });
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
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
                { key: 'all', label: 'كافة الأنواع' },
                ...(scope !== 'global' ? [{ key: 'incident', label: 'الحوادث' }] : []),
                { key: 'warning', label: 'التحذيرات' },
                { key: 'awareness', label: 'التوعية' },
                ...(scope !== 'global' ? [{ key: 'pattern', label: 'الأنماط' }] : []),
              ].map((t) => (
                <button
                  key={t.key}
                  type="button"
                  onClick={() => {
                    setItemType(t.key as ItemType);
                    updateFilter({ type: t.key as ItemType, p: 1 });
                  }}
                  className={`px-2.5 py-1 rounded-lg border transition-colors ${
                    itemType === t.key
                      ? 'bg-sky-50 dark:bg-sky-950/40 text-sky-700 dark:text-sky-300 border-sky-300 dark:border-sky-800 font-semibold'
                      : 'border-transparent text-stone-500 dark:text-slate-400 hover:text-stone-800 dark:hover:text-stone-200'
                  }`}
                >
                  {t.label}
                </button>
              ))}
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
                  <option value="all">كل الدول ({availableCountries.reduce((acc, c) => acc + c.count, 0)})</option>
                  {availableCountries.map((c) => {
                    const info = COUNTRY_MAP[c.country] || { name: c.country, flag: '🌐' };
                    return (
                      <option key={c.country} value={c.country}>
                        {info.flag} {info.name} ({c.count})
                      </option>
                    );
                  })}
                </select>
              </div>
            )}

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
          </div>
        </div>
      </div>

      {/* Results Count Bar & Scope indicator */}
      <div className="flex items-center justify-between text-xs text-stone-500 dark:text-slate-400 px-1 font-mono">
        <div className="flex items-center gap-2">
          <span>نتائج البحث والتصفية: {unifiedItems.length} عنصر</span>
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
        <Unavailable
          title="لم يتم العثور على نتائج"
          note="جرب تغيير كلمات البحث أو اختيار دولة أخرى أو إعادة ضبط خيارات التصفية."
        />
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
                        <span className="font-mono text-[11px] bg-stone-100 dark:bg-slate-800 px-2 py-0.5 rounded text-stone-600 dark:text-slate-300">
                          {item.sourceCount} {isInc ? 'مصادر إخبارية' : 'مصدر أصلي'}
                        </span>
                        <span className="text-sky-600 dark:text-sky-400 text-xs font-medium hover:underline">
                          {isInc ? 'عرض تفاصيل الحادثة والمؤشرات ←' : 'عرض التفاصيل والتحليل ←'}
                        </span>
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
    </div>
  );
};

export default UnifiedExplore;
