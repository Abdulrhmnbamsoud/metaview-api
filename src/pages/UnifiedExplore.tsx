// src/pages/UnifiedExplore.tsx
// الاستكشاف الموحد: قائمة استخباراتية موحدة وسريعة البحث والتصفية مع حفظ الحالة في الرابط
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Search,
  X,
  ArrowUpDown,
  ExternalLink,
  MapPin,
  ShieldAlert,
  AlertTriangle,
  BellRing,
  Network,
  Globe2,
} from 'lucide-react';
import { apiService } from '../services/api';
import {
  SaudiIncident,
  SaudiFraudItem,
  SaudiPattern,
} from '../types/saudi';
import { Article } from '../types';
import { Chip, Unavailable, LoadingBlock, fmtDay } from '../components/ui';
import { IncidentDetailModal } from '../components/IncidentDetailModal';

type Scope = 'saudi' | 'global';
type ItemType = 'all' | 'incident' | 'warning' | 'awareness' | 'pattern';
type BankRel = 'all' | 'high' | 'medium' | 'low';
type SortOption = 'newest' | 'relevance' | 'review_priority';

interface UnifiedItem {
  id: string;
  type: 'incident' | 'warning' | 'awareness' | 'pattern' | 'global_article';
  typeLabel: string;
  title: string;
  summary: string;
  date: string | null;
  location?: string | null;
  sourceCount: number;
  sourceName?: string | null;
  url?: string | null;
  bankRelevance?: 'high' | 'medium' | 'low' | null;
  needsReview?: boolean;
  rawIncident?: SaudiIncident;
  rawNews?: SaudiFraudItem;
  rawPattern?: SaudiPattern;
}

export const UnifiedExplore: React.FC = () => {
  const [params, setParams] = useSearchParams();

  // Search & Filters state (initialized from URL params for perfect back/forward navigation)
  const [searchQuery, setSearchQuery] = useState(params.get('q') || '');
  const [scope, setScope] = useState<Scope>((params.get('scope') as Scope) || 'saudi');
  const [itemType, setItemType] = useState<ItemType>((params.get('type') as ItemType) || 'all');
  const [bankRel, setBankRel] = useState<BankRel>((params.get('bank') as BankRel) || 'all');
  const [sortBy, setSortBy] = useState<SortOption>('newest');

  // Loaded data state
  const [incidents, setIncidents] = useState<SaudiIncident[]>([]);
  const [news, setNews] = useState<SaudiFraudItem[]>([]);
  const [patterns, setPatterns] = useState<SaudiPattern[]>([]);
  const [globalArticles, setGlobalArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(true);

  // Detail Modal state
  const [selectedIncident, setSelectedIncident] = useState<SaudiIncident | null>(null);

  // Sync state when URL params change (e.g. browser back/forward buttons)
  useEffect(() => {
    const q = params.get('q') || '';
    const sc = (params.get('scope') as Scope) || 'saudi';
    const ty = (params.get('type') as ItemType) || 'all';
    const bk = (params.get('bank') as BankRel) || 'all';
    const incId = params.get('incident');

    if (q !== searchQuery) setSearchQuery(q);
    if (sc !== scope) setScope(sc);
    if (ty !== itemType) setItemType(ty);
    if (bk !== bankRel) setBankRel(bk);

    if (incId) {
      const numId = Number(incId);
      if (!isNaN(numId) && (!selectedIncident || selectedIncident.id !== numId)) {
        apiService.getSaudiIncident(numId).then(setSelectedIncident).catch(() => {});
      }
    } else if (selectedIncident && !incId) {
      setSelectedIncident(null);
    }
  }, [params]);

  // Helper to update filters and push URL params predictably
  const updateFilter = (updates: Partial<{ q: string; scope: Scope; type: ItemType; bank: BankRel }>) => {
    const nextQ = updates.q !== undefined ? updates.q : searchQuery;
    const nextScope = updates.scope !== undefined ? updates.scope : scope;
    const nextType = updates.type !== undefined ? updates.type : itemType;
    const nextBank = updates.bank !== undefined ? updates.bank : bankRel;

    const p = new URLSearchParams();
    if (nextQ.trim()) p.set('q', nextQ.trim());
    if (nextScope && nextScope !== 'saudi') p.set('scope', nextScope);
    if (nextType && nextType !== 'all') p.set('type', nextType);
    if (nextBank && nextBank !== 'all') p.set('bank', nextBank);
    setParams(p);
  };

  // Load data based on scope and search
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      if (scope === 'saudi') {
        const [incRes, newsRes, patRes] = await Promise.all([
          apiService.getSaudiIncidents({ page_size: 50, q: searchQuery || undefined }),
          apiService.getSaudiNews({ page_size: 100, q: searchQuery || undefined }),
          apiService.getSaudiPatterns(),
        ]);
        setIncidents(incRes.items || []);
        setNews(newsRes.items || []);
        setPatterns(patRes.patterns || []);
      } else {
        const res = searchQuery
          ? await apiService.searchText({ q: searchQuery, page_size: 50 })
          : await apiService.getArticles({ page_size: 50 });
        setGlobalArticles(res.results || []);
      }
    } catch (e) {
      console.error('Failed to load explore items:', e);
    } finally {
      setLoading(false);
    }
  }, [scope, searchQuery]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Unify and filter items into a single coherent list
  const unifiedItems: UnifiedItem[] = useMemo(() => {
    if (scope === 'global') {
      return globalArticles.map((a, idx) => ({
        id: `global-${idx}`,
        type: 'global_article',
        typeLabel: 'خبر عالمي',
        title: a.headline,
        summary: a.article_summary || a.content || '',
        date: a.published_at,
        sourceCount: 1,
        sourceName: a.source,
        url: a.url,
        bankRelevance: null,
      }));
    }

    const list: UnifiedItem[] = [];

    // Incidents
    if (itemType === 'all' || itemType === 'incident') {
      incidents.forEach((inc) => {
        list.push({
          id: `inc-${inc.id}`,
          type: 'incident',
          typeLabel: inc.record_type_ar || 'حادثة مستقلة',
          title: inc.title,
          summary: inc.summary || `حادثة احتيال مرصودة وموثقة عبر ${inc.article_count} مصادر إخبارية ورسمية.`,
          date: inc.first_reported_at,
          location: inc.locations.length > 0 ? inc.locations.join('، ') : null,
          sourceCount: inc.article_count,
          sourceName: inc.agencies?.[0] || 'جهات رسمية وقضائية',
          bankRelevance: inc.bank_relevance,
          needsReview: inc.needs_review,
          rawIncident: inc,
        });
      });
    }

    // Warnings & Awareness News (only those outside incidents to avoid duplicate entries)
    if (itemType === 'all' || itemType === 'warning' || itemType === 'awareness') {
      news.forEach((n) => {
        const isWarning = n.content_type === 'fraud_warning';
        const isAwareness = n.content_type === 'awareness_campaign';

        if ((itemType === 'warning' && !isWarning) || (itemType === 'awareness' && !isAwareness)) {
          return;
        }

        if (itemType === 'all' && !isWarning && !isAwareness) {
          return;
        }

        list.push({
          id: `news-${n.id}`,
          type: isWarning ? 'warning' : 'awareness',
          typeLabel: isWarning ? 'تحذير احتيالي' : 'حملة توعية',
          title: n.headline || n.title,
          summary: n.summary || n.content_excerpt || '',
          date: n.published_at || n.fetched_at,
          sourceCount: 1,
          sourceName: n.source,
          url: n.original_url,
          bankRelevance: n.bank_relevance,
          rawNews: n,
        });
      });
    }

    // Patterns
    if (itemType === 'all' || itemType === 'pattern') {
      patterns.forEach((p) => {
        list.push({
          id: `pat-${p.key}`,
          type: 'pattern',
          typeLabel: 'نمط احتيال',
          title: p.label_ar,
          summary: `نمط احتيالي متكرر مستخرج من ${p.incident_count} حوادث و${p.warning_article_count} تحذيرات.`,
          date: p.last_seen || p.first_seen,
          sourceCount: p.incident_count + p.warning_article_count,
          sourceName: p.channels.join('، ') || 'قنوات متعددة',
          bankRelevance: 'high',
          rawPattern: p,
        });
      });
    }

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
        const relScore = (r?: 'high' | 'medium' | 'low' | null) => (r === 'high' ? 3 : r === 'medium' ? 2 : 1);
        return relScore(b.bankRelevance) - relScore(a.bankRelevance);
      }
      // default: newest
      return (b.date || '').localeCompare(a.date || '');
    });
  }, [scope, itemType, bankRel, sortBy, incidents, news, patterns, globalArticles]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updateFilter({ q: searchQuery });
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

  return (
    <div className="space-y-5 max-w-5xl mx-auto">
      {/* Search Header */}
      <div className="space-y-3">
        <form onSubmit={handleSearchSubmit} className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="ابحث في الأخبار، الحوادث، التحذيرات، الأنماط، أو مؤشرات الخطر..."
            className="w-full bg-white dark:bg-[#111726] border border-stone-300 dark:border-slate-800 rounded-xl py-3 pr-11 pl-24 text-sm text-stone-900 dark:text-stone-100 placeholder:text-stone-400 dark:placeholder:text-slate-500 shadow-sm focus:border-sky-500 dark:focus:border-sky-500 outline-none transition-all"
          />
          <Search className="w-4 h-4 text-stone-400 dark:text-slate-500 absolute top-1/2 -translate-y-1/2 right-4 pointer-events-none" />
          <div className="absolute top-1/2 -translate-y-1/2 left-3 flex items-center gap-1.5">
            {searchQuery && (
              <button
                type="button"
                onClick={() => {
                  setSearchQuery('');
                  updateFilter({ q: '' });
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
            {/* Scope: Saudi vs Global */}
            <div className="inline-flex rounded-lg bg-stone-100 dark:bg-slate-800 p-0.5 border border-stone-200 dark:border-slate-700">
              <button
                type="button"
                onClick={() => {
                  setScope('saudi');
                  updateFilter({ scope: 'saudi' });
                }}
                className={`px-3 py-1 rounded-md font-medium transition-all ${
                  scope === 'saudi'
                    ? 'bg-white dark:bg-slate-900 text-stone-900 dark:text-stone-100 shadow-sm'
                    : 'text-stone-500 dark:text-slate-400 hover:text-stone-800'
                }`}
              >
                السعودية
              </button>
              <button
                type="button"
                onClick={() => {
                  setScope('global');
                  updateFilter({ scope: 'global' });
                }}
                className={`px-3 py-1 rounded-md font-medium transition-all ${
                  scope === 'global'
                    ? 'bg-white dark:bg-slate-900 text-stone-900 dark:text-stone-100 shadow-sm'
                    : 'text-stone-500 dark:text-slate-400 hover:text-stone-800'
                }`}
              >
                العالمي
              </button>
            </div>

            {/* Type selector (Saudi only) */}
            {scope === 'saudi' && (
              <div className="flex items-center gap-1 overflow-x-auto">
                {[
                  { key: 'all', label: 'الكل' },
                  { key: 'incident', label: 'الحوادث' },
                  { key: 'warning', label: 'التحذيرات' },
                  { key: 'awareness', label: 'التوعية' },
                  { key: 'pattern', label: 'الأنماط' },
                ].map((t) => (
                  <button
                    key={t.key}
                    type="button"
                    onClick={() => {
                      setItemType(t.key as ItemType);
                      updateFilter({ type: t.key as ItemType });
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
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Banking relevance filter */}
            {scope === 'saudi' && (
              <select
                value={bankRel}
                onChange={(e) => {
                  const val = e.target.value as BankRel;
                  setBankRel(val);
                  updateFilter({ bank: val });
                }}
                className="bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 rounded-lg px-2.5 py-1 text-xs text-stone-700 dark:text-slate-300 outline-none"
              >
                <option value="all">كل الصلات البنكية</option>
                <option value="high">صلة بنكية عالية</option>
                <option value="medium">صلة متوسطة</option>
                <option value="low">صلة منخفضة</option>
              </select>
            )}

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

      {/* Results Count Bar */}
      <div className="flex items-center justify-between text-xs text-stone-500 dark:text-slate-400 px-1 font-mono">
        <span>نتائج البحث والتصفية: {unifiedItems.length} عنصر</span>
        {searchQuery && (
          <span className="text-sky-600 dark:text-sky-400">
            مطابقة لعبارة: «{searchQuery}»
          </span>
        )}
      </div>

      {/* Results List */}
      {loading ? (
        <LoadingBlock label="جارٍ جلب وتحديث النتائج…" />
      ) : unifiedItems.length === 0 ? (
        <Unavailable
          title="لم يتم العثور على نتائج"
          note="جرب تغيير كلمات البحث أو إعادة ضبط خيارات التصفية."
        />
      ) : (
        <div className="space-y-3">
          {unifiedItems.map((item) => {
            const isInc = item.type === 'incident';
            return (
              <div
                key={item.id}
                onClick={() => {
                  if (isInc && item.rawIncident) {
                    handleOpenIncident(item.rawIncident);
                  }
                }}
                className={`panel p-4 transition-all space-y-2 ${
                  isInc
                    ? 'cursor-pointer hover:border-sky-300 dark:hover:border-sky-800'
                    : 'hover:border-stone-300 dark:hover:border-slate-700'
                }`}
              >
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
                    {item.sourceName && <span>{item.sourceName}</span>}
                    {item.location && (
                      <span className="flex items-center gap-1">
                        <MapPin className="w-3 h-3 text-sky-600 dark:text-sky-400" />
                        {item.location}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[11px] bg-stone-100 dark:bg-slate-800 px-2 py-0.5 rounded text-stone-600 dark:text-slate-300">
                      {item.sourceCount} {isInc ? 'مصادر إخبارية' : 'مصدر'}
                    </span>
                    {isInc && (
                      <span className="text-sky-600 dark:text-sky-400 text-xs font-medium hover:underline">
                        عرض التفاصيل والمؤشرات ←
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Incident Detail Modal */}
      <IncidentDetailModal
        incident={selectedIncident}
        onClose={handleCloseIncident}
      />
    </div>
  );
};

export default UnifiedExplore;
