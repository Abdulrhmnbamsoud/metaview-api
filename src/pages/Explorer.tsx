// src/pages/Explorer.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Search,
  Filter,
  X,
  Calendar,
  Building2,
  Globe,
  ArrowUpDown,
  ChevronRight,
  ChevronLeft,
  RotateCcw,
} from 'lucide-react';
import { apiService } from '../services/api';
import { Article, SearchParams, SourcesResponse, CountryNewsCount } from '../types';
import { NewsCard } from '../components/NewsCard';
import { ArticleDetailModal } from '../components/ArticleDetailModal';
import { NewsGridSkeleton } from '../components/LoadingSkeleton';
import { EmptyState } from '../components/EmptyState';
import { ErrorBanner } from '../components/ErrorBanner';

export const Explorer: React.FC = () => {
  const [urlParams, setUrlParams] = useSearchParams();

  const [query, setQuery] = useState(urlParams.get('q') || '');
  const [selectedSource, setSelectedSource] = useState(urlParams.get('source') || 'all');
  const [selectedCountry, setSelectedCountry] = useState(urlParams.get('country') || 'all');
  const [minDate, setMinDate] = useState(urlParams.get('min_date') || '');
  const [maxDate, setMaxDate] = useState(urlParams.get('max_date') || '');
  const [sortBy, setSortBy] = useState('published_at');
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc');
  const [page, setPage] = useState(Number(urlParams.get('page')) || 1);
  const [pageSize, setPageSize] = useState(24);

  const [articles, setArticles] = useState<Article[]>([]);
  const [totalRows, setTotalRows] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [availableSources, setAvailableSources] = useState<string[]>([]);
  const [availableCountries, setAvailableCountries] = useState<string[]>([]);
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null);

  // Load available filter options from backend
  useEffect(() => {
    apiService.getSources().then((res: SourcesResponse) => {
      const srcList = res.sources.map((s) => s.code);
      setAvailableSources(Array.from(new Set<string>(srcList)).sort());
    }).catch(() => {});

    apiService.getCountries().then((res: CountryNewsCount[]) => {
      setAvailableCountries(res.map((c) => c.country).filter(Boolean));
    }).catch(() => {});
  }, []);

  // Sync state if URL query changes
  useEffect(() => {
    const qParam = urlParams.get('q');
    if (qParam !== null && qParam !== query) {
      setQuery(qParam);
    }
  }, [urlParams]);

  const fetchArticles = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params: SearchParams = {
        q: query.trim() || undefined,
        source: selectedSource !== 'all' ? selectedSource : undefined,
        country: selectedCountry !== 'all' ? selectedCountry : undefined,
        min_date: minDate || undefined,
        max_date: maxDate || undefined,
        sort_by: sortBy,
        sort_dir: sortDir,
        page,
        page_size: pageSize,
      };

      const res = query.trim()
        ? await apiService.searchText(params)
        : await apiService.getArticles(params);

      setArticles(res.results || []);
      setTotalRows(res.total_rows || res.count || 0);
    } catch (err: any) {
      console.error('Fetch articles error:', err);
      setError(err.message || 'حدث خطأ أثناء جلب الأخبار من الخادم');
    } finally {
      setLoading(false);
    }
  }, [query, selectedSource, selectedCountry, minDate, maxDate, sortBy, sortDir, page, pageSize]);

  useEffect(() => {
    fetchArticles();
  }, [fetchArticles]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchArticles();
  };

  const handleResetFilters = () => {
    setQuery('');
    setSelectedSource('all');
    setSelectedCountry('all');
    setMinDate('');
    setMaxDate('');
    setSortBy('published_at');
    setSortDir('desc');
    setPage(1);
    setUrlParams({});
  };

  const totalPages = Math.ceil(totalRows / pageSize) || 1;

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Page Title & Stats */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-white/5">
        <div>
          <h1 className="text-2xl font-black text-white royal-gradient-text">
            مستكشف الأخبار العالمية (News Explorer)
          </h1>
          <p className="text-xs text-gray-400 mt-1">
            البحث والاستكشاف المباشر لكافة سجلات الأخبار مع إمكانية التصفية المتقدمة وترتيب النتائج
          </p>
        </div>
        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="text-gray-400">إجمالي السجلات:</span>
          <span className="px-3 py-1 rounded-lg bg-[#4cd7f6]/15 text-[#acedff] font-bold border border-[#4cd7f6]/30">
            {totalRows.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Filter and Search Control Bar */}
      <div className="sovereign-card rounded-2xl p-5 border border-[#4cd7f6]/15 space-y-4">
        {/* Search Row */}
        <form onSubmit={handleSearchSubmit} className="flex gap-3">
          <div className="flex-1 relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="ابحث بالكلمات المفتاحية في العنوان، الملخص، أو المحتوى..."
              className="w-full bg-[#0a0c14] border border-[#4cd7f6]/20 focus:border-[#4cd7f6]/60 py-3 pr-11 pl-4 rounded-xl text-sm text-white placeholder-gray-500 outline-none transition-all shadow-inner"
            />
            <Search className="w-4 h-4 text-gray-500 absolute top-1/2 -translate-y-1/2 right-4 pointer-events-none" />
            {query && (
              <button
                type="button"
                onClick={() => setQuery('')}
                className="absolute top-1/2 -translate-y-1/2 left-3 text-gray-500 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          <button
            type="submit"
            className="px-6 py-3 rounded-xl bg-[#4cd7f6] hover:bg-[#dfc377] text-black font-bold text-xs shrink-0 shadow-lg transition-all"
          >
            بحث
          </button>
        </form>

        {/* Filters Multi-bar */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3 pt-2">
          {/* Source Filter */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-mono text-gray-400 flex items-center gap-1">
              <Building2 className="w-3 h-3 text-[#4cd7f6]" />
              <span>المصدر</span>
            </label>
            <select
              value={selectedSource}
              onChange={(e) => {
                setSelectedSource(e.target.value);
                setPage(1);
              }}
              className="bg-[#0a0c14] border border-white/10 rounded-xl px-3 py-2 text-xs text-gray-200 outline-none focus:border-[#4cd7f6]/40"
            >
              <option value="all">كافة المصادر</option>
              {availableSources.map((src) => (
                <option key={src} value={src}>
                  {src.toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Country Filter */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-mono text-gray-400 flex items-center gap-1">
              <Globe className="w-3 h-3 text-[#4cd7f6]" />
              <span>بلد المصدر</span>
            </label>
            <select
              value={selectedCountry}
              onChange={(e) => {
                setSelectedCountry(e.target.value);
                setPage(1);
              }}
              className="bg-[#0a0c14] border border-white/10 rounded-xl px-3 py-2 text-xs text-gray-200 outline-none focus:border-[#4cd7f6]/40"
            >
              <option value="all">كافة الدول</option>
              {availableCountries.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          {/* Date Range: Min Date */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-mono text-gray-400 flex items-center gap-1">
              <Calendar className="w-3 h-3 text-[#4cd7f6]" />
              <span>من تاريخ</span>
            </label>
            <input
              type="date"
              value={minDate}
              onChange={(e) => {
                setMinDate(e.target.value);
                setPage(1);
              }}
              className="bg-[#0a0c14] border border-white/10 rounded-xl px-3 py-1.5 text-xs text-gray-200 outline-none focus:border-[#4cd7f6]/40"
            />
          </div>

          {/* Date Range: Max Date */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-mono text-gray-400 flex items-center gap-1">
              <Calendar className="w-3 h-3 text-[#4cd7f6]" />
              <span>إلى تاريخ</span>
            </label>
            <input
              type="date"
              value={maxDate}
              onChange={(e) => {
                setMaxDate(e.target.value);
                setPage(1);
              }}
              className="bg-[#0a0c14] border border-white/10 rounded-xl px-3 py-1.5 text-xs text-gray-200 outline-none focus:border-[#4cd7f6]/40"
            />
          </div>

          {/* Sort By / Sort Dir */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-mono text-gray-400 flex items-center gap-1">
              <ArrowUpDown className="w-3 h-3 text-[#4cd7f6]" />
              <span>الترتيب</span>
            </label>
            <div className="flex gap-1.5">
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="flex-1 bg-[#0a0c14] border border-white/10 rounded-xl px-2 py-2 text-xs text-gray-200 outline-none"
              >
                <option value="published_at">التاريخ</option>
                <option value="source">المصدر</option>
                <option value="country">الدولة</option>
              </select>
              <button
                type="button"
                onClick={() => setSortDir(sortDir === 'desc' ? 'asc' : 'desc')}
                className="px-2.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-xs font-mono text-gray-300 border border-white/10"
                title={sortDir === 'desc' ? 'تنازلي (الأحدث أولاً)' : 'تصاعدي (الأقدم أولاً)'}
              >
                {sortDir === 'desc' ? '↓' : '↑'}
              </button>
            </div>
          </div>
        </div>

        {/* Reset Filter Button */}
        <div className="flex justify-end pt-2">
          <button
            onClick={handleResetFilters}
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-[#acedff] transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>إعادة تعيين الفلاتر</span>
          </button>
        </div>
      </div>

      {error && <ErrorBanner message={error} onRetry={fetchArticles} />}

      {/* Articles Grid or Loading/Empty States */}
      {loading ? (
        <NewsGridSkeleton count={pageSize > 12 ? 12 : pageSize} />
      ) : articles.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {articles.map((article, idx) => (
            <NewsCard
              key={`${article.url}-${idx}`}
              article={article}
              onSelect={(a) => setSelectedArticle(a)}
            />
          ))}
        </div>
      ) : (
        <EmptyState onReset={handleResetFilters} />
      )}

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="sovereign-card rounded-2xl p-4 border border-white/5 flex items-center justify-between">
          <span className="text-xs text-gray-400 font-mono">
            الصفحة {page} من {totalPages} ({totalRows} نتيجة)
          </span>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs text-gray-300 disabled:opacity-30 disabled:pointer-events-none transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
              <span>السابق</span>
            </button>

            <span className="px-3 py-1 text-xs font-mono font-bold text-[#acedff] bg-[#4cd7f6]/15 rounded-lg border border-[#4cd7f6]/30">
              {page}
            </span>

            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs text-gray-300 disabled:opacity-30 disabled:pointer-events-none transition-colors"
            >
              <span>التالي</span>
              <ChevronLeft className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Article Detail Modal */}
      <ArticleDetailModal
        article={selectedArticle}
        onClose={() => setSelectedArticle(null)}
      />
    </div>
  );
};
