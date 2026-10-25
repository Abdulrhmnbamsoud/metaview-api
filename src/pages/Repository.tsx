// src/pages/Repository.tsx
import React, { useState, useEffect, useCallback } from 'react';
import {
  Database,
  Search,
  Download,
  ExternalLink,
  ArrowUpDown,
  Eye,
  Calendar,
  Building2,
  Globe,
  ChevronLeft,
  ChevronRight,
  Filter,
} from 'lucide-react';
import { apiService } from '../services/api';
import { Article, SearchParams } from '../types';
import { ArticleDetailModal } from '../components/ArticleDetailModal';
import { formatDate } from '../utils/formatDate';
import { exportArticlesToCsv } from '../utils/exportCsv';

export const Repository: React.FC = () => {
  const [articles, setArticles] = useState<Article[]>([]);
  const [totalRows, setTotalRows] = useState(0);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState('');
  const [selectedSource, setSelectedSource] = useState('all');
  const [sortBy, setSortBy] = useState('published_at');
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null);

  const fetchRepoArticles = useCallback(async () => {
    setLoading(true);
    try {
      const params: SearchParams = {
        q: query.trim() || undefined,
        source: selectedSource !== 'all' ? selectedSource : undefined,
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
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [query, selectedSource, sortBy, sortDir, page, pageSize]);

  useEffect(() => {
    fetchRepoArticles();
  }, [fetchRepoArticles]);

  const handleExport = () => {
    exportArticlesToCsv(articles, `metaview_articles_page_${page}.csv`);
  };

  const totalPages = Math.ceil(totalRows / pageSize) || 1;

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Page Title & Export Actions */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-white/5">
        <div>
          <h1 className="text-2xl font-black text-white royal-gradient-text">
            مستودع الأخبار الموحد (Unified MetaView Repository)
          </h1>
          <p className="text-xs text-gray-400 mt-1">
            سجل بيانات منظّم يتيح الفرز، الفحص المعمق للسجلات، والتصدير المباشر بصيغة CSV
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-[#4cd7f6]/15 hover:bg-[#4cd7f6]/25 text-[#acedff] font-semibold text-xs border border-[#4cd7f6]/30 transition-all shadow-md"
            title="تصدير السجلات المعروضة إلى ملف CSV يدعم اللغة العربية"
          >
            <Download className="w-4 h-4" />
            <span>تصدير إلى CSV ({articles.length} سجل)</span>
          </button>
        </div>
      </div>

      {/* Control Bar: Search and Quick Filter */}
      <div className="sovereign-card rounded-2xl p-4 border border-[#4cd7f6]/15 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="relative w-full md:w-96">
          <input
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(1);
            }}
            placeholder="البحث الفوري في المستودع..."
            className="w-full bg-[#0a0c14] border border-[#4cd7f6]/20 focus:border-[#4cd7f6]/60 py-2 pr-10 pl-4 rounded-xl text-xs text-white placeholder-gray-500 outline-none"
          />
          <Search className="w-4 h-4 text-gray-500 absolute top-1/2 -translate-y-1/2 right-3 pointer-events-none" />
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto">
          <div className="flex items-center gap-2 text-xs font-mono text-gray-400">
            <span>عدد النتائج:</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setPage(1);
              }}
              className="bg-[#0a0c14] border border-white/10 rounded-lg px-2.5 py-1 text-xs text-gray-200 outline-none"
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>

          <span className="text-xs font-mono text-gray-400">
            إجمالي السجلات: <strong className="text-white">{totalRows.toLocaleString()}</strong>
          </span>
        </div>
      </div>

      {/* High-Density Data Grid */}
      <div className="sovereign-card rounded-2xl border border-white/10 overflow-hidden shadow-2xl">
        <div className="overflow-x-auto">
          <table className="w-full text-right text-xs">
            <thead className="bg-[#0e101a] text-gray-400 font-mono text-[11px] border-b border-white/5 uppercase">
              <tr>
                <th className="p-4 w-12 text-center">#</th>
                <th className="p-4 w-16">الصورة</th>
                <th
                  onClick={() => {
                    setSortBy('published_at');
                    setSortDir(sortDir === 'desc' ? 'asc' : 'desc');
                  }}
                  className="p-4 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    <span>التاريخ</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => {
                    setSortBy('source');
                    setSortDir(sortDir === 'desc' ? 'asc' : 'desc');
                  }}
                  className="p-4 cursor-pointer hover:text-white"
                >
                  <div className="flex items-center gap-1">
                    <span>المصدر</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="p-4">الدولة</th>
                <th className="p-4 min-w-[280px]">عنوان الخبر</th>
                <th className="p-4">النطاق</th>
                <th className="p-4 text-center">إجراءات</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {loading ? (
                Array.from({ length: 8 }).map((_, i) => (
                  <tr key={i} className="animate-pulse">
                    <td colSpan={8} className="p-4 bg-white/[0.02]">
                      <div className="h-4 bg-white/5 rounded w-full" />
                    </td>
                  </tr>
                ))
              ) : articles.length > 0 ? (
                articles.map((art, idx) => (
                  <tr
                    key={`${art.url}-${idx}`}
                    onClick={() => setSelectedArticle(art)}
                    className="hover:bg-[#4cd7f6]/5 cursor-pointer transition-colors"
                  >
                    <td className="p-4 text-center font-mono text-gray-500">
                      {(page - 1) * pageSize + idx + 1}
                    </td>
                    <td className="p-4">
                      {art.image_url ? (
                        <img
                          src={art.image_url}
                          alt=""
                          className="w-10 h-8 rounded object-cover border border-white/10"
                        />
                      ) : (
                        <div className="w-10 h-8 rounded bg-white/5 flex items-center justify-center text-[9px] font-mono text-gray-600">
                          N/A
                        </div>
                      )}
                    </td>
                    <td className="p-4 font-mono text-gray-400 whitespace-nowrap">
                      {art.published_at ? art.published_at.slice(0, 16).replace('T', ' ') : '—'}
                    </td>
                    <td className="p-4">
                      <span className="px-2 py-0.5 rounded font-mono text-[10px] bg-white/5 border border-white/10 text-[#acedff] font-bold uppercase">
                        {art.source}
                      </span>
                    </td>
                    <td className="p-4 font-mono text-gray-400">
                      {art.country || 'Global'}
                    </td>
                    <td className="p-4 font-medium text-white max-w-md truncate">
                      {art.headline}
                    </td>
                    <td className="p-4 font-mono text-gray-500 text-[11px] truncate max-w-[120px]">
                      {art.domain || '—'}
                    </td>
                    <td className="p-4" onClick={(e) => e.stopPropagation()}>
                      <div className="flex items-center justify-center gap-2">
                        <button
                          onClick={() => setSelectedArticle(art)}
                          className="p-1.5 rounded-lg bg-white/5 hover:bg-[#4cd7f6]/20 text-gray-400 hover:text-white"
                          title="فحص السجل"
                        >
                          <Eye className="w-3.5 h-3.5" />
                        </button>
                        <a
                          href={art.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="p-1.5 rounded-lg bg-white/5 hover:bg-[#4cd7f6]/20 text-[#4cd7f6] hover:text-[#acedff]"
                          title="فتح في المصدر"
                        >
                          <ExternalLink className="w-3.5 h-3.5" />
                        </a>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} className="p-12 text-center text-gray-400">
                    لم يتم العثور على سجلات في المستودع
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        {totalPages > 1 && (
          <div className="p-4 bg-[#0a0c14] border-t border-white/5 flex items-center justify-between">
            <span className="text-xs text-gray-400 font-mono">
              صفحة {page} من {totalPages}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs text-gray-300 disabled:opacity-30 disabled:pointer-events-none"
              >
                السابق
              </button>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs text-gray-300 disabled:opacity-30 disabled:pointer-events-none"
              >
                التالي
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Article Detail Modal */}
      <ArticleDetailModal
        article={selectedArticle}
        onClose={() => setSelectedArticle(null)}
      />
    </div>
  );
};
