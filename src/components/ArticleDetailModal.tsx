// src/components/ArticleDetailModal.tsx
import React, { useEffect } from 'react';
import { X, ExternalLink, Copy, Check, Calendar, Globe, Building2, Code2, ShieldAlert } from 'lucide-react';
import { Article } from '../types';
import { formatDate } from '../utils/formatDate';

interface ArticleDetailModalProps {
  article: Article | null;
  onClose: () => void;
}

export const ArticleDetailModal: React.FC<ArticleDetailModalProps> = ({ article, onClose }) => {
  const [copied, setCopied] = React.useState(false);
  const [showRaw, setShowRaw] = React.useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!article) return null;

  const handleCopy = () => {
    if (article.url) {
      navigator.clipboard.writeText(article.url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <div 
        className="relative w-full max-w-3xl max-h-[90vh] bg-[#0c0e16] border border-[#c5a059]/30 rounded-3xl overflow-hidden shadow-2xl flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header Bar */}
        <div className="px-6 py-4 border-b border-[#c5a059]/15 flex items-center justify-between bg-[#08090f]">
          <div className="flex items-center gap-3">
            <span className="px-3 py-1 rounded-lg text-xs font-bold uppercase tracking-wider bg-[#c5a059]/15 text-[#e0c58a] border border-[#c5a059]/30">
              {article.source}
            </span>
            <span className="text-xs text-gray-400 font-mono">
              {formatDate(article.published_at)}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable Body */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Article Image if available */}
          {article.image_url && (
            <div className="w-full h-72 rounded-2xl overflow-hidden border border-[#c5a059]/20 bg-black">
              <img
                src={article.image_url}
                alt={article.headline}
                className="w-full h-full object-cover"
              />
            </div>
          )}

          {/* Headline */}
          <h2 className="text-xl md:text-2xl font-bold text-white leading-relaxed">
            {article.headline}
          </h2>

          {/* Metadata Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 rounded-2xl bg-[#121520] border border-white/5 text-xs">
            <div>
              <span className="text-gray-500 block mb-1">المصدر الإخباري</span>
              <span className="text-white font-semibold flex items-center gap-1">
                <Building2 className="w-3.5 h-3.5 text-[#c5a059]" />
                {article.source}
              </span>
            </div>
            <div>
              <span className="text-gray-500 block mb-1">بلد المصدر (Publisher)</span>
              <span className="text-white font-semibold flex items-center gap-1">
                <Globe className="w-3.5 h-3.5 text-[#c5a059]" />
                {article.country || 'Global'}
              </span>
            </div>
            <div>
              <span className="text-gray-500 block mb-1">النطاق (Domain)</span>
              <span className="text-white font-semibold font-mono truncate block">
                {article.domain || '—'}
              </span>
            </div>
            <div>
              <span className="text-gray-500 block mb-1">تاريخ النشر</span>
              <span className="text-white font-semibold flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5 text-[#c5a059]" />
                {formatDate(article.published_at)}
              </span>
            </div>
          </div>

          {/* Article Summary */}
          <div>
            <h4 className="text-xs uppercase font-mono tracking-widest text-[#c5a059] mb-2">
              ملخص المحتوى (Article Summary)
            </h4>
            <div className="p-4 rounded-2xl bg-[#090a10] border border-white/5 text-sm text-gray-200 leading-relaxed">
              {article.article_summary || 'لا يوجد ملخص متاح لهذا الخبر.'}
            </div>
          </div>

          {/* Content if returned by API */}
          {article.content && (
            <div>
              <h4 className="text-xs uppercase font-mono tracking-widest text-[#c5a059] mb-2">
                النص الكامل (Content)
              </h4>
              <div className="p-4 rounded-2xl bg-[#090a10] border border-white/5 text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                {article.content}
              </div>
            </div>
          )}

          {/* Toggle Raw JSON Inspector */}
          <div>
            <button
              onClick={() => setShowRaw(!showRaw)}
              className="flex items-center gap-2 text-xs font-mono text-gray-400 hover:text-[#e0c58a] transition-colors"
            >
              <Code2 className="w-4 h-4 text-[#c5a059]" />
              <span>{showRaw ? 'إخفاء السجل الخام (Hide Raw Record)' : 'معاينة سجل API الخام (View Raw Record)'}</span>
            </button>
            {showRaw && (
              <pre className="mt-3 p-4 rounded-xl bg-black border border-white/10 text-[11px] font-mono text-emerald-400 overflow-x-auto">
                {JSON.stringify(article, null, 2)}
              </pre>
            )}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-4 border-t border-[#c5a059]/15 flex items-center justify-between bg-[#08090f]">
          <button
            onClick={handleCopy}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-xs font-semibold text-gray-300 transition-colors"
          >
            {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            <span>{copied ? 'تم النسخ!' : 'نسخ رابط المقال'}</span>
          </button>

          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#c5a059] to-[#dfc377] text-black font-bold text-xs shadow-lg hover:brightness-110 transition-all"
          >
            <span>فتح المقال الأصلي في المصدر</span>
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </div>
    </div>
  );
};
