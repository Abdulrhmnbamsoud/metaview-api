// src/components/NewsCard.tsx
import React, { useState } from 'react';
import { ExternalLink, Copy, Check, Clock, Globe, Newspaper } from 'lucide-react';
import { Article } from '../types';
import { timeAgo } from '../utils/formatDate';

interface NewsCardProps {
  article: Article;
  onSelect?: (article: Article) => void;
}

export const NewsCard: React.FC<NewsCardProps> = ({ article, onSelect }) => {
  const [copied, setCopied] = useState(false);
  const [imgError, setImgError] = useState(false);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (article.url) {
      navigator.clipboard.writeText(article.url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleOpenOriginal = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (article.url) {
      window.open(article.url, '_blank', 'noopener,noreferrer');
    }
  };

  const hasImage = Boolean(article.image_url) && !imgError;

  return (
    <div
      onClick={() => onSelect && onSelect(article)}
      className="sovereign-card rounded-2xl overflow-hidden flex flex-col group cursor-pointer border border-[#4cd7f6]/15 hover:border-[#4cd7f6]/40 transition-all duration-300"
    >
      {/* Image or High-End Intelligence Placeholder */}
      <div className="relative h-48 w-full bg-[#0a0c12] overflow-hidden border-b border-[#4cd7f6]/10">
        {hasImage ? (
          <img
            src={article.image_url || ''}
            alt={article.headline}
            onError={() => setImgError(true)}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center relative p-6 bg-gradient-to-br from-[#0F172A] to-[#06070a]">
            {/* Visual tech grid / pattern */}
            <div className="absolute inset-0 opacity-10 bg-[radial-gradient(#4cd7f6_1px,transparent_1px)] [background-size:16px_16px]" />
            <div className="w-14 h-14 rounded-2xl bg-[#4cd7f6]/10 border border-[#4cd7f6]/20 flex items-center justify-center mb-2 shadow-inner">
              <Newspaper className="w-7 h-7 text-[#4cd7f6]" />
            </div>
            <span className="text-[11px] font-mono tracking-widest text-[#4cd7f6]/60 uppercase">
              {article.source || 'INTEL CORE'}
            </span>
            <span className="text-[9px] text-gray-500 mt-0.5">METAVIEW MONITORING</span>
          </div>
        )}

        {/* Source Badge Overlay */}
        <div className="absolute top-3 right-3 flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-lg text-xs font-bold uppercase tracking-wider bg-black/80 backdrop-blur-md text-[#acedff] border border-[#4cd7f6]/30 shadow-lg">
            {article.source}
          </span>
          {article.country && (
            <span className="px-2 py-0.5 rounded-md text-[11px] font-mono bg-black/70 backdrop-blur-md text-gray-300 border border-white/10">
              {article.country}
            </span>
          )}
        </div>

        {/* Time Ago pill */}
        <div className="absolute bottom-3 left-3 flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] bg-black/80 backdrop-blur-md text-gray-400 border border-white/5">
          <Clock className="w-3 h-3 text-[#4cd7f6]" />
          <span>{timeAgo(article.published_at)}</span>
        </div>
      </div>

      {/* Content Area */}
      <div className="p-5 flex-1 flex flex-col justify-between">
        <div>
          <h3 className="text-base font-bold text-white group-hover:text-[#acedff] transition-colors line-clamp-2 leading-snug mb-2.5">
            {article.headline}
          </h3>
          <p className="text-xs text-gray-400 line-clamp-3 leading-relaxed mb-4">
            {article.article_summary || 'لا يوجد ملخص متاح لهذا الخبر في المصدر الأصلي.'}
          </p>
        </div>

        {/* Action Footer */}
        <div className="pt-3 border-t border-white/5 flex items-center justify-between mt-auto">
          <div className="flex items-center gap-1 text-[11px] text-gray-500 font-mono">
            <Globe className="w-3 h-3 text-[#4cd7f6]/70" />
            <span className="truncate max-w-[120px]">{article.domain || article.source}</span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              title="نسخ رابط الخبر"
              className="p-1.5 rounded-lg bg-white/5 hover:bg-[#4cd7f6]/20 text-gray-400 hover:text-[#acedff] transition-all"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
            <button
              onClick={handleOpenOriginal}
              title="قراءة المقال الأصلي"
              className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-[#4cd7f6]/10 hover:bg-[#4cd7f6]/25 text-[#acedff] text-xs font-semibold border border-[#4cd7f6]/20 transition-all"
            >
              <span>المصدر</span>
              <ExternalLink className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
