// src/components/EmptyState.tsx
import React from 'react';
import { SearchX, RefreshCcw } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  description?: string;
  onReset?: () => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'لم يتم العثور على نتائج',
  description = 'لا توجد مقالات تطابق معايير البحث أو الفلترة المحددة. جرب توسيع نطاق البحث أو تصفير الفلاتر.',
  onReset,
}) => {
  return (
    <div className="sovereign-card rounded-3xl p-12 text-center flex flex-col items-center justify-center border border-[#4cd7f6]/15 my-8">
      <div className="w-16 h-16 rounded-2xl bg-[#4cd7f6]/10 border border-[#4cd7f6]/20 flex items-center justify-center mb-4 text-[#4cd7f6]">
        <SearchX className="w-8 h-8" />
      </div>
      <h3 className="text-lg font-bold text-white mb-2">{title}</h3>
      <p className="text-sm text-gray-400 max-w-md mx-auto mb-6 leading-relaxed">
        {description}
      </p>
      {onReset && (
        <button
          onClick={onReset}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#4cd7f6]/15 hover:bg-[#4cd7f6]/25 text-[#acedff] font-semibold text-xs border border-[#4cd7f6]/30 transition-all"
        >
          <RefreshCcw className="w-4 h-4" />
          <span>إعادة تعيين الفلاتر</span>
        </button>
      )}
    </div>
  );
};
