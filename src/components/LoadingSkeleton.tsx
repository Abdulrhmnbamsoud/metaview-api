// src/components/LoadingSkeleton.tsx
import React from 'react';

export const NewsGridSkeleton: React.FC<{ count?: number }> = ({ count = 6 }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {Array.from({ length: count }).map((_, idx) => (
        <div key={idx} className="sovereign-card rounded-2xl overflow-hidden border border-white/5 animate-pulse">
          <div className="h-48 bg-[#121520]" />
          <div className="p-5 space-y-3">
            <div className="h-4 bg-white/10 rounded w-3/4" />
            <div className="h-4 bg-white/10 rounded w-1/2" />
            <div className="space-y-1.5 pt-2">
              <div className="h-3 bg-white/5 rounded" />
              <div className="h-3 bg-white/5 rounded w-5/6" />
            </div>
            <div className="pt-4 flex justify-between">
              <div className="h-3 bg-white/5 rounded w-20" />
              <div className="h-6 bg-white/10 rounded w-16" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
