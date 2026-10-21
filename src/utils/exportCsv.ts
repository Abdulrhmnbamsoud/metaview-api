// src/utils/exportCsv.ts
import { Article } from '../types';

export function exportArticlesToCsv(articles: Article[], filename: string = 'metaview_intel_export.csv') {
  if (!articles || articles.length === 0) {
    alert('لا توجد بيانات متاحة للتصدير');
    return;
  }

  const headers = ['المصدر', 'الدولة', 'النطاق', 'العنوان', 'الملخص', 'تاريخ النشر', 'رابط الخبر', 'رابط الصورة'];
  
  const rows = articles.map(a => [
    `"${(a.source || '').replace(/"/g, '""')}"`,
    `"${(a.country || '').replace(/"/g, '""')}"`,
    `"${(a.domain || '').replace(/"/g, '""')}"`,
    `"${(a.headline || '').replace(/"/g, '""')}"`,
    `"${(a.article_summary || '').replace(/"/g, '""')}"`,
    `"${(a.published_at || '').replace(/"/g, '""')}"`,
    `"${(a.url || '').replace(/"/g, '""')}"`,
    `"${(a.image_url || '').replace(/"/g, '""')}"`,
  ]);

  // UTF-8 BOM (\uFEFF) ensures Excel opens Arabic correctly
  const csvContent = '\uFEFF' + [headers.join(','), ...rows.map(r => r.join(','))].join('\r\n');
  
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
