// src/utils/exportCsv.ts
import { Article } from '../types';
import { sanitizeCsvCell } from './security';

export function exportArticlesToCsv(articles: Article[], filename: string = 'metaview_intel_export.csv') {
  if (!articles || articles.length === 0) {
    alert('لا توجد بيانات متاحة للتصدير');
    return;
  }

  const headers = ['المصدر', 'الدولة', 'النطاق', 'العنوان', 'الملخص', 'تاريخ النشر', 'رابط الخبر', 'رابط الصورة'];

  const rows = articles.map(a => [
    sanitizeCsvCell(a.source),
    sanitizeCsvCell(a.country),
    sanitizeCsvCell(a.domain),
    sanitizeCsvCell(a.headline),
    sanitizeCsvCell(a.article_summary),
    sanitizeCsvCell(a.published_at),
    sanitizeCsvCell(a.url),
    sanitizeCsvCell(a.image_url),
  ]);

  // UTF-8 BOM (\uFEFF) ensures Excel opens Arabic correctly
  const csvContent = '\uFEFF' + [headers.join(','), ...rows.map(r => r.join(','))].join('\r\n');

  const safeFilename = filename.replace(/[^a-zA-Z0-9._-]/g, '_') || 'metaview_intel_export.csv';
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', safeFilename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
