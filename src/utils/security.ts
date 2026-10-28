// src/utils/security.ts
/**
 * MetaView 3 — Frontend Security & Sanitization Utilities
 *
 * Protects against:
 * 1. DOM XSS via malicious URL schemes (`javascript:`, `data:`, `vbscript:`, `file:`) in href/src attributes.
 * 2. Reverse Tabnabbing (`window.opener`) when opening external source links.
 * 3. CSV / Spreadsheet Formula Injection (CWE-1236) when exporting intelligence tables.
 */

const SAFE_PROTOCOLS = new Set(['http:', 'https:']);

/**
 * Validates that an external link uses strictly `http:` or `https:`.
 * Returns '#' if the URL is empty, malformed, or uses an unsafe scheme (e.g. `javascript:`).
 */
export function sanitizeExternalUrl(url: string | null | undefined): string {
  if (!url || typeof url !== 'string') return '#';
  const trimmed = url.trim();
  if (!trimmed) return '#';

  try {
    const parsed = new URL(trimmed, window.location.origin);
    if (!SAFE_PROTOCOLS.has(parsed.protocol.toLowerCase())) {
      return '#';
    }
    return parsed.href;
  } catch {
    return '#';
  }
}

/**
 * Validates image URLs for `<img src="...">` rendering.
 * Allows `https:`, `http:`, or safe local relative asset paths (`/assets/...`), blocking `javascript:` or HTML `data:` URIs.
 */
export function sanitizeImageUrl(url: string | null | undefined): string | null {
  if (!url || typeof url !== 'string') return null;
  const trimmed = url.trim();
  if (!trimmed) return null;

  if (trimmed.startsWith('/') && !trimmed.startsWith('//')) {
    return trimmed;
  }

  try {
    const parsed = new URL(trimmed);
    if (!SAFE_PROTOCOLS.has(parsed.protocol.toLowerCase())) {
      return null;
    }
    return parsed.href;
  } catch {
    return null;
  }
}

/**
 * Safely opens an external URL in a new tab with `noopener,noreferrer` and explicit opener nulling.
 */
export function safeOpenExternal(url: string | null | undefined): void {
  const safeUrl = sanitizeExternalUrl(url);
  if (safeUrl === '#') return;
  const newWin = window.open(safeUrl, '_blank', 'noopener,noreferrer');
  if (newWin) {
    newWin.opener = null;
  }
}

/**
 * Neutralizes CSV / Excel Formula Injection (DDE / CWE-1236) by prefixing dangerous leading characters
 * (`=`, `+`, `-`, `@`, `\t`, `\r`) with a single quote (`'`), and escaping double quotes.
 */
export function sanitizeCsvCell(value: string | number | null | undefined): string {
  const raw = value === null || value === undefined ? '' : String(value);
  const escaped = raw.replace(/"/g, '""');
  if (/^[=+\-@\t\r]/.test(escaped)) {
    return `"'${escaped}"`;
  }
  return `"${escaped}"`;
}

