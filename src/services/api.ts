// src/services/api.ts
/**
 * MetaView 3 — Centralized API Client
 * Single source of truth connecting to MetaView API.
 * Supports configurable base URL, timeout, retry with exponential backoff, and cancellation.
 */

import {
  Article,
  SearchResponse,
  SearchParams,
  HealthStatus,
  StatsSummary,
  SourcesResponse,
  IngestRunResult,
  CountryNewsCount,
} from '../types';
import {
  SaudiOverview,
  SaudiNewsResponse,
  SaudiNewsParams,
  SaudiFraudItem,
  SaudiSourceInfo,
  SaudiPatternsResponse,
  SaudiRulesResponse,
  SaudiIncident,
  SaudiIncidentsResponse,
  SaudiIncidentReport,
  SaudiRuntime,
  MergeSuggestion,
  CountryDistributionResponse,
} from '../types/saudi';

// Base URL resolution: Environment variable -> Vite Proxy (/api) -> direct localhost
let API_BASE_URL: string;
if (import.meta.env.VITE_API_URL) {
  // Production: use provided URL, strip trailing slash.
  API_BASE_URL = import.meta.env.VITE_API_URL.replace(/\/$/, '');
} else if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
  // Local development – proxy through Vite dev server.
  API_BASE_URL = '/api';
} else {
  // Mis‑configuration: no URL supplied.
  console.error('VITE_API_URL is not defined; API requests will fail.');
  API_BASE_URL = '';
}

const DEFAULT_TIMEOUT_MS = 15000;

// ---- Arabic-only language filter (persisted; applied to every global list/stat request) ----
const ARABIC_ONLY_KEY = 'metaview.arabicOnly';

export const getArabicOnly = (): boolean => {
  try {
    return localStorage.getItem(ARABIC_ONLY_KEY) === '1';
  } catch {
    return false;
  }
};

export const setArabicOnly = (on: boolean): void => {
  try {
    localStorage.setItem(ARABIC_ONLY_KEY, on ? '1' : '0');
  } catch {
    /* storage unavailable: filter simply won't persist */
  }
};

const withLang = (query: URLSearchParams): URLSearchParams => {
  if (getArabicOnly()) query.set('lang', 'ar');
  return query;
};

interface FetchOptions extends RequestInit {
  timeoutMs?: number;
  retries?: number;
}

async function requestWithRetry<T>(endpoint: string, options: FetchOptions = {}): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, retries = 2, ...fetchOptions } = options;
  const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;

  let lastError: any = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        ...fetchOptions,
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          ...(fetchOptions.headers || {}),
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
      }

      return (await response.json()) as T;
    } catch (err: any) {
      clearTimeout(timeoutId);
      lastError = err;

      // Don't retry if aborted explicitly by user
      if (err.name === 'AbortError' && attempt === retries) {
        throw new Error(`Request timed out after ${timeoutMs}ms`);
      }

      if (attempt < retries) {
        // Exponential backoff: 500ms, 1000ms...
        await new Promise((res) => setTimeout(res, 500 * Math.pow(2, attempt)));
      }
    }
  }

  throw lastError || new Error(`Failed to fetch ${endpoint}`);
}

export const apiService = {
  /**
   * Health check of the backend service
   */
  async getHealth(): Promise<HealthStatus> {
    try {
      return await requestWithRetry<HealthStatus>('/health', { timeoutMs: 5000, retries: 1 });
    } catch {
      return {
        status: 'offline',
        service: 'metaview',
        rows: 0,
        time: new Date().toISOString(),
      };
    }
  },

  /**
   * Aggregated intelligence statistics for Dashboard
   */
  async getStats(): Promise<StatsSummary> {
    try {
      return await requestWithRetry<StatsSummary>(`/stats${getArabicOnly() ? '?lang=ar' : ''}`);
    } catch (err) {
      console.warn('API /stats fallback:', err);
      // Fallback calculation via /articles if /stats is unavailable
      const articlesRes = await this.getArticles({ page: 1, page_size: 50 });
      return {
        total_articles: articlesRes.total_rows || articlesRes.count || 0,
        total_sources: 0,
        articles_last_24h: 0,
        last_sync: new Date().toISOString(),
        source_distribution: [],
        country_distribution: [],
      };
    }
  },

  /**
   * Fetch articles with full pagination and filters
   */
  async getArticles(params: SearchParams = {}): Promise<SearchResponse> {
    const query = new URLSearchParams();
    if (params.page) query.append('page', params.page.toString());
    if (params.page_size) query.append('page_size', params.page_size.toString());
    if (params.include_content) query.append('include_content', 'true');
    if (params.source && params.source !== 'all') query.append('source', params.source);
    if (params.country && params.country !== 'all') query.append('country', params.country);
    if (params.domain && params.domain !== 'all') query.append('domain', params.domain);
    if (params.category && params.category !== 'all') query.append('category', params.category);
    if (params.min_date) query.append('min_date', params.min_date);
    if (params.max_date) query.append('max_date', params.max_date);
    if (params.sort_by) query.append('sort_by', params.sort_by);
    if (params.sort_dir) query.append('sort_dir', params.sort_dir);

    withLang(query);
    const qs = query.toString();
    const endpoint = `/articles${qs ? `?${qs}` : ''}`;
    return await requestWithRetry<SearchResponse>(endpoint);
  },

  /**
   * Full-text search with keyword and optional filters
   */
  async searchText(params: SearchParams = {}): Promise<SearchResponse> {
    const query = new URLSearchParams();
    if (params.q) query.append('q', params.q);
    if (params.page) query.append('page', params.page.toString());
    if (params.page_size) query.append('page_size', params.page_size.toString());
    if (params.include_content) query.append('include_content', 'true');
    if (params.source && params.source !== 'all') query.append('source', params.source);
    if (params.country && params.country !== 'all') query.append('country', params.country);
    if (params.min_date) query.append('min_date', params.min_date);
    if (params.max_date) query.append('max_date', params.max_date);
    if (params.sort_by) query.append('sort_by', params.sort_by);
    if (params.sort_dir) query.append('sort_dir', params.sort_dir);

    withLang(query);
    const qs = query.toString();
    const endpoint = `/search-text${qs ? `?${qs}` : ''}`;
    return await requestWithRetry<SearchResponse>(endpoint);
  },

  /**
   * Get supported news sources and live distribution
   */
  async getSources(): Promise<SourcesResponse> {
    return await requestWithRetry<SourcesResponse>('/sources');
  },

  /**
   * Get countries with article counts
   */
  async getCountries(): Promise<CountryNewsCount[]> {
    try {
      return await requestWithRetry<CountryNewsCount[]>(`/countries${getArabicOnly() ? '?lang=ar' : ''}`);
    } catch {
      return [];
    }
  },

  /**
   * Run live RSS ingestion process
   */
  async runIngest(limitPerFeed: number = 25, feeds?: string[], allowWebScrape: boolean = false): Promise<IngestRunResult> {
    return await requestWithRetry<IngestRunResult>('/ingest/run', {
      method: 'POST',
      body: JSON.stringify({
        limit_per_feed: limitPerFeed,
        feeds: feeds || null,
        allow_web_scrape: allowWebScrape,
      }),
      timeoutMs: 60000, // 60 seconds for ingestion
      retries: 0,
    });
  },

  // ---- Saudi Fraud Intelligence ----
  async getSaudiOverview(): Promise<SaudiOverview> {
    return await requestWithRetry<SaudiOverview>('/saudi-fraud/overview');
  },

  async getSaudiNews(params: SaudiNewsParams = {}): Promise<SaudiNewsResponse> {
    const query = new URLSearchParams();
    if (params.category) query.append('category', params.category);
    if (params.source) query.append('source', params.source);
    if (params.q) query.append('q', params.q);
    if (params.bank_relevance) query.append('bank_relevance', params.bank_relevance);
    if (params.content_type) query.append('content_type', params.content_type);
    if (params.scope) query.append('scope', params.scope);
    if (params.country) query.append('country', params.country);
    query.append('page', String(params.page || 1));
    query.append('page_size', String(params.page_size || 20));
    return await requestWithRetry<SaudiNewsResponse>(`/saudi-fraud/news?${query.toString()}`);
  },

  async getFraudCountries(scope: string = 'all'): Promise<CountryDistributionResponse> {
    return await requestWithRetry<CountryDistributionResponse>(`/saudi-fraud/countries?scope=${encodeURIComponent(scope)}`);
  },

  async getSaudiNewsItem(id: number): Promise<SaudiFraudItem> {
    return await requestWithRetry<SaudiFraudItem>(`/saudi-fraud/news/${id}`);
  },

  async getSaudiSources(): Promise<{ sources: SaudiSourceInfo[] }> {
    return await requestWithRetry<{ sources: SaudiSourceInfo[] }>('/saudi-fraud/sources');
  },

  async getSaudiPatterns(): Promise<SaudiPatternsResponse> {
    return await requestWithRetry<SaudiPatternsResponse>('/saudi-fraud/patterns');
  },

  async getSaudiRules(): Promise<SaudiRulesResponse> {
    return await requestWithRetry<SaudiRulesResponse>('/saudi-fraud/rules');
  },

  // ---- Incident Registry (read-only; reviewer decisions are admin-API only) ----
  async getSaudiIncidents(params: {
    category?: string;
    q?: string;
    bank_relevance?: string;
    record_type?: string;
    review_status?: string;
    page?: number;
    page_size?: number;
  } = {}): Promise<SaudiIncidentsResponse> {
    const query = new URLSearchParams();
    (['category', 'q', 'bank_relevance', 'record_type', 'review_status'] as const).forEach((k) => {
      if (params[k]) query.append(k, String(params[k]));
    });
    query.append('page', String(params.page || 1));
    query.append('page_size', String(params.page_size || 20));
    return await requestWithRetry<SaudiIncidentsResponse>(`/saudi-fraud/incidents?${query.toString()}`);
  },

  async getSaudiIncident(id: number): Promise<SaudiIncident> {
    return await requestWithRetry<SaudiIncident>(`/saudi-fraud/incidents/${id}`);
  },

  async getSaudiMergeSuggestions(status: string = 'pending'): Promise<{ status: string; total: number; items: MergeSuggestion[]; note: string }> {
    return await requestWithRetry(`/saudi-fraud/incidents/merge-suggestions?status=${encodeURIComponent(status)}`);
  },

  async getSaudiIncidentReport(): Promise<SaudiIncidentReport> {
    return await requestWithRetry<SaudiIncidentReport>('/saudi-fraud/incidents/report');
  },

  async getSaudiRuntime(): Promise<SaudiRuntime> {
    return await requestWithRetry<SaudiRuntime>('/saudi-fraud/runtime');
  },

  /**
   * Get the active base URL for transparency in Operations Center
   */
  getBaseUrl(): string {
    return API_BASE_URL;
  },
};
