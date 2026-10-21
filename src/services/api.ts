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
      return await requestWithRetry<StatsSummary>('/stats');
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
      return await requestWithRetry<CountryNewsCount[]>('/countries');
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

  /**
   * Get the active base URL for transparency in Operations Center
   */
  getBaseUrl(): string {
    return API_BASE_URL;
  },
};
