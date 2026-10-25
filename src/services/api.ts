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
import {
  User,
  UserRole,
  UserStatus,
  Invitation,
  AuditLog,
  AnalystNote,
  SavedIncidentItem,
  CustomPattern,
  CustomRule,
  ReviewRequest,
  WorkspaceOverview,
  DetectionRuleTestResult,
} from '../types/auth';

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

let _sessionToken: string | null = null;

export const setSessionToken = (token: string | null) => {
  _sessionToken = token;
  if (token) {
    sessionStorage.setItem('metaview_session_token', token);
  } else {
    sessionStorage.removeItem('metaview_session_token');
  }
};

export const getSessionToken = (): string | null => {
  if (!_sessionToken) {
    _sessionToken = sessionStorage.getItem('metaview_session_token');
  }
  return _sessionToken;
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
      const token = getSessionToken();
      const authHeader: Record<string, string> = {};
      if (token) {
        authHeader['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch(url, {
        ...fetchOptions,
        signal: controller.signal,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          ...authHeader,
          ...(fetchOptions.headers || {}),
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorMsg = `HTTP error ${response.status}: ${response.statusText}`;
        try {
          const errJson = await response.json();
          if (errJson?.detail) {
            errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
          }
        } catch {
          // non-json response
        }
        const err: any = new Error(errorMsg);
        err.status = response.status;
        throw err;
      }

      return (await response.json()) as T;
    } catch (err: any) {
      clearTimeout(timeoutId);
      lastError = err;

      // Don't retry client 4xx errors
      if (err.status && err.status >= 400 && err.status < 500) {
        throw err;
      }

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

  // --------------------------------------------------------------------------- //
  // Authentication & Session Endpoints
  // --------------------------------------------------------------------------- //
  async login(payload: { email: string; password: string }): Promise<{ token: string; user: User; message: string }> {
    const res = await requestWithRetry<{ token: string; user: User; message: string }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    setSessionToken(res.token);
    return res;
  },

  async logout(): Promise<{ message: string }> {
    try {
      const res = await requestWithRetry<{ message: string }>('/auth/logout', {
        method: 'POST',
      });
      return res;
    } finally {
      setSessionToken(null);
    }
  },

  async getMe(): Promise<User> {
    return await requestWithRetry<User>('/auth/me');
  },

  async acceptInvitation(payload: { token: string; full_name: string; password: string }): Promise<{ token: string; user: User; message: string }> {
    const res = await requestWithRetry<{ token: string; user: User; message: string }>('/auth/invitations/accept', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    setSessionToken(res.token);
    return res;
  },

  // --------------------------------------------------------------------------- //
  // Team Management (Admin Only)
  // --------------------------------------------------------------------------- //
  async getTeamMembers(): Promise<{ members: User[]; total: number }> {
    return await requestWithRetry<{ members: User[]; total: number }>('/team/members');
  },

  async inviteMember(payload: { email: string; role: UserRole }): Promise<{
    invitation_id: number;
    email: string;
    role: UserRole;
    expires_at: string;
    invite_token: string;
    invite_url: string;
    message: string;
  }> {
    return await requestWithRetry('/team/invite', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async getTeamInvitations(): Promise<{ invitations: Invitation[]; total: number }> {
    return await requestWithRetry<{ invitations: Invitation[]; total: number }>('/team/invitations');
  },

  async revokeInvitation(invitationId: number): Promise<{ message: string }> {
    return await requestWithRetry<{ message: string }>(`/team/invitations/${invitationId}`, {
      method: 'DELETE',
    });
  },

  async updateMemberRole(userId: number, role: UserRole): Promise<{ message: string; user: User }> {
    return await requestWithRetry<{ message: string; user: User }>(`/team/members/${userId}/role`, {
      method: 'POST',
      body: JSON.stringify({ role }),
    });
  },

  async updateMemberStatus(userId: number, status: UserStatus): Promise<{ message: string; user: User }> {
    return await requestWithRetry<{ message: string; user: User }>(`/team/members/${userId}/status`, {
      method: 'POST',
      body: JSON.stringify({ status }),
    });
  },

  async getTeamAuditLogs(limit: number = 50): Promise<{ logs: AuditLog[]; total: number }> {
    return await requestWithRetry<{ logs: AuditLog[]; total: number }>(`/team/audit-logs?limit=${limit}`);
  },

  async getRolesMatrix(): Promise<Record<string, any>> {
    return await requestWithRetry<Record<string, any>>('/team/roles-matrix');
  },

  // --------------------------------------------------------------------------- //
  // Incident Notes & Collaboration
  // --------------------------------------------------------------------------- //
  async getIncidentNotes(incidentId: number): Promise<{ incident_id: number; notes: AnalystNote[]; total: number }> {
    return await requestWithRetry<{ incident_id: number; notes: AnalystNote[]; total: number }>(`/incidents/${incidentId}/notes`);
  },

  async addIncidentNote(incidentId: number, payload: { note_type: string; content: string }): Promise<AnalystNote> {
    return await requestWithRetry<AnalystNote>(`/incidents/${incidentId}/notes`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async deleteIncidentNote(incidentId: number, noteId: number): Promise<{ message: string }> {
    return await requestWithRetry<{ message: string }>(`/incidents/${incidentId}/notes/${noteId}`, {
      method: 'DELETE',
    });
  },

  // --------------------------------------------------------------------------- //
  // Saved Incidents (Analyst Watchlist)
  // --------------------------------------------------------------------------- //
  async getSavedIncidents(): Promise<{ items: SavedIncidentItem[]; total: number }> {
    return await requestWithRetry<{ items: SavedIncidentItem[]; total: number }>('/workspace/saved-incidents');
  },

  async saveIncident(incidentId: number, notes?: string): Promise<{ message: string; saved_id: number }> {
    return await requestWithRetry<{ message: string; saved_id: number }>('/workspace/saved-incidents', {
      method: 'POST',
      body: JSON.stringify({ incident_id: incidentId, notes: notes || '' }),
    });
  },

  async unsaveIncident(incidentId: number): Promise<{ message: string }> {
    return await requestWithRetry<{ message: string }>(`/workspace/saved-incidents/${incidentId}`, {
      method: 'DELETE',
    });
  },

  // --------------------------------------------------------------------------- //
  // Custom Patterns Lifecycle
  // --------------------------------------------------------------------------- //
  async getWorkspacePatterns(): Promise<{ patterns: CustomPattern[]; total: number }> {
    return await requestWithRetry<{ patterns: CustomPattern[]; total: number }>('/workspace/patterns');
  },

  async createPatternDraft(payload: {
    key: string;
    label_ar: string;
    description?: string;
    modus_operandi?: string;
    channels?: string[];
    target_groups?: string[];
    banking_relevance?: string;
    linked_incident_ids?: number[];
    hypothesis?: string;
    evidence_summary?: string;
  }): Promise<{ id: number; key: string; status: string; message: string }> {
    return await requestWithRetry('/workspace/patterns', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async submitPatternForReview(key: string, notes?: string): Promise<{ key: string; status: string; message: string }> {
    return await requestWithRetry(`/workspace/patterns/${encodeURIComponent(key)}/submit`, {
      method: 'POST',
      body: JSON.stringify({ notes: notes || '' }),
    });
  },

  async decidePatternReview(key: string, decision: 'approved' | 'changes_requested' | 'rejected', reviewNotes: string): Promise<{ key: string; status: string; message: string }> {
    return await requestWithRetry(`/workspace/patterns/${encodeURIComponent(key)}/decide`, {
      method: 'POST',
      body: JSON.stringify({ decision, review_notes: reviewNotes }),
    });
  },

  // --------------------------------------------------------------------------- //
  // Custom Rules Lifecycle
  // --------------------------------------------------------------------------- //
  async getWorkspaceRules(): Promise<{ rules: CustomRule[]; total: number }> {
    return await requestWithRetry<{ rules: CustomRule[]; total: number }>('/workspace/rules');
  },

  async createRuleDraft(payload: {
    rule_id: string;
    name: string;
    hypothesis: string;
    logic: string;
    pattern_key?: string;
    required_data?: string[];
    exceptions?: any;
    false_positive_risk?: string;
  }): Promise<{ id: number; rule_id: string; status: string; message: string }> {
    return await requestWithRetry('/workspace/rules', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async testDetectionRule(ruleId: string): Promise<{ rule_id: string; test_result: DetectionRuleTestResult; message: string }> {
    return await requestWithRetry(`/workspace/rules/${encodeURIComponent(ruleId)}/test`, {
      method: 'POST',
      body: JSON.stringify({}),
    });
  },

  async submitRuleForReview(ruleId: string, notes?: string): Promise<{ rule_id: string; status: string; message: string }> {
    return await requestWithRetry(`/workspace/rules/${encodeURIComponent(ruleId)}/submit`, {
      method: 'POST',
      body: JSON.stringify({ notes: notes || '' }),
    });
  },

  async decideRuleReview(ruleId: string, decision: 'approved' | 'changes_requested' | 'rejected', reviewNotes: string): Promise<{ rule_id: string; status: string; message: string }> {
    return await requestWithRetry(`/workspace/rules/${encodeURIComponent(ruleId)}/decide`, {
      method: 'POST',
      body: JSON.stringify({ decision, review_notes: reviewNotes }),
    });
  },

  // --------------------------------------------------------------------------- //
  // Governance Review Requests & Workspace Overview
  // --------------------------------------------------------------------------- //
  async getPendingReviews(status: string = 'pending'): Promise<{ reviews: ReviewRequest[]; total: number }> {
    return await requestWithRetry<{ reviews: ReviewRequest[]; total: number }>(`/governance/reviews?status=${encodeURIComponent(status)}`);
  },

  async getWorkspaceOverview(): Promise<WorkspaceOverview> {
    return await requestWithRetry<WorkspaceOverview>('/workspace/overview');
  },

  async decideMergeSuggestion(suggestionId: number, decision: 'approved' | 'rejected', reason: string, reviewer?: string): Promise<any> {
    return await requestWithRetry(`/saudi-fraud/incidents/merge-suggestions/${suggestionId}/decision`, {
      method: 'POST',
      body: JSON.stringify({ decision, reason, reviewer: reviewer || '' }),
    });
  },

  /**
   * Get the active base URL for transparency in Operations Center
   */
  getBaseUrl(): string {
    return API_BASE_URL;
  },
};
