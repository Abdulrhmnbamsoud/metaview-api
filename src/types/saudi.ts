// src/types/saudi.ts
export interface SaudiCategoryCount {
  key: string;
  label_ar: string;
  count: number;
}

export interface SaudiOverview {
  total_items: number;
  last_7_days: number;
  high_bank_relevance: number;
  distinct_sources: number;
  categories: SaudiCategoryCount[];
  last_fetch_at: string | null;
  sources_ok: number;
  sources_total: number;
  refresh_every_hours: number;
  auto_ingest_enabled: boolean;
  ingest_running: boolean;
  // incident layer
  incidents_total?: number;
  incident_articles_total?: number;
  incidents_confirmed?: number;
  incidents_reported?: number;
  incidents_needs_review?: number;
  merge_suggestions_pending?: number;
  warnings_total?: number;
  awareness_total?: number;
  excluded_general_cybersecurity?: number;
  excluded_unrelated?: number;
  unclassified?: number;
  by_content_type?: Record<string, number>;
}

export interface SaudiDetectionRule {
  rule_id: string;
  status: string;
  is_inference: boolean;
  name: string;
  hypothesis: string;
  logic: string;
  required_data: string[];
  validation: string[];
  derived_from?: { category: string; signals_in_article: string[]; channels: string[] };
}

export interface SaudiFraudItem {
  id: number;
  article_id: number;
  title: string;
  headline: string | null;
  summary: string | null;
  fraud_category: string;
  fraud_category_ar: string;
  event_type: string | null;
  content_type?: string | null;
  content_type_ar?: string | null;
  incident_id?: number | null;
  source: string | null;
  source_code: string | null;
  original_url: string | null;
  published_at: string | null;
  fetched_at: string | null;
  bank_relevance: 'high' | 'medium' | 'low' | null;
  confidence: number | null;
  evidence_confidence: number | null;
  analysis_confidence: number | null;
  modus_operandi: string | null;
  channels: string[];
  target_group: string[];
  detection_rule: SaudiDetectionRule | null;
  // detail only
  verified_facts?: any;
  analytical_assessment?: any;
  detection_indicators?: { mentioned_in_source: string[]; suggested_by_system: string[] } | null;
  preventive_recommendations?: { origin: string; items: string[] } | null;
  required_data?: string[] | null;
  matched_terms?: string[] | null;
  language?: string | null;
  content_excerpt?: string;
}

export interface SaudiNewsResponse {
  page: number;
  page_size: number;
  total: number;
  items: SaudiFraudItem[];
}

export interface SaudiSourceInfo {
  code: string;
  name: string;
  url: string;
  kind: string;
  official: number;
  last_status: string | null;
  last_success_at: string | null;
  last_failure_at: string | null;
  last_error: string | null;
  last_entries: number;
  consecutive_failures: number;
  stored: number;
}

export interface SaudiPattern {
  key: string;
  label_ar: string;
  origin: string;
  incident_count: number;
  warning_article_count: number;
  support: string;
  categories: { label_ar: string; incidents: number }[];
  channels: string[];
  first_seen: string | null;
  last_seen: string | null;
  incidents: { id: number; title: string; first_reported_at: string | null; article_count: number }[];
  evidence: { type: string; incident_id?: number; title: string; quote: string; url: string; source: string }[];
}

export interface SaudiPatternsResponse {
  patterns: SaudiPattern[];
  channels: { channel: string; count: number }[];
  note?: string;
}

export interface SaudiRule {
  rule_id: string;
  category: string;
  category_ar: string;
  name: string;
  hypothesis: string;
  logic: string;
  required_data: string[];
  status: string;
  is_inference: boolean;
  test_results: any[];
  validation_plan: string[];
  origin: string;
  evidence: {
    basis: string;
    note: string;
    incident_count: number;
    incident_ids: number[];
    incidents: { id: number; title: string; article_count: number; first_reported_at: string | null; article_urls: string[] }[];
    warning_article_count: number;
    warning_articles: { threat_id: number; title: string; url: string }[];
    patterns: { key: string; label_ar: string; incident_count: number }[];
  };
}

export interface SaudiRulesResponse {
  rules: SaudiRule[];
  summary: { total: number; tested: number; proposed_untested: number };
  note: string;
}

export interface SaudiNewsParams {
  category?: string;
  source?: string;
  q?: string;
  bank_relevance?: string;
  content_type?: string;
  page?: number;
  page_size?: number;
}

// ---- Incident Registry ----
export interface IncidentArticleRef {
  article_id: number;
  threat_id: number;
  title: string;
  source: string | null;
  domain: string;
  source_code: string | null;
  url: string;
  published_at: string | null;
  evidence_confidence: number | null;
  content_type: string;
  match_basis: { type: string; [k: string]: any };
}

export interface SaudiIncident {
  id: number;
  ref: string;
  title: string;
  summary: string | null;
  fraud_category: string;
  fraud_category_ar: string;
  record_type: 'confirmed_incident' | 'reported_incident';
  record_type_ar: string;
  review_status: 'needs_review' | 'human_confirmed' | 'auto_merged' | 'single_source';
  review_status_ar: string;
  needs_review: boolean;
  first_reported_at: string | null;
  last_reported_at: string | null;
  event_date: string | null;
  article_count: number;
  source_count: number;
  bank_relevance: 'high' | 'medium' | 'low' | null;
  confidence: number | null;
  evidence_confidence: number | null;
  locations: string[];
  agencies: string[];
  channels: string[];
  articles: IncidentArticleRef[];
  // detail only
  facts?: {
    origin: string;
    persons_count: number | null;
    persons_counts_seen: number[];
    nationalities: { key: string; label: string }[];
    amounts: { value: number; [k: string]: any }[];
    locations: { label: string }[];
    agencies: { label: string; articles: number }[];
    seized_items: { key: string; label: string }[];
    fact_sentences: { text: string; source: string; url: string }[];
    corroborated_by_articles: number;
    corroborated_by_sources: number;
  };
  methods?: { tag: string; label: string; articles: number; quotes: { quote: string; source: string; url: string }[] }[];
  indicators?: {
    mentioned_in_source: { label: string; quote: string; url: string; articles: number; origin: string }[];
    suggested_by_system: { origin: string; items: string[] };
  };
  match_basis?: string;
  rule?: { rule_id: string; name: string; status: string } | null;
  pending_merge_suggestions?: MergeSuggestion[];
}

export interface SaudiIncidentsResponse {
  page: number;
  page_size: number;
  total: number;
  items: SaudiIncident[];
}

export interface MergeSuggestionArticle {
  id: number;
  headline?: string | null;
  source?: string | null;
  url?: string | null;
  published_at?: string | null;
  incident_id?: number | null;
  content_type?: string | null;
}

export interface MergeSuggestion {
  id: number;
  status: 'pending' | 'approved' | 'rejected';
  score: number;
  article_a: MergeSuggestionArticle;
  article_b: MergeSuggestionArticle;
  why_not_merged: any;
  reviewer?: string | null;
  reason?: string | null;
  decided_at?: string | null;
  created_at?: string | null;
}


export interface SaudiIncidentReport {
  generated_at: string;
  articles: { stored_in_db: number; saudi_fraud_articles: number; by_content_type: Record<string, number> };
  incidents: {
    independent_total: number;
    confirmed: number;
    reported: number;
    needs_review: number;
    multi_article: number;
    single_source: number;
    articles_behind_incidents: number;
    articles_per_incident: number;
  };
  separate_from_incidents: { fraud_warnings: number; awareness_campaigns: number };
  excluded_from_indicators: { general_cybersecurity: number; unrelated: number };
  unclassified: number;
  pending_merge_suggestions: number;
}

export interface SaudiRuntime {
  commit: string | null;
  branch: string | null;
  service: string | null;
  on_render: boolean;
  disk: {
    data_dir: string;
    db_path: string;
    db_inside_data_dir: boolean;
    data_dir_is_mount_point?: boolean;
    data_dir_on_separate_device?: boolean;
    db_size_bytes?: number | null;
    free_bytes?: number;
    error?: string;
  };
  sqlite_journal_mode: string | null;
  scheduler: {
    last_tick_at?: string | null;
    last_decision?: string | null;
    last_error?: string | null;
    last_run_started_at?: string | null;
    thread_alive?: boolean;
    [k: string]: any;
  };
  lease: { active: boolean; holder?: string; acquired_at?: string; expires_at?: string };
  admin_token_configured: boolean;
  manual_cooldown_seconds: number;
}
