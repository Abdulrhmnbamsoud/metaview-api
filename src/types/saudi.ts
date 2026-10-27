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
  // Separate telemetry indicators
  last_source_check?: string | null;
  last_successful_fetch?: string | null;
  last_new_item?: string | null;
  last_analysis_update?: string | null;
  // incident layer
  incidents_total?: number;
  incidents_high_bank_relevance?: number;
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
  country?: string | null;
  incident_country?: string | null;
  incident_city?: string | null;
  occurred_at?: string | null;
  domain?: string | null;
  is_saudi?: boolean;
  content_excerpt?: string;
}

export interface CountryCount {
  country: string;
  count: number;
}

export interface CountryDistributionResponse {
  countries: CountryCount[];
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
  country?: string;
  handle?: string;
  notes?: string;
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
  confirmed_incident_count?: number;
  reported_incident_count?: number;
  warning_article_count: number;
  source_count?: number;
  pattern_tier?: 'confirmed_fraud_pattern' | 'observed_repeated_pattern' | 'reported_fraud_method' | 'warning_based_intelligence' | 'emerging_pattern' | 'insufficient_evidence';
  pattern_tier_ar?: string;
  classification_rationale?: string;
  summary?: string;
  bank_relevance?: 'high' | 'medium' | 'low';
  bank_relevance_ar?: string;
  bank_relevance_rationale?: string;
  confidence_score?: number;
  confidence_level?: 'high' | 'medium' | 'low';
  confidence_level_ar?: string;
  confidence_rationale?: string;
  support: string;
  categories: { label_ar: string; incidents: number }[];
  channels: string[];
  first_seen: string | null;
  last_seen: string | null;
  how_it_works?: {
    description: string;
    steps: string[];
    channels: string[];
    impersonated_entities: string[];
    targeted_victims: string[];
  };
  banking_analysis?: {
    potential_bank_risks: string[];
    fraud_indicators: string[];
    detection_hypotheses: string[];
    required_data: string[];
    uncertainties_and_limitations: string[];
  };
  incidents: { id: number; title: string; first_reported_at: string | null; article_count: number; record_type?: string }[];
  warning_articles?: { threat_id: number; title: string; url: string; source?: string; published_at?: string }[];
  evidence: { type: string; incident_id?: number; threat_id?: number; title: string; quote: string; url: string; source: string }[];
  linked_rules?: { rule_id: string; name: string; hypothesis: string; logic?: string }[];
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
  scope?: 'saudi' | 'global' | 'all';
  country?: string;
  window_days?: number;
  date_from?: string;
  date_to?: string;
  sort_by?: 'latest' | 'importance';
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
  incident_country?: string;
  incident_city?: string;
  incident_region?: string;
  fraud_type?: string;
  fraud_type_ar?: string;
  targeted_sector_ar?: string;
  reporting_agency?: string;
  occurred_at?: string | null;
  modus_operandi_detailed?: string | null;
  country?: string;
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

export interface XPublicCollectorStatus {
  source_code: string;
  handle: string;
  status: 'active' | 'requires_authentication' | 'access_restricted' | 'failed' | 'not_supported' | string;
  status_label: string;
  status_arabic: string;
  last_success_at: string | null;
  last_error: string | null;
  total_posts_stored: number;
  official_api_token_configured: boolean;
}

export interface XPublicPostDetail {
  post_id: string;
  text: string;
  published_at: string | null;
  url: string;
  author_handle: string;
  author_name: string;
  source_code: string;
  outbound_urls: string[];
  media_urls: string[];
  category_key: string;
  category_label: string;
  content_type: string;
  classification_reasons: string[];
  is_eligible_explore: boolean;
  is_eligible_incident: boolean;
}

export interface XPublicCollectionReport {
  handle: string;
  source_code: string;
  status: string;
  status_label: string;
  status_arabic: string;
  error: string | null;
  total_fetched: number;
  new_stored: number;
  duplicates_skipped: number;
  classified_categories: Record<string, number>;
  eligible_explore_items: number;
  canonical_incidents_created: number;
  stored_items: Array<{
    post_id: string;
    url: string;
    category: string;
    content_type: string;
    text_snippet: string;
  }>;
}
