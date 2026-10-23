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

export interface SaudiPatternsResponse {
  patterns: { key: string; label_ar: string; count: number; modus_operandi: string | null }[];
  channels: { channel: string; count: number }[];
}

export interface SaudiRulesResponse {
  rules: {
    category: string;
    category_ar: string;
    rule: SaudiDetectionRule | null;
    required_data: string[] | null;
    supporting_items: number;
    example_threat_ids: number[];
  }[];
  note: string;
}

export interface SaudiNewsParams {
  category?: string;
  source?: string;
  q?: string;
  bank_relevance?: string;
  page?: number;
  page_size?: number;
}
