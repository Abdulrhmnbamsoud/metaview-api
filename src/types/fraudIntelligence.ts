// src/types/fraudIntelligence.ts
/**
 * Types for MetaView 3 — Fraud Intelligence Dashboard
 */

export interface FraudIntelligenceSummary {
  total_unique_incidents: number;
  confirmed_incidents: number;
  reported_incidents: number;
  emerging_threats_count: number;
  total_articles_evaluated: number;
  incident_articles_count: number;
  deduplication_saved_duplicates: number;
  financial_loss_documented: {
    cases_count: number;
    total_usd_estimate: number;
    breakdown_by_currency: Record<string, number>;
  };
  excluded_content_breakdown: {
    unrelated_news: number;
    general_cybersecurity: number;
    awareness_campaigns: number;
    fraud_warnings: number;
  };
}

export interface TrendTimelinePoint {
  date: string;
  observed_incidents: number;
  articles_volume: number;
}

export interface TrendResponse {
  window_days: number;
  period_current: { start: string; end: string };
  period_previous: { start: string; end: string };
  current_observed_incidents: number;
  previous_observed_incidents: number;
  observed_incidents_change_pct: number;
  current_source_coverage_volume: number;
  previous_source_coverage_volume: number;
  source_coverage_change_pct: number;
  neutral_disclaimer: string;
  series: TrendTimelinePoint[];
}

export interface TypologyStat {
  key: string;
  name_ar: string;
  name_en: string;
  count: number;
  percentage?: number;
  incident_ids: number[];
}

export interface GeoCountryStat {
  country_code: string;
  name_ar: string;
  name_en: string;
  count: number;
  cities_count: number;
  cities: string[];
  incident_ids: number[];
}

export interface GeoCityStat {
  city: string;
  region: string;
  country_code: string;
  country_ar: string;
  count: number;
  incident_ids: number[];
}

export interface GeoResponse {
  documented_geo_incidents: number;
  unknown_geo_incidents: number;
  saudi_incidents_count: number;
  saudi_incident_ids: number[];
  countries: GeoCountryStat[];
  cities: GeoCityStat[];
}

export interface SectorStat {
  key: string;
  name_ar: string;
  count: number;
  incident_ids: number[];
}

export interface ImpersonatedEntityStat {
  entity: string;
  count: number;
  incident_ids: number[];
}

export interface EntitiesAndSectorsResponse {
  sectors: SectorStat[];
  impersonated_entities: ImpersonatedEntityStat[];
}

export interface ChannelStat {
  channel: string;
  name_ar: string;
  count: number;
  incident_ids: number[];
}

export interface SupportingArticle {
  id: number;
  headline: string;
  article_summary: string;
  url: string;
  source: string;
  country: string;
  published_at: string;
  role: string;
  match_basis: string;
  content_type: string;
}

export interface EnrichedIncident {
  id: number;
  ref: string;
  title: string;
  summary: string;
  record_type: 'confirmed_incident' | 'reported_incident';
  record_type_ar: string;
  fraud_type: string;
  fraud_type_ar: string;
  incident_country: string;
  incident_city: string;
  incident_region: string;
  victim_location: string;
  suspect_location: string;
  channel: string;
  channel_ar: string;
  targeted_sector: string;
  targeted_sector_ar: string;
  impersonated_entity: string;
  reporting_agency: string;
  financial_loss_amount: number | null;
  financial_loss_currency: string;
  financial_loss_raw: string;
  occurred_at: string | null;
  published_at: string;
  evidence_text: string;
  evidence_sources: string[];
  article_count: number;
  articles: Array<{ article_id: number; headline: string; url: string }>;
  is_emerging_threat: boolean;
  emerging_rationale?: string | null;
  confidence: number;
  supporting_articles?: SupportingArticle[];
}

export interface BankingRiskResponse {
  total_banking_incidents: number;
  estimated_banking_loss_usd: number;
  top_impersonated_banks: Record<string, number>;
  incidents: EnrichedIncident[];
}

export interface FieldCompletionStat {
  field: string;
  filled_count: number;
  total: number;
  percentage: number;
}

export interface RegisteredSourceHealth {
  code: string;
  name: string;
  kind: string;
  country: string;
  official: number;
  enabled: number;
  last_status: string;
  last_success_at: string | null;
}

export interface DataQualityResponse {
  total_incidents: number;
  field_completion_stats: FieldCompletionStat[];
  registered_sources_count: number;
  sources: RegisteredSourceHealth[];
  audit_history: any[];
}

export interface DrilldownResponse {
  page: number;
  page_size: number;
  total: number;
  items: EnrichedIncident[];
}

export interface HeroInsight {
  headline: string;
  summary: string;
  trend_badge: string;
  growth_pct: string;
  growth_direction: 'up' | 'down' | 'neutral';
  growth_note?: string;
  sources_count: number;
  incidents_count: number;
  confidence_label: string;
  incident_ids: number[];
  is_neutral?: boolean;
  fraud_type?: string;
}

export interface MapPoint {
  id: string;
  name: string;
  count: number;
  color: string;
  region: string;
  x: number;
  y: number;
  top_fraud_types?: Array<{ type_ar: string; count: number }>;
  top_channels?: Array<{ channel_ar: string; count: number }>;
  incident_ids?: number[];
  incidents?: Array<{
    id: number;
    title: string;
    fraud_type_ar?: string;
    channel_ar?: string;
    occurred_at?: string;
    evidence_text?: string;
  }>;
}

export interface TopPatternItem {
  key: string;
  name_ar: string;
  incidents_count: number;
  sources_count: number;
  change_pct: number | null;
  change_label?: string;
  calculation_tooltip?: string;
  direction: 'up' | 'down' | 'neutral';
  color: string;
  sparkline: number[];
  icon: string;
  incident_ids?: number[];
}

export interface FlowNode {
  id: string;
  name_ar: string;
  percentage: number;
  count?: number;
  total_incidents?: number;
  pct_basis?: string;
  color: string;
  icon?: string;
}

export interface FlowRibbon {
  from: string;
  to: string;
  pct: number;
  count?: number;
  total_incidents?: number;
  pct_basis?: string;
}

export interface ObservedMechanismItem {
  incident_id: number;
  incident_ref: string;
  title: string;
  channel_id: string;
  channel_ar: string;
  fraud_type_id: string;
  fraud_type_ar: string;
  sector_id: string;
  sector_ar: string;
  evidence_text: string;
  occurred_at: string;
  status: string;
  status_color: 'emerald' | 'amber';
}

export interface RelationshipFlowData {
  channels: FlowNode[];
  typologies: FlowNode[];
  sectors: FlowNode[];
  flows: FlowRibbon[];
  observed_mechanisms?: ObservedMechanismItem[];
}

export interface WhatChangedItem {
  id: number;
  title: string;
  detail: string;
  direction: 'up' | 'down';
  icon: string;
}

export interface LatestTableIncident {
  id: number;
  date: string;
  title: string;
  type: string;
  location: string;
  channel: string;
  status: string;
  status_color: 'emerald' | 'amber';
}

export interface BankingImplicationItem {
  id: number;
  title: string;
  detail: string;
  icon: string;
}

export interface DataQualityStats {
  articles_analyzed: number;
  articles_change_pct: string;
  incidents_extracted: number;
  incidents_change_pct: string;
  active_sources: number;
  sources_change_pct: string;
  pending_sources: number;
  pending_change_pct: string;
  fields_completion_pct: number;
  missing_fields_note: string;
}

export interface SaudiLocationStats {
  total_saudi_incidents: number;
  verified_location_incidents: number;
  unknown_location_incidents: number;
}

export interface DashboardViewResponse {
  is_empty?: boolean;
  empty_message?: string | null;
  data_state?: 'data_rich' | 'data_limited' | 'insufficient_evidence';
  hero_insight: HeroInsight;
  map_points: MapPoint[];
  unknown_location_count?: number;
  unknown_location_incident_ids?: number[];
  unknown_location_incidents?: Array<{
    id: number;
    title: string;
    fraud_type_ar?: string;
    channel_ar?: string;
    occurred_at?: string;
    evidence_text?: string;
  }>;
  total_scope_incidents?: number;
  mapped_incidents_count?: number;
  saudi_stats?: SaudiLocationStats;
  top_patterns: TopPatternItem[];
  relationship_flow: RelationshipFlowData;
  what_changed: WhatChangedItem[];
  latest_incidents: LatestTableIncident[];
  banking_implications: BankingImplicationItem[];
  data_quality_stats: DataQualityStats;
  active_filters?: {
    geo: string;
    country: string;
    fraud_type: string;
    date_from: string;
    date_to: string;
    date_basis: string;
    window_days: number;
    map_mode?: string;
  };
}

