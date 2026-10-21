// src/types/index.ts

export interface Article {
  headline: string;
  content?: string | null;
  article_summary?: string | null;
  published_at: string;
  source: string;
  domain?: string | null;
  country?: string | null;
  url: string;
  category?: string | null;
  entity?: string | null;
  image_url?: string | null;
}

export interface SearchResponse {
  count: number;
  results: Article[];
  total_rows?: number;
  returned?: number;
  page?: number;
  page_size?: number;
  offset?: number;
  sort_by?: string;
  sort_dir?: string;
}

export interface SearchParams {
  q?: string;
  page?: number;
  page_size?: number;
  source?: string;
  country?: string;
  domain?: string;
  category?: string;
  entity?: string;
  min_date?: string;
  max_date?: string;
  sort_by?: string;
  sort_dir?: 'asc' | 'desc';
  include_content?: boolean;
}

export interface HealthStatus {
  status: string;
  service: string;
  rows: number;
  db_path?: string;
  time: string;
}

export interface DistributionItem {
  source?: string;
  country?: string;
  count: number;
}

export interface StatsSummary {
  total_articles: number;
  total_sources: number;
  articles_last_24h: number;
  last_sync?: string | null;
  source_distribution: DistributionItem[];
  country_distribution: DistributionItem[];
}

export interface SourceInfo {
  code: string;
  name: string;
  feed_url: string;
  country: string;
  articles_count: number;
  latest_published?: string | null;
}

export interface SourcesResponse {
  count: number;
  total_sources: number;
  sources: SourceInfo[];
  raw_feeds: string[];
}

export interface IngestRunResult {
  status: string;
  started_at: string;
  finished_at: string;
  inserted_total: number;
  errors_total: number;
  skipped_no_url_total: number;
  skipped_dup_total: number;
  with_images_total: number;
  rows_after: number;
  sources: Array<{
    source: string;
    feed_url: string;
    fetched_entries: number;
    inserted: number;
    skipped_no_url: number;
    skipped_dup: number;
    with_images?: number;
    limit_per_feed: number;
    time_sec: number;
    error?: string | null;
  }>;
}

export interface CountryNewsCount {
  country: string;
  count: number;
}
