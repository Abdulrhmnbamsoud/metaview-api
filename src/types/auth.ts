// src/types/auth.ts
// MetaView 3 — RBAC, Team Management & Analyst Governance Types

export type UserRole = 'admin' | 'analyst' | 'viewer';
export type UserStatus = 'active' | 'invited' | 'suspended' | 'revoked';

export interface User {
  id: number;
  email: string;
  full_name: string;
  role: UserRole;
  status: UserStatus;
  created_at: string;
  updated_at: string;
  last_login_at?: string | null;
  permissions: string[];
}

export interface Invitation {
  id: number;
  email: string;
  role: UserRole;
  invited_by: number;
  inviter_name?: string;
  created_at: string;
  expires_at: string;
  accepted_at?: string | null;
  revoked_at?: string | null;
  status: 'pending' | 'accepted' | 'expired' | 'revoked';
  invite_token?: string;
  invite_url?: string;
}

export interface AuditLog {
  id: number;
  user_id?: number | null;
  user_name?: string | null;
  user_email?: string | null;
  user_role?: string | null;
  action: string;
  target_type?: string | null;
  target_id?: string | null;
  details?: Record<string, any> | null;
  ip_address?: string | null;
  created_at: string;
}

export interface AnalystNote {
  id: number;
  incident_id: number;
  user_id: number;
  author_name: string;
  author_role: string;
  note_type: 'analysis' | 'classification_correction' | 'missing_info' | 'conflicting_info' | 'merge_suggestion';
  content: string;
  created_at: string;
  updated_at: string;
}

export interface SavedIncidentItem {
  id: number;
  incident_id: number;
  notes?: string;
  saved_at: string;
  title: string;
  headline?: string;
  fraud_category: string;
  fraud_category_ar: string;
  severity: string;
  articles_count: number;
  is_saudi: boolean;
}

export interface CustomPattern {
  id: number;
  key: string;
  label_ar: string;
  description?: string;
  modus_operandi?: string;
  channels: string[];
  target_groups: string[];
  banking_relevance?: string;
  linked_incident_ids: number[];
  hypothesis?: string;
  evidence_summary?: string;
  status: 'draft' | 'pending_review' | 'approved' | 'changes_requested' | 'rejected' | 'archived';
  author_id: number;
  author_name?: string;
  reviewed_by?: number;
  reviewer_name?: string;
  review_notes?: string;
  created_at: string;
  updated_at: string;
  approved_at?: string;
}

export interface DetectionRuleTestResult {
  tested_at: string;
  dataset: string;
  evaluated_records: number;
  alerts_simulated: number;
  precision: number;
  recall: number;
  false_positive_rate: string;
  tester_id: number;
  tester_name: string;
  status: string;
}

export interface CustomRule {
  id: number;
  rule_id: string;
  name: string;
  hypothesis: string;
  logic: string;
  pattern_key?: string;
  pattern_label_ar?: string;
  required_data: string[];
  exceptions?: string | string[];
  false_positive_risk?: string;
  test_results?: DetectionRuleTestResult | null;
  status: 'draft' | 'pending_review' | 'approved' | 'changes_requested' | 'rejected' | 'archived';
  author_id: number;
  author_name?: string;
  reviewed_by?: number;
  reviewer_name?: string;
  review_notes?: string;
  created_at: string;
  updated_at: string;
  approved_at?: string;
}

export interface ReviewRequest {
  id: number;
  item_type: 'pattern' | 'rule';
  item_id: string;
  item_title?: string;
  author_id: number;
  author_name: string;
  reviewer_id?: number | null;
  reviewer_name?: string | null;
  status: 'pending' | 'approved' | 'changes_requested' | 'rejected';
  notes?: string;
  review_comment?: string;
  created_at: string;
  updated_at: string;
  decided_at?: string | null;
}

export interface WorkspaceOverview {
  my_notes_count: number;
  saved_incidents_count: number;
  draft_patterns_count: number;
  pending_patterns_count: number;
  draft_rules_count: number;
  pending_rules_count: number;
  pending_reviews_for_admin?: number;
}
