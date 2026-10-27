// src/types/rules_studio.ts
// Unified Fraud Detection Rules Intelligence & Deployment Studio Types

export type RuleOrigin = 'intelligence' | 'analyst' | 'hybrid';

export type RuleStatus =
  | 'draft'
  | 'evidence_review'
  | 'logic_review'
  | 'simulation'
  | 'ready_for_bank_testing'
  | 'approved_externally'
  | 'rejected'
  | 'needs_revision';

export type DeploymentEnvironmentKey = 'sas' | 'branch_group' | 'shared_room' | 'hybrid';

export interface SystemCapabilityEnvironment {
  id: DeploymentEnvironmentKey;
  name_ar: string;
  name_en: string;
  type: string;
  type_ar: string;
  processing_mode: string;
  processing_mode_ar: string;
  supported_operations: string[];
  supported_fields: string[];
  event_correlation: {
    supported: boolean;
    description_ar: string;
  };
  temporal_windows: {
    supported: boolean;
    range_ar: string;
  };
  rule_execution_capabilities: string[];
  supported_actions: string[];
  testing_capabilities: string[];
  technical_limitations: string[];
  access_requirements: string[];
}

export interface RuleEvidence {
  incident_ids: number[];
  article_ids: number[];
  sources: string[];
  extracted_facts: string[];
  corroboration_count: number;
  pattern_key?: string;
  note?: string;
}

export interface RuleCondition {
  id: string;
  field: string;
  operator: string;
  value: any;
  description_ar: string;
  is_calibrated: boolean;
}

export interface RuleLogic {
  raw_expression?: string;
  trigger_events: string[];
  conditions: RuleCondition[];
  event_sequence: string[];
  time_window_minutes?: number;
  time_window_description?: string;
  customer_baseline_required?: string;
  exception_conditions: string[];
  detection_limitations: string[];
}

export interface RuleDataField {
  name: string;
  type: string;
  description: string;
  availability?: 'high' | 'medium' | 'low' | 'rare';
}

export interface RuleDataRequirements {
  required_fields: RuleDataField[];
  optional_fields: RuleDataField[];
  data_sources: string[];
  dependencies: string[];
}

export interface RuleScoring {
  evidence_confidence: number;
  detection_feasibility: number;
  operational_readiness: number;
  rationales: {
    evidence: string;
    feasibility: string;
    readiness: string;
  };
}

export interface RuleDeployment {
  recommended_environment: DeploymentEnvironmentKey;
  recommended_environment_details?: SystemCapabilityEnvironment;
  alternative_environments: (DeploymentEnvironmentKey | SystemCapabilityEnvironment)[];
  rationale: string;
  missing_capabilities: string[];
  implementation_complexity: 'low' | 'medium' | 'high';
  complexity_factors: string[];
  hybrid_breakdown?: {
    detection_engine: string;
    operational_response: string;
    coordination_flow: string;
  };
}

export interface SimulationTestCase {
  id: string;
  type: 'positive_fraud' | 'negative_normal' | 'boundary_case' | 'missing_data' | 'out_of_order_events' | 'velocity_repeated';
  type_ar: string;
  description: string;
  synthetic_tag: string;
  input_payload: Record<string, any>;
  expected_outcome: string;
  rationale: string;
}

export interface EvaluatedTestCaseResult {
  test_case_id: string;
  type: string;
  type_ar: string;
  description: string;
  synthetic_tag: string;
  input_payload: Record<string, any>;
  matched_conditions: string[];
  unmatched_conditions: string[];
  missing_fields: string[];
  alert_fired: boolean;
  verdict: 'ALERT_TRIGGERED' | 'NO_ALERT' | 'INSUFFICIENT_DATA' | 'NO_ALERT_ORDER_CONFLICT';
  verdict_ar: string;
  decision_reason: string;
  is_logic_test_passed: boolean;
}

export interface SimulationSuiteResult {
  rule_id: string;
  simulated_at: string;
  total_test_cases: number;
  passed_cases: number;
  all_logic_passed: boolean;
  disclaimer_ar: string;
  results: EvaluatedTestCaseResult[];
}

export interface RuleVersionEntry {
  version: string;
  timestamp: string;
  author: string;
  change_summary: string;
  reason: string;
}

export interface RuleGovernance {
  created_by: string;
  created_at: string;
  updated_at: string;
  current_version: string;
  version_history: RuleVersionEntry[];
  external_approval?: {
    authority: string;
    reference_doc: string;
    approved_at: string | null;
    notes: string;
  } | null;
}

export interface UnifiedRule {
  id: string;
  rule_id: string;
  name: string;
  name_en?: string;
  fraud_category: string;
  fraud_category_ar: string;
  origin: RuleOrigin;
  status: RuleStatus;
  detection_hypothesis: string;
  hypothesis?: string;
  fraud_scenario: string;
  evidence: RuleEvidence;
  technical_assumptions: string[];
  logic: RuleLogic;
  data_requirements: RuleDataRequirements;
  scoring: RuleScoring;
  deployment: RuleDeployment;
  simulation_suite?: {
    test_cases: SimulationTestCase[];
    last_simulated_at?: string;
    all_passed?: boolean;
  };
  governance: RuleGovernance;
  scenario_extraction?: {
    triggers: string[];
    conditions: RuleCondition[];
    time_window: string;
    baseline: string;
    entities: Array<{ entity: string; role: string }>;
    exclusions: string[];
    suggested_actions: string[];
    missing_info: string[];
  };
}

export interface RulesStudioOverview {
  total_rules: number;
  testable_rules: number;
  needing_data_rules: number;
  awaiting_review_rules: number;
  origin_breakdown: Record<RuleOrigin, number>;
  status_breakdown: Record<RuleStatus, number>;
  category_breakdown: Record<string, number>;
  environment_breakdown: Record<DeploymentEnvironmentKey, number>;
  timestamp: string;
}
