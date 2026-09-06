export type Role = 'user' | 'assistant';

export interface BaseMessage {
  id: string;
  role: Role;
  timestamp: number;
}

export interface TextMessage extends BaseMessage {
  type: 'text';
  content: string;
  /** SQL query used to generate this response (assistant only) */
  sql?: string | null;
  /** Intent classification: READ | GENERAL | WRITE_BLOCKED | AMBIGUOUS */
  intent?: string;
  /** Number of DB rows returned */
  row_count?: number | null;
  /** Raw DB rows — rendered by RecordsPanel (first 10 visible, rest toggleable) */
  rows?: Record<string, unknown>[] | null;
  /** CSV filename served at /download/{csv_path} */
  csv_path?: string | null;
  /** DB id of the assistant turn — used to submit feedback */
  turn_id?: number | null;
  /** Answered from the semantic cache (no generation) */
  cache_hit?: boolean;
  /** LLM tokens + estimated cost for this turn (GH#57) */
  usage?: TurnUsage | null;
  /** How this answer was arrived at (GH#67) — evidence, not a verdict */
  verification?: TurnVerification | null;
  /** Feedback the user submitted for this message ('up' | 'down' | null) */
  feedback?: 'up' | 'down' | null;
}

/** Per-turn LLM accounting, as returned by /chat and /chat/stream. */
export interface TurnUsage {
  calls: number;
  input_tokens: number;
  output_tokens: number;
  est_cost_usd: number;
  by_stage?: Record<string, {
    calls: number;
    input_tokens: number;
    output_tokens: number;
    est_cost_usd: number;
  }>;
}

/**
 * Evidence about how the winning SQL was chosen (GH#67).
 *
 * Deliberately facts, not a confidence verdict: nothing has yet established
 * that agreement predicts correctness (that is the calibration study), and a
 * miscalibrated badge would teach users to trust the wrong answers.
 */
export interface TurnVerification {
  /** SQL candidates generated this turn */
  candidates: number;
  /** Candidates that validated and executed successfully */
  executed: number;
  /** Candidates whose result set matched the winner's */
  agreement: number;
  /** How many genuinely different result sets the candidates produced */
  distinct_result_sets: number;
  /** Which mechanism picked the winner */
  decided_by:
    | 'plurality'            // most candidates produced this result set
    | 'non_empty_tiebreak'   // tie, resolved toward the non-empty result
    | 'unit_test'            // unit-test screen left one survivor
    | 'pairwise'             // LLM pairwise comparison broke the tie
    | 'reranker'             // voting inconclusive — text-only reranker chose
    | 'inconclusive'
    | 'single_candidate'     // only one candidate — nothing to compare
    | 'voting_disabled'
    | 'none_executed'
    | 'generation_failed'
    | 'cache_hit'            // reused a stored query; no voting this turn
    | 'not_generated'        // GENERAL / blocked / clarifying turn
    | string;
  /** ReAct execute→refine rounds taken (0 = first attempt worked) */
  refine_rounds?: number;
  validated?: boolean;
  validation_checks?: Record<string, boolean>;
  /** Gold-free risk flags on the executed result (GH#69) */
  risks?: TurnRisk[];
  /** Severity-weighted flag count. A ranking aid, NOT a probability. */
  risk_score?: number;
}

/** One thing that looks wrong about an answer that nonetheless ran fine. */
export interface TurnRisk {
  code: string;
  severity: 'high' | 'medium' | 'low' | string;
  /** Written to be actionable on its own — shown verbatim to the user. */
  message: string;
  detail?: Record<string, unknown> | null;
}

export type Message = TextMessage;

// ── Session sidebar types ────────────────────────────────────────────────────

export interface SessionSummary {
  session_id: string;
  title: string | null;
  created_at: string | null;
  updated_at: string | null;
}

// ── Schema browser types ────────────────────────────────────────────────────

export interface SchemaColumn {
  name: string;
  type: string;
  not_null: boolean;
  primary_key: boolean;
  foreign_key: string | null;
  description?: string | null;
  value_description?: string | null;
}

export interface SchemaTable {
  table_name: string;
  column_count: number;
  columns: SchemaColumn[];
}

export interface SchemaInfo {
  db_name: string;
  table_count: number;
  tables: SchemaTable[];
}
