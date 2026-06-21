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
  /** Feedback the user submitted for this message ('up' | 'down' | null) */
  feedback?: 'up' | 'down' | null;
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
