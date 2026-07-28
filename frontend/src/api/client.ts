import type { SchemaInfo, SessionSummary } from '../types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000';

export interface ChatApiResponse {
  response: string;
  sql: string | null;
  intent: string;
  validation: Record<string, unknown>;
  row_count: number | null;
  rows: Record<string, unknown>[] | null;
  csv_path: string | null;
  turn_id: number | null;
  cache_hit?: boolean;
}

export interface Turn {
  role: 'user' | 'assistant' | 'summary';
  content: string;
  sql?: string | null;
  valid?: boolean | null;
  timestamp?: string | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function createSession(): Promise<string> {
  const data = await request<{ session_id: string }>('/session', { method: 'POST' });
  return data.session_id;
}

export type StreamEvent =
  | { type: 'stage'; stage: string }
  | { type: 'sql'; sql: string }
  | ({ type: 'done' } & ChatApiResponse)
  | { type: 'error'; message: string };

/**
 * Streaming variant of sendChat (SSE). Emits stage/sql events via onEvent and
 * resolves with the final ChatApiResponse from the terminal `done` event.
 * Throws on `error` events or transport failure — callers fall back to sendChat.
 */
export async function sendChatStream(
  sessionId: string,
  message: string,
  apiKey: string | undefined,
  onEvent: (e: StreamEvent) => void,
): Promise<ChatApiResponse> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      ...(apiKey ? { api_key: apiKey } : {}),
    }),
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let donePayload: ChatApiResponse | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const chunk = buffer.slice(0, sep).trim();
      buffer = buffer.slice(sep + 2);
      if (!chunk.startsWith('data: ')) continue;
      const evt = JSON.parse(chunk.slice(6)) as StreamEvent;
      onEvent(evt);
      if (evt.type === 'done') donePayload = evt as unknown as ChatApiResponse;
      if (evt.type === 'error') throw new Error(evt.message);
    }
  }
  if (!donePayload) throw new Error('Stream ended without a result');
  return donePayload;
}

export async function sendChat(
  sessionId: string,
  message: string,
  apiKey?: string,
): Promise<ChatApiResponse> {
  return request<ChatApiResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      message,
      ...(apiKey ? { api_key: apiKey } : {}),
    }),
  });
}

export async function getHistory(sessionId: string): Promise<Turn[]> {
  return request<Turn[]>(`/history/${encodeURIComponent(sessionId)}`);
}

export async function listSessions(): Promise<SessionSummary[]> {
  return request<SessionSummary[]>('/sessions');
}

export async function getSchema(): Promise<SchemaInfo> {
  return request<SchemaInfo>('/schema');
}

export async function submitFeedback(
  turnId: number,
  sessionId: string,
  rating: 'up' | 'down',
  comment?: string,
): Promise<void> {
  await request<{ status: string }>('/feedback', {
    method: 'POST',
    body: JSON.stringify({
      turn_id: turnId,
      session_id: sessionId,
      rating,
      ...(comment ? { comment } : {}),
    }),
  });
}

export function downloadUrl(filename: string): string {
  return `${API_BASE}/download/${encodeURIComponent(filename)}`;
}

// ── Schema versioning / index sync (Issue #7) ──────────────────────────────

export interface SchemaChange {
  action: 'add_table' | 'drop_table' | 'add_column' | 'drop_column' | 'modify_column';
  table: string;
  column?: string | null;
  detail?: string | null;
}

export interface SchemaDiff {
  db_name: string;
  in_sync: boolean;
  last_synced_at: string | null;
  changes: SchemaChange[];
  unindexed_tables: string[];
}

export interface SchemaSyncResult {
  db_name: string;
  schema_hash: string;
  changes_applied: SchemaChange[];
  reembedded_tables: string[];
  removed_tables: string[];
  full: boolean;
}

export async function getSchemaDiff(): Promise<SchemaDiff> {
  return request<SchemaDiff>('/schema/diff');
}

export async function syncSchema(full: boolean): Promise<SchemaSyncResult> {
  return request<SchemaSyncResult>('/schema/sync', {
    method: 'POST',
    body: JSON.stringify({ full }),
  });
}
