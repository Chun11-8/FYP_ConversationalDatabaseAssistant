import type { SchemaInfo, SessionSummary } from '../types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000';

export interface ChatApiResponse {
  response: string;
  sql: string | null;
  intent: string;
  validation: Record<string, unknown>;
  row_count: number | null;
  csv_path: string | null;
  turn_id: number | null;
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
