import type { SchemaInfo } from '../types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000';

export interface ChatApiResponse {
  response: string;
  sql: string | null;
  intent: string;
  validation: Record<string, unknown>;
  row_count: number | null;
  csv_path: string | null;
}

export interface Turn {
  role: 'user' | 'assistant';
  content: string;
  sql?: string | null;
  timestamp?: string;
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

export async function getSchema(): Promise<SchemaInfo> {
  return request<SchemaInfo>('/schema');
}

export function downloadUrl(filename: string): string {
  return `${API_BASE}/download/${encodeURIComponent(filename)}`;
}
