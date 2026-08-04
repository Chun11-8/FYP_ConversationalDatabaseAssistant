import { useState, useRef, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import type { Message, TextMessage, Role, SessionSummary } from './types';
import { ChatLayout } from './components/ChatLayout';
import type { AppTab } from './components/ChatLayout';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { DatabaseTab } from './components/DatabaseTab';
import {
  createSession, sendChat, sendChatStream, submitFeedback, listSessions, getHistory,
  getModels,
} from './api/client';
import type { ChatApiResponse } from './api/client';

const LS_API_KEY = 'db_assistant_api_key';
const LS_SESSION_ID = 'db_assistant_session_id';
const LS_MESSAGES = 'db_assistant_messages';
const LS_MODEL = 'db_assistant_model';

/** Human labels for pipeline stage events (GH#39). */
const STAGE_LABELS: Record<string, string> = {
  classify_intent: 'Understanding the question…',
  condense_query: 'Resolving follow-up…',
  retrieve_schema: 'Retrieving schema…',
  link_schema: 'Linking schema…',
  generate_sql: 'Generating SQL…',
  validate_sql: 'Validating SQL…',
  execute_sql: 'Running the query…',
  refine_sql: 'Refining the query…',
  format_response: 'Writing the answer…',
  handle_general: 'Searching…',
  ask_clarification: 'Thinking…',
  fallback_handler: 'Retrying…',
};

function loadStoredMessages(): Message[] {
  try {
    const raw = localStorage.getItem(LS_MESSAGES);
    return raw ? (JSON.parse(raw) as Message[]) : [];
  } catch {
    return [];
  }
}

/**
 * Persist the conversation, tolerating localStorage quota limits. `rows` can
 * hold up to `response.csv_row_limit` records per turn — large enough that an
 * uncaught QuotaExceededError here used to throw inside a useEffect and blank
 * the whole app. On failure, retry once with row payloads stripped: the
 * conversation text is what's worth restoring after a reload, not the last
 * query's row cache.
 */
function persistMessages(messages: Message[]): void {
  try {
    localStorage.setItem(LS_MESSAGES, JSON.stringify(messages));
  } catch {
    try {
      const slim = messages.map((m) =>
        m.type === 'text' && m.rows ? { ...m, rows: null } : m,
      );
      localStorage.setItem(LS_MESSAGES, JSON.stringify(slim));
    } catch {
      // Quota exhausted even without rows, or storage unavailable — the
      // session's in-memory state is still correct, only reload-restore is lost.
    }
  }
}

function App() {
  const [sessionId, setSessionId] = useState<string | null>(
    () => localStorage.getItem(LS_SESSION_ID),
  );
  const [messages, setMessages] = useState<Message[]>(loadStoredMessages);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<AppTab>('chat');
  const [apiKey, setApiKey] = useState<string>(() => localStorage.getItem(LS_API_KEY) ?? '');
  const [stage, setStage] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [defaultModel, setDefaultModel] = useState<string>('');
  const [model, setModel] = useState<string>(() => localStorage.getItem(LS_MODEL) ?? '');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Persist API key to localStorage whenever it changes
  useEffect(() => {
    if (apiKey) {
      localStorage.setItem(LS_API_KEY, apiKey);
    } else {
      localStorage.removeItem(LS_API_KEY);
    }
  }, [apiKey]);

  // Persist the model choice only when it diverges from the fetched default —
  // an untouched selection sends no override (GH#52), matching the api_key idiom.
  useEffect(() => {
    if (model && model !== defaultModel) {
      localStorage.setItem(LS_MODEL, model);
    } else {
      localStorage.removeItem(LS_MODEL);
    }
  }, [model, defaultModel]);

  // Fetch the selectable models once at startup; the picker stays hidden
  // (ChatInput guards on models.length) if this fails — chat still works.
  useEffect(() => {
    getModels()
      .then(({ models, default: def }) => {
        setAvailableModels(models);
        setDefaultModel(def);
        // GH#58: a persisted custom model (not in the curated `models` list)
        // must survive a reload — only fall back to the default when nothing
        // was chosen at all. `models.includes(prev)` here would silently
        // discard every custom entry the moment the page refreshed.
        setModel((prev) => prev || def);
      })
      .catch(() => { /* picker stays hidden; chat still works */ });
  }, []);

  // Persist current session id + messages so a page refresh restores the chat
  useEffect(() => {
    if (sessionId) localStorage.setItem(LS_SESSION_ID, sessionId);
  }, [sessionId]);

  useEffect(() => {
    persistMessages(messages);
  }, [messages]);

  const refreshSessions = useCallback(() => {
    listSessions().then(setSessions).catch(() => { /* sidebar is best-effort */ });
  }, []);

  // Initialise: reuse the persisted session if present, else create a new one.
  useEffect(() => {
    if (!sessionId) {
      createSession()
        .then(setSessionId)
        .catch(() => setError('Could not connect to the server. Is the backend running?'));
    }
    refreshSessions();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = useCallback(async (content: string) => {
    if (!sessionId || isLoading) return;

    const userMsg: TextMessage = {
      id: uuidv4(),
      role: 'user',
      timestamp: Date.now(),
      type: 'text',
      content,
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
    setError(null);

    // Only send a model override when it diverges from the fetched default
    // (GH#52) — an untouched selection keeps the backend's fast path.
    const modelOverride = model && model !== defaultModel ? model : undefined;

    try {
      // Streaming first (live stage progress); plain POST as fallback (GH#39).
      // The fallback is only safe when the stream never actually started: the
      // backend records the turn (and bills the LLM calls) BEFORE it queues
      // the terminal `done` event, so once a `stage` event has been seen the
      // pipeline is already running server-side — retrying via /chat would
      // re-run and double-bill it rather than recover a lost response.
      let result: ChatApiResponse;
      let sawStage = false;
      try {
        result = await sendChatStream(
          sessionId, content, apiKey.trim() || undefined, modelOverride,
          (e) => { if (e.type === 'stage') { sawStage = true; setStage(e.stage); } },
        );
      } catch (streamErr) {
        if (sawStage) throw streamErr;
        result = await sendChat(sessionId, content, apiKey.trim() || undefined, modelOverride);
      }
      const botMsg: TextMessage = {
        id: uuidv4(),
        role: 'assistant',
        timestamp: Date.now(),
        type: 'text',
        content: result.response,
        sql: result.sql,
        intent: result.intent,
        row_count: result.row_count,
        rows: result.rows,
        csv_path: result.csv_path,
        turn_id: result.turn_id,
        cache_hit: result.cache_hit,
        usage: result.usage,
        feedback: null,
      };
      setMessages((prev) => [...prev, botMsg]);
      refreshSessions();  // surface the (possibly new) session in the sidebar
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred.');
    } finally {
      setIsLoading(false);
      setStage(null);
    }
  }, [sessionId, isLoading, apiKey, model, defaultModel, refreshSessions]);

  const handleFeedback = useCallback(async (messageId: string, rating: 'up' | 'down') => {
    // Optimistic update — apply immediately, revert is not needed (feedback is best-effort)
    setMessages((prev) =>
      prev.map((m) =>
        m.id === messageId && m.type === 'text' && m.role === 'assistant'
          ? { ...m, feedback: rating }
          : m,
      ),
    );
    const msg = messages.find((m) => m.id === messageId);
    if (!msg || msg.type !== 'text' || msg.role !== 'assistant') return;
    const turnId = (msg as TextMessage).turn_id;
    if (turnId == null || !sessionId) return;
    try {
      await submitFeedback(turnId, sessionId, rating);
    } catch {
      // Feedback failure is non-critical — keep the optimistic UI state
    }
  }, [messages, sessionId]);

  const handleNewChat = useCallback(async () => {
    setMessages([]);
    setError(null);
    try {
      const newId = await createSession();
      setSessionId(newId);
    } catch {
      setError('Could not start a new session.');
    }
  }, []);

  const handleSelectSession = useCallback(async (id: string) => {
    if (id === sessionId || isLoading) return;
    setError(null);
    try {
      const turns = await getHistory(id);
      const restored: Message[] = turns
        .filter((t) => t.role === 'user' || t.role === 'assistant')
        .map((t) => ({
          id: uuidv4(),
          role: t.role as Role,
          timestamp: t.timestamp ? Date.parse(t.timestamp) : Date.now(),
          type: 'text',
          content: t.content,
          sql: t.sql ?? null,
          feedback: null,
        }));
      setSessionId(id);
      setMessages(restored);
    } catch {
      setError('Could not load that conversation.');
    }
  }, [sessionId, isLoading]);

  const handleExport = useCallback(() => {
    if (messages.length === 0) return;
    const blocks = messages.map((m) => {
      const who = m.role === 'user' ? '## You' : '## Assistant';
      let block = `${who}\n\n${m.content}`;
      if (m.role === 'assistant' && m.sql) {
        block += `\n\n\`\`\`sql\n${m.sql.trim()}\n\`\`\``;
      }
      return block;
    });
    const md = `# Conversation Export\n\n${blocks.join('\n\n---\n\n')}\n`;
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.md`;
    // Some browsers only honor .click() on an anchor that's actually in the
    // DOM, and revoking the blob URL on the very next line raced the download
    // starting. Append/remove around the click, and defer the revoke past the
    // current task so the browser has a chance to begin reading the blob.
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }, [messages]);

  return (
    <ChatLayout
      onNewChat={handleNewChat}
      sessions={sessions}
      activeSessionId={sessionId}
      onSelectSession={handleSelectSession}
      onExport={handleExport}
      canExport={messages.length > 0}
      activeTab={activeTab}
      onTabChange={setActiveTab}
      apiKey={apiKey}
      onApiKeyChange={setApiKey}
      models={availableModels}
      model={model || defaultModel}
      onModelChange={setModel}
    >
      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss">✕</button>
        </div>
      )}

      {activeTab === 'database' ? (
        <DatabaseTab />
      ) : (
        <>
          {messages.length === 0 && !isLoading ? (
            <div className="welcome-screen">
              <div className="welcome-logo">
                <svg stroke="currentColor" fill="none" strokeWidth="1.5" viewBox="0 0 24 24"
                  strokeLinecap="round" strokeLinejoin="round" height="2rem" width="2rem"
                  xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 8V4H8" />
                  <rect width="16" height="12" x="4" y="8" rx="2" />
                  <path d="M2 14h2M20 14h2M15 13v2M9 13v2" />
                </svg>
              </div>
              <h1 className="welcome-title">Database Assistant</h1>
              <p className="welcome-subtitle">
                Ask questions about your data in plain English.
              </p>
              <div className="welcome-examples">
                {[
                  'Show the top 10 players by overall rating',
                  'How many players are from Brazil?',
                  'What is the average age of goalkeepers?',
                ].map((q) => (
                  <button
                    key={q}
                    className="example-chip"
                    onClick={() => handleSend(q)}
                    disabled={!sessionId}
                    title={sessionId ? undefined : 'Connecting…'}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="chat-scroll">
              <div className="chat-container">
                {messages.map((msg) => (
                  <ChatMessage key={msg.id} message={msg} onFeedback={handleFeedback} />
                ))}

                {isLoading && (
                  <div className="message-wrapper message-bot">
                    <div className="message-inner">
                      <div className="avatar avatar-bot">
                        <svg stroke="currentColor" fill="none" strokeWidth="1.5" viewBox="0 0 24 24"
                          strokeLinecap="round" strokeLinejoin="round" height="16" width="16"
                          xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 8V4H8" />
                          <rect width="16" height="12" x="4" y="8" rx="2" />
                          <path d="M2 14h2M20 14h2M15 13v2M9 13v2" />
                        </svg>
                      </div>
                      <div>
                        <div className="typing-indicator">
                          <span /><span /><span />
                        </div>
                        {stage && (
                          <div className="msg-time" style={{ marginTop: '0.25rem' }}>
                            {STAGE_LABELS[stage] ?? stage}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} style={{ height: '1px' }} />
              </div>
            </div>
          )}

          <ChatInput onSend={handleSend} disabled={isLoading || !sessionId} />
        </>
      )}
    </ChatLayout>
  );
}

export default App;
