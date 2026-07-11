import { useState, useRef, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import type { Message, TextMessage, Role, SessionSummary } from './types';
import { ChatLayout } from './components/ChatLayout';
import type { AppTab } from './components/ChatLayout';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { DatabaseTab } from './components/DatabaseTab';
import {
  createSession, sendChat, submitFeedback, listSessions, getHistory,
} from './api/client';

const LS_API_KEY = 'db_assistant_api_key';
const LS_SESSION_ID = 'db_assistant_session_id';
const LS_MESSAGES = 'db_assistant_messages';

function loadStoredMessages(): Message[] {
  try {
    const raw = localStorage.getItem(LS_MESSAGES);
    return raw ? (JSON.parse(raw) as Message[]) : [];
  } catch {
    return [];
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
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Persist API key to localStorage whenever it changes
  useEffect(() => {
    if (apiKey) {
      localStorage.setItem(LS_API_KEY, apiKey);
    } else {
      localStorage.removeItem(LS_API_KEY);
    }
  }, [apiKey]);

  // Persist current session id + messages so a page refresh restores the chat
  useEffect(() => {
    if (sessionId) localStorage.setItem(LS_SESSION_ID, sessionId);
  }, [sessionId]);

  useEffect(() => {
    localStorage.setItem(LS_MESSAGES, JSON.stringify(messages));
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

    try {
      const result = await sendChat(sessionId, content, apiKey.trim() || undefined);
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
        feedback: null,
      };
      setMessages((prev) => [...prev, botMsg]);
      refreshSessions();  // surface the (possibly new) session in the sidebar
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred.');
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, isLoading, apiKey, refreshSessions]);

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
    a.click();
    URL.revokeObjectURL(url);
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
                  <button key={q} className="example-chip" onClick={() => handleSend(q)}>
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
                      <div className="typing-indicator">
                        <span /><span /><span />
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
