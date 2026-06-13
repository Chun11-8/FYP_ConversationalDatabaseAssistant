import { useState, useRef, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import type { Message, TextMessage } from './types';
import { ChatLayout } from './components/ChatLayout';
import type { AppTab } from './components/ChatLayout';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { DatabaseTab } from './components/DatabaseTab';
import { createSession, sendChat } from './api/client';

const LS_API_KEY = 'db_assistant_api_key';

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
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

  // Initialise session on mount
  useEffect(() => {
    createSession()
      .then(setSessionId)
      .catch(() => setError('Could not connect to the server. Is the backend running?'));
  }, []);

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
        csv_path: result.csv_path,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred.');
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, isLoading, apiKey]);

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

  return (
    <ChatLayout
      onNewChat={handleNewChat}
      chatHistory={[]}
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
                  <ChatMessage key={msg.id} message={msg} />
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
