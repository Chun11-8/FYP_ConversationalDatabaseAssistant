import React, { useState, useRef, useEffect } from 'react';
import type { ReactNode } from 'react';
import { Plus, Menu, X, Database, MessageSquare, Eye, EyeOff, Download, ChevronDown } from 'lucide-react';
import type { SessionSummary } from '../types';

export type AppTab = 'chat' | 'database';

interface ChatLayoutProps {
  children: ReactNode;
  onNewChat: () => void;
  sessions: SessionSummary[];
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onExport: () => void;
  canExport: boolean;
  activeTab: AppTab;
  onTabChange: (tab: AppTab) => void;
  apiKey: string;
  onApiKeyChange: (key: string) => void;
  models: string[];
  model: string;
  onModelChange: (model: string) => void;
}

export const ChatLayout: React.FC<ChatLayoutProps> = ({
  children,
  onNewChat,
  sessions,
  activeSessionId,
  onSelectSession,
  onExport,
  canExport,
  activeTab,
  onTabChange,
  apiKey,
  onApiKeyChange,
  models,
  model,
  onModelChange,
}) => {
  // Desktop: sidebar starts open. Mobile (<768px): sidebar starts closed (overlay).
  const isMobile = () => window.innerWidth < 768;
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile());
  const [showKey, setShowKey] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);
  const pickerRef = useRef<HTMLDivElement>(null);

  const toggleSidebar = () => setSidebarOpen((v) => !v);

  // Close the model picker on an outside click.
  useEffect(() => {
    if (!pickerOpen) return;
    const onClick = (e: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setPickerOpen(false);
      }
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [pickerOpen]);

  return (
    <div className="app-layout">
      {/* Mobile overlay — tap to close */}
      {sidebarOpen && isMobile() && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : 'sidebar-collapsed'}`}>
        <div className="sidebar-content">
          {/* Branding */}
          <div className="sidebar-brand">
            <Database size={18} className="sidebar-brand-icon" />
            <span className="sidebar-brand-name">DB Assistant</span>
          </div>

          <button
            className="new-chat-btn"
            onClick={() => { onNewChat(); if (isMobile()) setSidebarOpen(false); }}
          >
            <Plus size={16} />
            New Chat
          </button>

          {sessions.length > 0 && (
            <>
              <div className="sidebar-section-label">Recent</div>
              <div className="sidebar-history">
                {sessions.map((s) => (
                  <button
                    key={s.session_id}
                    className={`history-item ${s.session_id === activeSessionId ? 'history-item--active' : ''}`}
                    onClick={() => { onSelectSession(s.session_id); if (isMobile()) setSidebarOpen(false); }}
                    title={s.title ?? 'Untitled conversation'}
                  >
                    <span className="history-item-text">{s.title ?? 'Untitled conversation'}</span>
                  </button>
                ))}
              </div>
            </>
          )}

          {/* Model + API Key sections — pinned to bottom */}
          <div className="sidebar-spacer" />

          {models.length > 0 && (
            <div className="model-section">
              <div className="sidebar-section-label">Model</div>
              <div className="model-picker" ref={pickerRef}>
                <button
                  type="button"
                  className="model-picker-trigger"
                  onClick={() => setPickerOpen((v) => !v)}
                  aria-haspopup="listbox"
                  aria-expanded={pickerOpen}
                >
                  <span>{model || 'Model'}</span>
                  <ChevronDown size={14} />
                </button>
                {pickerOpen && (
                  <div className="model-picker-menu" role="listbox">
                    {models.map((m) => (
                      <button
                        key={m}
                        type="button"
                        role="option"
                        aria-selected={m === model}
                        className={`model-picker-option${m === model ? ' model-picker-option--active' : ''}`}
                        onClick={() => { onModelChange(m); setPickerOpen(false); }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="api-key-section">
            <div className="sidebar-section-label">OpenAI API Key</div>
            <div className="api-key-input-wrap">
              <input
                type={showKey ? 'text' : 'password'}
                className="api-key-input"
                value={apiKey}
                onChange={(e) => onApiKeyChange(e.target.value)}
                placeholder="Leave empty for system default"
                spellCheck={false}
                autoComplete="off"
              />
              <button
                className="api-key-toggle"
                onClick={() => setShowKey((v) => !v)}
                title={showKey ? 'Hide key' : 'Show key'}
              >
                {showKey ? <EyeOff size={13} /> : <Eye size={13} />}
              </button>
            </div>
            <p className="api-key-hint">
              {apiKey
                ? '✓ Using your custom key'
                : 'Using system default key'}
            </p>
          </div>
        </div>
      </aside>

      <main className="main-view">
        <header className="top-bar">
          <button className="icon-button" onClick={toggleSidebar} aria-label="Toggle sidebar">
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>

          {/* Tab navigation */}
          <nav className="tab-nav">
            <button
              className={`tab-btn ${activeTab === 'chat' ? 'tab-btn--active' : ''}`}
              onClick={() => onTabChange('chat')}
            >
              <MessageSquare size={14} />
              Chat
            </button>
            <button
              className={`tab-btn ${activeTab === 'database' ? 'tab-btn--active' : ''}`}
              onClick={() => onTabChange('database')}
            >
              <Database size={14} />
              Database
            </button>
          </nav>

          {activeTab === 'chat' ? (
            <button
              className="icon-button"
              onClick={onExport}
              disabled={!canExport}
              aria-label="Export conversation"
              title="Export conversation as Markdown"
            >
              <Download size={18} />
            </button>
          ) : (
            <div style={{ width: '2.5rem' }} />
          )}
        </header>

        {children}
      </main>
    </div>
  );
};
