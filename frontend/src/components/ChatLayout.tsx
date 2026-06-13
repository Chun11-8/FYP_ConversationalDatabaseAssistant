import React, { useState } from 'react';
import type { ReactNode } from 'react';
import { Plus, Menu, X, Database, MessageSquare, Eye, EyeOff } from 'lucide-react';

export type AppTab = 'chat' | 'database';

interface ChatLayoutProps {
  children: ReactNode;
  onNewChat: () => void;
  chatHistory: { id: string; title: string }[];
  activeTab: AppTab;
  onTabChange: (tab: AppTab) => void;
  apiKey: string;
  onApiKeyChange: (key: string) => void;
}

export const ChatLayout: React.FC<ChatLayoutProps> = ({
  children,
  onNewChat,
  chatHistory,
  activeTab,
  onTabChange,
  apiKey,
  onApiKeyChange,
}) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [showKey, setShowKey] = useState(false);

  const toggleSidebar = () => setIsSidebarOpen((v) => !v);

  return (
    <div className="app-layout">
      {isSidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setIsSidebarOpen(false)} />
      )}

      <aside className={`sidebar ${isSidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-content">
          {/* Branding */}
          <div className="sidebar-brand">
            <Database size={18} className="sidebar-brand-icon" />
            <span className="sidebar-brand-name">DB Assistant</span>
          </div>

          <button
            className="new-chat-btn"
            onClick={() => { onNewChat(); setIsSidebarOpen(false); }}
          >
            <Plus size={16} />
            New Chat
          </button>

          {chatHistory.length > 0 && (
            <>
              <div className="sidebar-section-label">Recent</div>
              <div className="sidebar-history">
                {chatHistory.map((chat) => (
                  <button key={chat.id} className="history-item">
                    <span className="history-item-text">{chat.title}</span>
                  </button>
                ))}
              </div>
            </>
          )}

          {/* API Key section — pinned to bottom */}
          <div className="sidebar-spacer" />
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
            {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
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

          <div style={{ width: '2.5rem' }} />
        </header>

        {children}
      </main>
    </div>
  );
};
