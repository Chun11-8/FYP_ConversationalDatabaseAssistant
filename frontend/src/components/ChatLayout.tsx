import React, { useState } from 'react';
import type { ReactNode } from 'react';
import { Plus, Menu, X, Database } from 'lucide-react';

interface ChatLayoutProps {
  children: ReactNode;
  onNewChat: () => void;
  chatHistory: { id: string; title: string }[];
}

export const ChatLayout: React.FC<ChatLayoutProps> = ({ children, onNewChat, chatHistory }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

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
        </div>
      </aside>

      <main className="main-view">
        <header className="top-bar">
          <button className="icon-button" onClick={toggleSidebar} aria-label="Toggle sidebar">
            {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
          <div className="top-bar-title">
            <Database size={16} />
            <span>Database Assistant</span>
          </div>
          <div style={{ width: '2.5rem' }} />
        </header>

        {children}
      </main>
    </div>
  );
};
