import React, { ReactNode, useState } from 'react';
import { Plus, Menu, X, MessageSquare } from 'lucide-react';

interface ChatLayoutProps {
  children: ReactNode;
  onNewChat: () => void;
  chatHistory: { id: string; title: string }[];
}

export const ChatLayout: React.FC<ChatLayoutProps> = ({ children, onNewChat, chatHistory }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  return (
    <div className="app-layout">
      {/* Mobile Sidebar Overlay */}
      {isSidebarOpen && (
        <div 
          className="sidebar-overlay" 
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`sidebar ${isSidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-content">
          <div className="flex justify-between items-center" style={{ marginBottom: '1.5rem' }}>
            <button className="new-chat-btn" onClick={() => {
              onNewChat();
              setIsSidebarOpen(false);
            }}>
              <div className="flex items-center" style={{ gap: '0.5rem' }}>
                <Plus size={18} />
                New Chat
              </div>
            </button>
            
            {/* Mobile close button */}
            <button 
              className="icon-button" 
              onClick={() => setIsSidebarOpen(false)}
              style={{ display: isSidebarOpen ? 'flex' : 'none' }}
              aria-label="Close sidebar"
            >
              <X size={20} />
            </button>
          </div>

          <div style={{ padding: '0 0.5rem', fontSize: '0.75rem', fontWeight: '600', color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>
            Recent
          </div>
          
          <div className="flex-col" style={{ gap: '0.25rem', overflowY: 'auto' }}>
            {chatHistory.map((chat) => (
              <button 
                key={chat.id} 
                className="new-chat-btn" 
                style={{ justifyContent: 'flex-start', padding: '0.5rem', fontSize: '0.875rem', fontWeight: 'normal' }}
              >
                <div className="flex items-center" style={{ gap: '0.5rem', overflow: 'hidden' }}>
                  <MessageSquare size={16} style={{ flexShrink: 0 }} />
                  <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {chat.title}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-view">
        {/* Mobile Header (Only visible on small screens typically, but useful for sidebar toggle) */}
        <header className="top-bar">
          <button 
            className="icon-button" 
            onClick={toggleSidebar}
            aria-label="Open sidebar"
          >
            <Menu size={24} />
          </button>
          <div style={{ margin: '0 auto', fontWeight: '500' }}>Conversational UI</div>
          <div style={{ width: '2.5rem' }}></div> {/* Spacer for centering */}
        </header>

        {children}
      </main>
    </div>
  );
};
