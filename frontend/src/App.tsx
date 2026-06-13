import { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import type { Message, McqOption, McqMessage, TextMessage } from './types';
import { ChatLayout } from './components/ChatLayout';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';

const SIMULATED_DELAY = 1000;

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const addBotMessage = (messageDetails: Omit<TextMessage, 'id' | 'role' | 'timestamp'> | Omit<McqMessage, 'id' | 'role' | 'timestamp'>) => {
    setIsTyping(true);
    setTimeout(() => {
      const botMsg: Message = {
        id: uuidv4(),
        role: 'assistant',
        timestamp: Date.now(),
        ...messageDetails,
      } as Message;
      
      setMessages((prev) => [...prev, botMsg]);
      setIsTyping(false);
    }, SIMULATED_DELAY);
  };

  const handleSend = (content: string) => {
    const userMsg: Message = {
      id: uuidv4(),
      role: 'user',
      timestamp: Date.now(),
      type: 'text',
      content,
    };
    
    setMessages((prev) => [...prev, userMsg]);

    if (messages.length === 0) {
      addBotMessage({
        type: 'text',
        content: `Thanks for reaching out! You said: "${content}". How can I help you today? Please choose an area below:`,
      });
      setTimeout(() => {
        addBotMessage({
          type: 'mcq',
          question: 'Select a topic to focus on:',
          options: [
            { id: 'opt1', text: '🎨 Frontend UI Development' },
            { id: 'opt2', text: '⚙️ Backend & APIs' },
            { id: 'opt3', text: '🗄️ Database Architecture' },
            { id: 'opt4', text: '🚀 CI/CD & Deployment' },
          ]
        });
      }, SIMULATED_DELAY + 400);
    } else {
      addBotMessage({
        type: 'text',
        content: `I've noted your input: "${content}". This is a responsive test to ensure the UI handles extensive text efficiently. The layout should now dynamically wrap and scroll perfectly regardless of device size.`,
      });
    }
  };

  const handleMcqSelect = (messageId: string, option: McqOption) => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId ? { ...msg, selectedOptionId: option.id } as McqMessage : msg
      )
    );
    
    const userMsg: Message = {
      id: uuidv4(),
      role: 'user',
      timestamp: Date.now(),
      type: 'text',
      content: option.text,
    };
    
    setTimeout(() => {
       setMessages((prev) => [...prev, userMsg]);
       
       addBotMessage({
         type: 'text',
         content: `Great choice! Focusing on "${option.text}" will be excellent. Is there a specific framework or tool you'd like to dive into?`
       });
    }, 400);
  };

  const handleNewChat = () => {
    setMessages([]);
  };

  return (
    <ChatLayout
      onNewChat={handleNewChat}
      chatHistory={[{ id: '1', title: 'Example Conversation' }, { id: '2', title: 'React Hooks Deep Dive' }]}
    >
      {messages.length === 0 ? (
        <div className="welcome-screen">
          <div className="avatar avatar-bot" style={{ width: '4rem', height: '4rem', marginBottom: '1rem', borderRadius: '50%' }}>
            <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="2rem" width="2rem" xmlns="http://www.w3.org/2000/svg"><path d="M12 8V4H8"></path><rect width="16" height="12" x="4" y="8" rx="2"></rect><path d="M2 14h2"></path><path d="M20 14h2"></path><path d="M15 13v2"></path><path d="M9 13v2"></path></svg>
          </div>
          <h1 className="welcome-title">How can I help you today?</h1>
          <p className="welcome-subtitle">
            I'm a highly dynamic and responsive conversational assistant.
          </p>
        </div>
      ) : (
        <div className="chat-scroll">
          <div className="chat-container">
            {messages.map((msg) => (
              <ChatMessage
                key={msg.id}
                message={msg}
                onMcqSelect={handleMcqSelect}
              />
            ))}
            {isTyping && (
              <div className="message-wrapper message-bot">
                 <div className="message-inner">
                   <div className="avatar avatar-bot">
                     <svg stroke="currentColor" fill="none" strokeWidth="2" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round" height="18" width="18" xmlns="http://www.w3.org/2000/svg"><path d="M12 8V4H8"></path><rect width="16" height="12" x="4" y="8" rx="2"></rect><path d="M2 14h2"></path><path d="M20 14h2"></path><path d="M15 13v2"></path><path d="M9 13v2"></path></svg>
                   </div>
                   <div className="message-content w-full" style={{ paddingTop: '0.25rem' }}>
                      <div style={{ display: 'flex', gap: '4px', alignItems: 'center', height: '1.5rem' }}>
                        <div style={{ width: '6px', height: '6px', backgroundColor: 'var(--text-secondary)', borderRadius: '50%', animation: 'slideUp 0.5s infinite alternate' }} />
                        <div style={{ width: '6px', height: '6px', backgroundColor: 'var(--text-secondary)', borderRadius: '50%', animation: 'slideUp 0.5s infinite alternate 0.2s' }} />
                        <div style={{ width: '6px', height: '6px', backgroundColor: 'var(--text-secondary)', borderRadius: '50%', animation: 'slideUp 0.5s infinite alternate 0.4s' }} />
                      </div>
                   </div>
                 </div>
              </div>
            )}
            <div ref={messagesEndRef} style={{ height: '1px' }} />
          </div>
        </div>
      )}

      <ChatInput onSend={handleSend} disabled={isTyping} />
    </ChatLayout>
  );
}

export default App;
