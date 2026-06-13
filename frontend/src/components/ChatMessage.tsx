import React from 'react';
import { User, Bot } from 'lucide-react';
import type { Message, McqOption } from '../types';

interface ChatMessageProps {
  message: Message;
  onMcqSelect?: (messageId: string, option: McqOption) => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, onMcqSelect }) => {
  const isBot = message.role === 'assistant';

  return (
    <div className={`message-wrapper ${isBot ? 'message-bot' : 'message-user'}`}>
      <div className="message-inner">
        
        {/* Avatar (only show for bot typically in this style, but let's show both) */}
        {isBot && (
          <div className="avatar avatar-bot">
            <Bot size={18} />
          </div>
        )}
        
        <div className="message-content w-full">
          {message.type === 'text' && (
            <div>
              {message.content}
            </div>
          )}
          
          {message.type === 'mcq' && (
            <>
              <div>
                {message.question}
              </div>
              <div className="mcq-container">
                {message.options.map((option) => {
                  const isSelected = message.selectedOptionId === option.id;
                  const isDisabled = message.selectedOptionId !== undefined;
                  
                  return (
                    <button
                      key={option.id}
                      className={`mcq-option ${isSelected ? 'selected' : ''}`}
                      disabled={isDisabled}
                      onClick={() => {
                        if (!isDisabled && onMcqSelect) {
                          onMcqSelect(message.id, option);
                        }
                      }}
                    >
                      {option.text}
                    </button>
                  );
                })}
              </div>
            </>
          )}
        </div>

        {/* User avatar on right */}
        {!isBot && (
          <div className="avatar" style={{ backgroundColor: '#4b5563' }}>
            <User size={18} />
          </div>
        )}

      </div>
    </div>
  );
};
