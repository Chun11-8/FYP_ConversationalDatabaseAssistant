import React, { useState, useRef, useEffect } from 'react';
import { ArrowUp, ChevronDown } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  models: string[];
  model: string;
  onModelChange: (model: string) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend, disabled, models, model, onModelChange,
}) => {
  const [input, setInput] = useState('');
  const [pickerOpen, setPickerOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const pickerRef = useRef<HTMLDivElement>(null);

  const handleSubmit = () => {
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = '44px';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '44px';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 200)}px`;
    }
  }, [input]);

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
    <div className="input-area">
      <div className="input-container">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your data…"
          className="input-textarea"
          disabled={disabled}
          rows={1}
        />
        <button
          className="send-btn"
          onClick={handleSubmit}
          disabled={!input.trim() || disabled}
          aria-label="Send message"
        >
          <ArrowUp size={20} strokeWidth={3} />
        </button>
      </div>
      <div className="input-footer">
        {models.length > 0 && (
          <div className="model-picker" ref={pickerRef}>
            <button
              type="button"
              className="model-picker-trigger"
              onClick={() => setPickerOpen((v) => !v)}
              aria-haspopup="listbox"
              aria-expanded={pickerOpen}
            >
              {model || 'Model'}
              <ChevronDown size={12} />
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
        )}
        <div className="input-hint">
          Results are read-only. Always verify critical data.
        </div>
      </div>
    </div>
  );
};
