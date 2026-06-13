import React, { useState, useCallback } from 'react';
import { User, Bot, ChevronDown, ChevronRight, Copy, Check, Download, ThumbsUp, ThumbsDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';
import type { Message, McqOption, TextMessage } from '../types';
import { downloadUrl } from '../api/client';

interface ChatMessageProps {
  message: Message;
  onMcqSelect?: (messageId: string, option: McqOption) => void;
  onFeedback?: (messageId: string, rating: 'up' | 'down') => void;
}

const SQL_PREVIEW_LINES = 3;

function SqlPanel({ sql }: { sql: string }) {
  const lines = sql.trim().split('\n');
  const isLong = lines.length > SQL_PREVIEW_LINES;
  const [expanded, setExpanded] = useState(!isLong);
  const preview = isLong ? lines.slice(0, SQL_PREVIEW_LINES).join('\n') + '\n…' : sql;

  return (
    <div className="sql-panel">
      <button className="sql-toggle" onClick={() => setExpanded((v) => !v)}>
        {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        <span>SQL used</span>
        {isLong && !expanded && (
          <span className="sql-line-count">{lines.length} lines</span>
        )}
      </button>
      <div className={`sql-body ${expanded ? 'sql-body--open' : ''}`}>
        <ReactMarkdown
          rehypePlugins={[rehypeHighlight]}
        >
          {`\`\`\`sql\n${expanded ? sql.trim() : preview}\n\`\`\``}
        </ReactMarkdown>
      </div>
    </div>
  );
}

function IntentBadge({ intent }: { intent: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    READ:          { label: 'READ',          cls: 'badge-read' },
    GENERAL:       { label: 'GENERAL',       cls: 'badge-general' },
    WRITE_BLOCKED: { label: 'WRITE BLOCKED', cls: 'badge-blocked' },
    AMBIGUOUS:     { label: 'AMBIGUOUS',     cls: 'badge-ambiguous' },
  };
  const entry = map[intent.toUpperCase()] ?? { label: intent, cls: 'badge-general' };
  return <span className={`intent-badge ${entry.cls}`}>{entry.label}</span>;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [text]);
  return (
    <button className="icon-btn" onClick={handleCopy} title="Copy response">
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

function formatTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function FeedbackButtons({
  messageId,
  current,
  onFeedback,
}: {
  messageId: string;
  current: 'up' | 'down' | null | undefined;
  onFeedback: (id: string, rating: 'up' | 'down') => void;
}) {
  const rated = current != null;
  return (
    <div className="feedback-btns">
      <button
        className={`feedback-btn ${current === 'up' ? 'feedback-btn--active-up' : ''}`}
        onClick={() => !rated && onFeedback(messageId, 'up')}
        disabled={rated}
        title="Good response"
        aria-label="Thumbs up"
      >
        <ThumbsUp size={13} />
      </button>
      <button
        className={`feedback-btn ${current === 'down' ? 'feedback-btn--active-down' : ''}`}
        onClick={() => !rated && onFeedback(messageId, 'down')}
        disabled={rated}
        title="Poor response"
        aria-label="Thumbs down"
      >
        <ThumbsDown size={13} />
      </button>
    </div>
  );
}

function AssistantMessage({
  message,
  onFeedback,
}: {
  message: TextMessage;
  onFeedback?: (id: string, rating: 'up' | 'down') => void;
}) {
  return (
    <div className="message-wrapper message-bot">
      <div className="message-inner">
        <div className="avatar avatar-bot">
          <Bot size={16} />
        </div>

        <div className="message-content w-full">
          {/* Metadata row */}
          <div className="msg-meta">
            {message.intent && <IntentBadge intent={message.intent} />}
            {message.row_count != null && (
              <span className="row-count-chip">{message.row_count} row{message.row_count !== 1 ? 's' : ''}</span>
            )}
            <span className="msg-time">{formatTime(message.timestamp)}</span>
          </div>

          {/* Markdown body */}
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {message.content}
            </ReactMarkdown>
          </div>

          {/* SQL collapsible panel */}
          {message.sql && <SqlPanel sql={message.sql} />}

          {/* CSV download */}
          {message.csv_path && (
            <a
              className="download-btn"
              href={downloadUrl(message.csv_path)}
              download={message.csv_path}
            >
              <Download size={14} />
              Download CSV
            </a>
          )}

          {/* Action bar */}
          <div className="msg-actions">
            <CopyButton text={message.content} />
            {onFeedback && message.turn_id != null && (
              <FeedbackButtons
                messageId={message.id}
                current={message.feedback}
                onFeedback={onFeedback}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, onMcqSelect, onFeedback }) => {
  const isBot = message.role === 'assistant';

  if (isBot && message.type === 'text') {
    return <AssistantMessage message={message} onFeedback={onFeedback} />;
  }

  return (
    <div className={`message-wrapper ${isBot ? 'message-bot' : 'message-user'}`}>
      <div className="message-inner">
        {isBot && (
          <div className="avatar avatar-bot">
            <Bot size={16} />
          </div>
        )}

        <div className="message-content w-full">
          {/* Timestamp for user messages */}
          {!isBot && (
            <div className="msg-meta msg-meta--right">
              <span className="msg-time">{formatTime(message.timestamp)}</span>
            </div>
          )}

          {message.type === 'text' && <div>{message.content}</div>}

          {message.type === 'mcq' && (
            <>
              <div>{message.question}</div>
              <div className="mcq-container">
                {message.options.map((option) => {
                  const isSelected = message.selectedOptionId === option.id;
                  const isDisabled = message.selectedOptionId !== undefined;
                  return (
                    <button
                      key={option.id}
                      className={`mcq-option ${isSelected ? 'selected' : ''}`}
                      disabled={isDisabled}
                      onClick={() => !isDisabled && onMcqSelect?.(message.id, option)}
                    >
                      {option.text}
                    </button>
                  );
                })}
              </div>
            </>
          )}
        </div>

        {!isBot && (
          <div className="avatar" style={{ backgroundColor: '#4b5563' }}>
            <User size={16} />
          </div>
        )}
      </div>
    </div>
  );
};
