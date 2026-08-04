import React, { useState, useCallback } from 'react';
import { User, Bot, ChevronDown, ChevronRight, Copy, Check, Download, ThumbsUp, ThumbsDown, Zap } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';
import type { Message, TextMessage, TurnUsage } from '../types';
import { downloadUrl } from '../api/client';

interface ChatMessageProps {
  message: Message;
  onFeedback?: (messageId: string, rating: 'up' | 'down') => void;
}

const RECORDS_PREVIEW = 10;

function RecordsPanel({ rows, total }: { rows: Record<string, unknown>[]; total: number }) {
  const [expanded, setExpanded] = useState(false);
  if (!rows.length) return null;

  const headers = Object.keys(rows[0]);
  const visible = expanded ? rows : rows.slice(0, RECORDS_PREVIEW);
  const hidden  = rows.length - RECORDS_PREVIEW;

  return (
    <div className="records-panel">
      <div className="records-header">
        <span className="records-title">Fetched records from database</span>
        <span className="records-count">{total} record{total !== 1 ? 's' : ''}</span>
      </div>
      <div className="records-table-wrap">
        <table className="records-table">
          <thead>
            <tr>
              {headers.map((h) => <th key={h}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {visible.map((row, i) => (
              <tr key={i}>
                {headers.map((h) => (
                  <td key={h}>{row[h] == null ? <span className="null-val">NULL</span> : String(row[h])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > RECORDS_PREVIEW && (
        <button className="records-toggle" onClick={() => setExpanded((v) => !v)}>
          {expanded ? (
            <><ChevronDown size={13} /> Show less</>
          ) : (
            <><ChevronRight size={13} /> Show {hidden} more record{hidden !== 1 ? 's' : ''}</>
          )}
        </button>
      )}
    </div>
  );
}

function SqlPanel({ sql }: { sql: string }) {
  const [expanded, setExpanded] = useState(false);
  const lines = sql.trim().split('\n');

  return (
    <div className="sql-panel">
      <button className="sql-toggle" onClick={() => setExpanded((v) => !v)}>
        {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        <span>SQL Used</span>
        <span className="sql-line-count">{lines.length} line{lines.length !== 1 ? 's' : ''}</span>
      </button>
      <div className={`sql-body ${expanded ? 'sql-body--open' : ''}`}>
        <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
          {`\`\`\`sql\n${sql.trim()}\n\`\`\``}
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
  // The backend normalizes intent to one of the four labels, but never render an
  // arbitrary-length API string: an unnormalized one used to blow up into a
  // full-width all-caps banner. Truncate anything unexpected.
  const known = map[intent.toUpperCase()];
  // Show a truncated fragment of an unexpected value, never the whole string —
  // an unnormalized one used to render as a full-width all-caps banner. The
  // untruncated text stays available on hover.
  const entry = known ?? { label: 'UNKNOWN', cls: 'badge-general' };
  return (
    <span className={`intent-badge ${entry.cls}`} title={known ? entry.label : intent}>
      {entry.label}
    </span>
  );
}

/** "Cached" chip — this turn reused a past validated query instead of generating. */
function CacheChip() {
  return (
    <span className="meta-chip meta-chip--cache" title="Answered from the semantic cache — the stored SQL was re-validated and re-executed, not regenerated">
      <Zap size={11} />
      Cached
    </span>
  );
}

/**
 * Token/cost chip (GH#57). Zero LLM calls is the interesting case — it means the
 * cached answer was reused because the rows hadn't changed — so it gets its own
 * label rather than reading as a suspicious "$0.000000".
 */
function UsageChip({ usage }: { usage: TurnUsage }) {
  const tokens = usage.input_tokens + usage.output_tokens;
  const detail = Object.entries(usage.by_stage ?? {})
    .sort((a, b) => b[1].input_tokens + b[1].output_tokens - (a[1].input_tokens + a[1].output_tokens))
    .map(([stage, s]) => `${stage}: ${(s.input_tokens + s.output_tokens).toLocaleString()} tok`)
    .join('\n');

  if (usage.calls === 0) {
    return (
      <span className="meta-chip meta-chip--free" title="No LLM calls — the stored answer was reused because the rows were unchanged">
        0 LLM calls
      </span>
    );
  }
  return (
    <span
      className="meta-chip"
      title={`${usage.calls} LLM call${usage.calls !== 1 ? 's' : ''}, ` +
             `${usage.input_tokens.toLocaleString()} in / ${usage.output_tokens.toLocaleString()} out` +
             (detail ? `\n\n${detail}` : '')}
    >
      {usage.calls} call{usage.calls !== 1 ? 's' : ''} · {tokens.toLocaleString()} tok · ${usage.est_cost_usd.toFixed(4)}
    </span>
  );
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
          {/* Metadata row: intent badge · timestamp */}
          <div className="msg-meta">
            {message.intent && <IntentBadge intent={message.intent} />}
            {message.cache_hit && <CacheChip />}
            {message.usage && <UsageChip usage={message.usage} />}
            <span className="msg-time">{formatTime(message.timestamp)}</span>
          </div>

          {/* 1. LLM response text */}
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {message.content}
            </ReactMarkdown>
          </div>

          {/* 2. Fetched records — first 10 visible, rest behind toggle */}
          {message.rows && message.rows.length > 0 && (
            <RecordsPanel rows={message.rows} total={message.row_count ?? message.rows.length} />
          )}

          {/* 3. CSV download — only when results exceed 500 rows */}
          {message.csv_path && (
            <a
              className="download-btn"
              href={downloadUrl(message.csv_path)}
              download={message.csv_path}
            >
              <Download size={14} />
              Download full CSV ({message.row_count?.toLocaleString()} records)
            </a>
          )}

          {/* 4. Toggleable SQL Used */}
          {message.sql && <SqlPanel sql={message.sql} />}

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

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, onFeedback }) => {
  const isBot = message.role === 'assistant';

  if (isBot) {
    return <AssistantMessage message={message} onFeedback={onFeedback} />;
  }

  return (
    <div className="message-wrapper message-user">
      <div className="message-inner">
        {/* No w-full here: the bubble must shrink to its content so it sits
            beside its avatar instead of stretching the whole column. */}
        <div className="message-content">
          {/* Timestamp for user messages */}
          <div className="msg-meta msg-meta--right">
            <span className="msg-time">{formatTime(message.timestamp)}</span>
          </div>

          <div>{message.content}</div>
        </div>

        <div className="avatar" style={{ backgroundColor: '#4b5563' }}>
          <User size={16} />
        </div>
      </div>
    </div>
  );
};
