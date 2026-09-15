import React, { useState, useCallback } from 'react';
import { User, Bot, ChevronDown, ChevronRight, Copy, Check, Download, ThumbsUp, ThumbsDown, Zap, ShieldCheck, AlertTriangle, ShieldAlert, ShieldX, CircleHelp, MessageSquareQuote, Database } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';
import type { Message, TextMessage, TurnUsage, TurnVerification, TurnRisk, TurnTrust, TrustGate } from '../types';
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
  // `rows` is only what the backend chose to send inline — large results are
  // truncated before they leave the server, while `total` (message.row_count)
  // is always the true count. The toggle must promise only what `rows` can
  // actually reveal (rows.length), never `total` — promising more would open
  // an "expanded" view still missing data with no further control to get it.
  const hidden = rows.length - RECORDS_PREVIEW;
  const truncated = rows.length < total;

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
      {truncated && (
        <p className="records-truncated-hint">
          Showing {rows.length.toLocaleString()} of {total.toLocaleString()} rows inline —
          download the CSV below for the rest.
        </p>
      )}
    </div>
  );
}

/**
 * The query ran and matched nothing (GH#113).
 *
 * Rendering nothing here was wrong: an empty result is a FACT about the data,
 * and `zero_rows` raises a risk flag, so the user saw a warning with no table
 * and reasonably read it as "the query never executed". Naming the outcome
 * separates "no matching rows" from "no query ran".
 */
function EmptyResultPanel() {
  return (
    <div className="records-empty">
      <Database size={13} className="records-empty-icon" />
      <span>
        <strong>The query ran and returned 0 rows.</strong> That is a result,
        not a failure — either nothing in the data matches, or a filter is
        wrong. The SQL is below so you can check which.
      </span>
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
  // The backend normalizes intent to one of the four labels (see
  // _normalize_intent in db_agent.py), but this must never trust that and
  // render an arbitrary-length API string directly: an unnormalized value
  // used to blow up into a full-width all-caps banner. An unrecognized value
  // renders as the fixed label "UNKNOWN" — never a fragment of the original
  // string, which could itself be unbounded — with the real value available
  // on hover via `title`.
  const known = map[intent.toUpperCase()];
  const entry = known ?? { label: 'UNKNOWN', cls: 'badge-general' };
  return (
    // `title` only adds information for the fallback case; on a known intent
    // it would just repeat the visible label.
    <span className={`intent-badge ${entry.cls}`} title={known ? undefined : intent}>
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

/**
 * How the winner was decided, in the user's words rather than the enum's.
 * `null` for turns where nothing was chosen (cache reuse, non-READ turns) —
 * the caller renders nothing at all rather than an empty or confusing chip.
 */
const DECIDED_BY_LABEL: Record<string, string> = {
  plurality:          'agreed on this result',
  non_empty_tiebreak: 'tie broken toward the non-empty result',
  unit_test:          'chosen by unit-test screen',
  pairwise:           'chosen by pairwise comparison',
  reranker:           'no execution agreement — chosen by text reranker',
  inconclusive:       'no execution agreement',
  single_candidate:   'only one candidate generated',
  voting_disabled:    'execution voting disabled',
  none_executed:      'no candidate executed successfully',
  generation_failed:  'SQL generation failed',
};

/**
 * Agreement chip (GH#67). Reports what happened — "4/5 agreed" — and
 * deliberately NOT a High/Medium/Low verdict: whether agreement predicts
 * correctness is exactly what the calibration study has yet to measure, and a
 * miscalibrated badge would teach the user to trust the wrong answers.
 */
function AgreementChip({ v }: { v: TurnVerification }) {
  const label = DECIDED_BY_LABEL[v.decided_by];
  // Nothing was chosen this turn — a cached answer or a non-READ turn.
  if (!label) return null;

  const voted = v.executed > 0 && v.agreement > 0;
  // A lone dissenting result set is the signal worth seeing at a glance.
  const split = voted && v.distinct_result_sets > 1;
  const weak = !voted || v.decided_by === 'reranker' || v.decided_by === 'none_executed';

  return (
    <span
      className={`meta-chip${split ? ' meta-chip--split' : ''}${weak ? ' meta-chip--weak' : ''}`}
      title={
        `${v.candidates} candidate${v.candidates !== 1 ? 's' : ''} generated, ` +
        `${v.executed} executed — ${label}.` +
        (v.distinct_result_sets > 1
          ? `\n${v.distinct_result_sets} different result sets were produced.`
          : '') +
        (v.refine_rounds ? `\nRefined ${v.refine_rounds}× after a failed attempt.` : '')
      }
    >
      <ShieldCheck size={11} />
      {voted ? `${v.agreement}/${v.executed} agreed` : label}
    </span>
  );
}

/**
 * Risk flags (GH#69). Shown OUTSIDE the collapsed panel, unlike the rest of
 * the evidence: a specific, checkable objection to the answer the user is
 * reading is worth interrupting for, where "5/5 agreed" is not.
 *
 * Never phrased as a verdict on the whole answer — each line states the one
 * concrete thing that looks wrong, so the user can judge it themselves.
 */
function RiskNotes({ risks }: { risks: TurnRisk[] }) {
  if (!risks.length) return null;
  return (
    <ul className="risk-notes">
      {risks.map((r, i) => (
        <li key={`${r.code}-${i}`} className={`risk-note risk-note--${r.severity}`}>
          <AlertTriangle size={12} className="risk-note-icon" />
          <span>{r.message}</span>
        </li>
      ))}
    </ul>
  );
}

const TRUST_ICON = {
  passed: ShieldCheck,
  review: ShieldAlert,
  untrusted: ShieldX,
  not_graded: CircleHelp,
} as const;

const GATE_MARK: Record<TrustGate['status'], string> = {
  ok: '✓', warn: '!', fail: '✕', skipped: '–',
};

/**
 * The verdict (GH#91). The one thing on screen that answers "can I use this?".
 *
 * Deliberately carries no percentage: every signal behind it is a check on
 * THIS query, and a hit-rate measured on benchmark databases would say nothing
 * about a question over the user's own data. The top level is worded as
 * "checks passed", never "correct" — they are necessary conditions, not
 * sufficient ones, and the statement says so.
 */
function TrustVerdict({ trust, executed }: { trust: TurnTrust; executed: boolean }) {
  const [open, setOpen] = useState(trust.level !== 'passed');
  if (trust.level === 'not_graded') return null;
  const Icon = TRUST_ICON[trust.level];
  const graded = trust.gates.filter((g) => g.status !== 'skipped');
  const passed = graded.filter((g) => g.status === 'ok').length;

  return (
    <div className={`trust trust--${trust.level}`}>
      <button
        className="trust-head"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <Icon size={15} className="trust-icon" />
        <span className="trust-headline">{trust.headline}</span>
        <span className="trust-count">{passed}/{graded.length} checks passed</span>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
      </button>

      <p className="trust-statement">{trust.statement}</p>

      {/* GH#113: the verdict is an opinion, not a gate. When the query ran,
          say plainly that the data is still there — otherwise a red banner
          reads as "your results were withheld". */}
      {executed && trust.level !== 'passed' && (
        <p className="trust-still-shown">
          The query ran, so the results are shown below either way — judge them
          against the checks.
        </p>
      )}

      {open && (
        <ul className="trust-gates">
          {trust.gates.map((g) => (
            <li key={g.id} className={`trust-gate trust-gate--${g.status}`}>
              <span className="trust-gate-mark" aria-hidden>{GATE_MARK[g.status]}</span>
              <div>
                <span className="trust-gate-label">{g.label}</span>
                <span className="trust-gate-detail">{g.detail}</span>
                {g.remedy && g.status !== 'ok' && (
                  <span className="trust-gate-remedy">Try: {g.remedy}</span>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * The SQL read back into plain English (GH#91), generated without showing the
 * model the question. Shown whether or not the comparison reached a verdict:
 * a user who reads "the lowest single monthly reading" against a question
 * that meant "the lowest yearly total" needs no verdict at all.
 */
function ReadsAs({ text }: { text: string }) {
  return (
    <div className="reads-as">
      <MessageSquareQuote size={13} className="reads-as-icon" />
      <span><strong>This query returns:</strong> {text}</span>
    </div>
  );
}

/** Collapsed "why this answer" panel — the full evidence, on demand. */
function EvidencePanel({ v }: { v: TurnVerification }) {
  const [open, setOpen] = useState(false);
  if (!DECIDED_BY_LABEL[v.decided_by]) return null;

  const checks = Object.entries(v.validation_checks ?? {});
  return (
    <div className="evidence-panel">
      <button
        className="evidence-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        Why this answer
      </button>
      {open && (
        <dl className="evidence-body">
          <dt>Candidates</dt>
          <dd>{v.candidates} generated · {v.executed} executed successfully</dd>
          <dt>Agreement</dt>
          <dd>
            {v.executed > 0
              ? `${v.agreement} of ${v.executed} produced the same result set`
              : 'no candidate executed'}
            {v.distinct_result_sets > 1 &&
              ` · ${v.distinct_result_sets} distinct result sets`}
          </dd>
          <dt>Decided by</dt>
          <dd>{DECIDED_BY_LABEL[v.decided_by]}</dd>
          {v.topic_switch && (
            <>
              <dt>Context</dt>
              <dd>
                Answered on its own — this question is about different tables
                from the previous one, so the earlier conversation was not used.
              </dd>
            </>
          )}
          {Boolean(v.refine_rounds) && (
            <>
              <dt>Refine rounds</dt>
              <dd>{v.refine_rounds}</dd>
            </>
          )}
          {checks.length > 0 && (
            <>
              <dt>Validation</dt>
              <dd>
                {checks.map(([k, ok]) => (
                  <span key={k} className={`evidence-check${ok ? '' : ' evidence-check--fail'}`}>
                    {ok ? '✓' : '✗'} {k.replace(/_ok$/, '')}
                  </span>
                ))}
              </dd>
            </>
          )}
        </dl>
      )}
    </div>
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
            {message.verification && <AgreementChip v={message.verification} />}
            {message.usage && <UsageChip usage={message.usage} />}
            <span className="msg-time">{formatTime(message.timestamp)}</span>
          </div>

          {/* 1. LLM response text */}
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {message.content}
            </ReactMarkdown>
          </div>

          {/* 1b. The verdict (GH#91) — before the rows it judges, because
                 "can I use this?" is the first question a reader has. */}
          {message.verification?.trust && (
            <TrustVerdict
              trust={message.verification.trust}
              executed={message.row_count !== null && message.row_count !== undefined}
            />
          )}

          {/* 1c. Risk flags (GH#69) — the specific objections behind a
                 non-passing verdict, each one checkable on its own. */}
          {message.verification?.risks?.length ? (
            <RiskNotes risks={message.verification.risks} />
          ) : null}

          {/* 2. Fetched records — first 10 visible, rest behind toggle.
                 GH#113: deliberately NOT gated on the trust verdict. If the
                 query executed, the user gets the data and judges it against
                 the checks above; withholding results would make the verdict
                 a gatekeeper rather than an opinion. */}
          {message.rows && message.rows.length > 0 && (
            <RecordsPanel rows={message.rows} total={message.row_count ?? message.rows.length} />
          )}
          {/* row_count === 0 means it RAN and matched nothing; null means no
              query ran at all (refused, blocked, or a general answer). */}
          {message.row_count === 0 && <EmptyResultPanel />}

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

          {/* 4. Toggleable SQL Used, with its plain-English read-back (GH#91)
                 directly beneath — the two belong together. */}
          {message.sql && <SqlPanel sql={message.sql} />}
          {message.sql && message.verification?.back_translation?.description && (
            <ReadsAs text={message.verification.back_translation.description} />
          )}

          {/* 5. Why this answer (GH#67) — sits under the SQL, since it explains
                 how that particular query came to be chosen. */}
          {message.verification && <EvidencePanel v={message.verification} />}

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
