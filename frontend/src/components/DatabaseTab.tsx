import { useEffect, useState, useCallback, useMemo } from 'react';
import { ChevronRight, ChevronDown, Key, Link, RefreshCw, Info, Database } from 'lucide-react';
import type { SchemaInfo, SchemaTable, SchemaColumn } from '../types';
import { getSchema, getSchemaDiff, syncSchema } from '../api/client';
import type { SchemaDiff } from '../api/client';

function ColumnRow({ col }: { col: SchemaColumn }) {
  const [showDesc, setShowDesc] = useState(false);
  const hasDesc = Boolean(col.description || col.value_description);

  return (
    <div className="schema-column-wrap">
      <div
        className={`schema-column ${hasDesc ? 'schema-column--has-desc' : ''}`}
        onClick={hasDesc ? () => setShowDesc((v) => !v) : undefined}
        role={hasDesc ? 'button' : undefined}
        title={hasDesc ? (showDesc ? 'Hide description' : 'Show description') : undefined}
      >
        <span className="schema-col-name">{col.name}</span>
        <span className="schema-col-type">{col.type}</span>
        <span className="schema-col-badges">
          {col.primary_key && (
            <span className="col-badge col-badge--pk" title="Primary Key">
              <Key size={9} /> PK
            </span>
          )}
          {col.foreign_key && (
            <span className="col-badge col-badge--fk" title={`→ ${col.foreign_key}`}>
              <Link size={9} /> FK
            </span>
          )}
          {col.not_null && !col.primary_key && (
            <span className="col-badge col-badge--nn" title="Not Null">NN</span>
          )}
          {hasDesc && (
            <span className={`col-desc-toggle ${showDesc ? 'col-desc-toggle--open' : ''}`}>
              <Info size={11} />
            </span>
          )}
        </span>
      </div>

      {hasDesc && showDesc && (
        <div className="schema-col-desc">
          {col.description && <p className="col-desc-text">{col.description}</p>}
          {col.value_description && (
            <p className="col-desc-values">
              <span className="col-desc-label">Values:</span> {col.value_description}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function TableRow({ table }: { table: SchemaTable }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="schema-table">
      <button className="schema-table-header" onClick={() => setOpen((v) => !v)}>
        <span className="schema-table-toggle">
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
        <span className="schema-table-name">{table.table_name}</span>
        <span className="schema-table-count">{table.column_count} cols</span>
      </button>

      {open && (
        <div className="schema-columns">
          {table.columns.map((col) => (
            <ColumnRow key={col.name} col={col} />
          ))}
        </div>
      )}
    </div>
  );
}

/** Tables that carry no domain tag — shown under this heading. */
const UNGROUPED = 'Other tables';

/**
 * 75 tables from 11 unrelated databases share one PostgreSQL schema,
 * so a flat list buries the 5 tables a question is actually about. Group by
 * the logical database that owns each table; the group matching the active
 * retrieval scope opens by default because that is the one being queried.
 */
function DomainGroup({
  domain, tables, active,
}: { domain: string; tables: SchemaTable[]; active: boolean }) {
  const [open, setOpen] = useState(active);

  return (
    <div className={`schema-domain${active ? ' schema-domain--active' : ''}`}>
      <button className="schema-domain-header" onClick={() => setOpen((v) => !v)}>
        <span className="schema-table-toggle">
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
        <Database size={13} className="schema-domain-icon" />
        <span className="schema-domain-name">{domain}</span>
        {active && (
          <span className="schema-domain-badge" title="Questions are answered from this database">
            answering
          </span>
        )}
        <span className="schema-table-count">
          {tables.length} table{tables.length !== 1 ? 's' : ''}
        </span>
      </button>

      {open && (
        <div className="schema-domain-tables">
          {tables.map((t) => <TableRow key={t.table_name} table={t} />)}
        </div>
      )}
    </div>
  );
}

/** Stable {domain: tables} grouping — insertion order follows the API's. */
function groupByDomain(tables: SchemaTable[]): [string, SchemaTable[]][] {
  const groups = new Map<string, SchemaTable[]>();
  for (const t of tables) {
    const key = t.domain || UNGROUPED;
    const bucket = groups.get(key);
    if (bucket) bucket.push(t);
    else groups.set(key, [t]);
  }
  // Untagged tables last; everything else alphabetical.
  return [...groups.entries()].sort(([a], [b]) => {
    if (a === UNGROUPED) return 1;
    if (b === UNGROUPED) return -1;
    return a.localeCompare(b);
  });
}

const CHANGE_META: Partial<Record<string, { label: string; cls: string }>> = {
  add_table:     { label: '+ table',  cls: 'chg-add' },
  drop_table:    { label: '− table',  cls: 'chg-drop' },
  add_column:    { label: '+ column', cls: 'chg-add' },
  drop_column:   { label: '− column', cls: 'chg-drop' },
  modify_column: { label: '~ type',   cls: 'chg-mod' },
};
// `SchemaChange['action']` is a closed union in TypeScript, but that's a
// compile-time promise only — the backend's own type is a plain unconstrained
// `str` (backend/api/models.py), so a value outside the five known actions can
// arrive over the wire at runtime and must degrade to a label, not throw.
function changeMeta(action: string): { label: string; cls: string } {
  return CHANGE_META[action] ?? { label: action, cls: 'chg-unknown' };
}

function IndexSyncPanel({
  onSynced,
  message,
}: {
  // Fires with the success message text; the caller (DatabaseTab) owns the
  // message so it survives this panel's own unmount/remount when a sync
  // triggers the parent's schema reload (see `message` prop below).
  onSynced: (message: string) => void;
  message: string | null;
}) {
  const [diff, setDiff] = useState<SchemaDiff | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const check = useCallback(() => {
    setBusy(true); setErr(null);
    getSchemaDiff()
      .then(setDiff)
      .catch((e) => setErr(e instanceof Error ? e.message : 'Check failed'))
      .finally(() => setBusy(false));
  }, []);

  useEffect(() => { check(); }, [check]);

  const runSync = (full: boolean) => {
    setBusy(true); setErr(null);
    syncSchema(full)
      .then((r) => {
        const n = r.reembedded_tables.length, d = r.removed_tables.length;
        const archived = Object.values(r.archived_pairs ?? {}).reduce((a, b) => a + b, 0);
        const demoted = Object.values(r.demoted_pairs ?? {}).reduce((a, b) => a + b, 0);
        const text = `Re-indexed ${n} table${n !== 1 ? 's' : ''}` +
               (d ? `, removed ${d}` : '') +
               (archived ? `, archived ${archived} cached example${archived !== 1 ? 's' : ''}` : '') +
               (demoted ? `, demoted ${demoted} cached example${demoted !== 1 ? 's' : ''}` : '') +
               (full ? ' (full re-index)' : '') + '.';
        onSynced(text);
        check();
      })
      .catch((e) => setErr(e instanceof Error ? e.message : 'Sync failed'))
      .finally(() => setBusy(false));
  };

  const changeCount = (diff?.changes.length ?? 0) + (diff?.unindexed_tables.length ?? 0);

  return (
    <div className="sync-panel">
      <div className="sync-header">
        <div>
          <span className="sync-title">Index Sync</span>
          {diff && (
            <span className={`sync-status ${diff.in_sync ? 'sync-ok' : 'sync-stale'}`}>
              {diff.in_sync ? '✓ in sync' : `${changeCount} change${changeCount !== 1 ? 's' : ''}`}
            </span>
          )}
        </div>
        <button className="icon-btn" onClick={check} disabled={busy} title="Check for changes">
          <RefreshCw size={13} className={busy ? 'spin' : ''} />
        </button>
      </div>

      <p className="sync-hint">
        Keep the assistant's knowledge in step with the live database. Detects
        new/removed tables & columns and type changes.
        {diff?.last_synced_at && ` Last synced: ${new Date(diff.last_synced_at).toLocaleString()}.`}
      </p>

      {err && <div className="db-tab-error">{err}</div>}
      {message && <div className="sync-success">{message}</div>}

      {diff && !diff.in_sync && (
        <div className="sync-changes">
          {diff.changes.map((c, i) => {
            const meta = changeMeta(c.action);
            return (
              <div key={i} className="sync-change">
                <span className={`chg-badge ${meta.cls}`}>{meta.label}</span>
                <span className="chg-target">
                  {c.table}{c.column ? `.${c.column}` : ''}
                  {c.detail ? ` (${c.detail})` : ''}
                </span>
              </div>
            );
          })}
          {diff.unindexed_tables.map((t) => (
            <div key={`u-${t}`} className="sync-change">
              <span className="chg-badge chg-warn">unindexed</span>
              <span className="chg-target">{t}</span>
            </div>
          ))}
        </div>
      )}

      <div className="sync-actions">
        <button className="sync-btn sync-btn--primary" disabled={busy || !diff}
                onClick={() => runSync(false)}>
          Sync changes
        </button>
        <button className="sync-btn" disabled={busy || !diff}
                onClick={() => runSync(true)}
                title="Re-embed every table — use when table DATA changed (not just schema)">
          Full re-index (data changed)
        </button>
      </div>
    </div>
  );
}

export function DatabaseTab({ dbScope = '' }: { dbScope?: string }) {
  const [schema, setSchema] = useState<SchemaInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Owned here, not by IndexSyncPanel: a sync success triggers `load()`,
  // which flips `loading` to true and unmounts IndexSyncPanel (rendered only
  // when `!loading`) before its own locally-set message ever painted.
  const [syncMessage, setSyncMessage] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getSchema()
      .then(setSchema)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Failed to load schema'))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  const grouped = useMemo(
    () => groupByDomain(schema?.tables ?? []), [schema]);

  const handleSynced = useCallback((message: string) => {
    setSyncMessage(message);
    load();
  }, [load]);

  return (
    <div className="db-tab">
      <div className="db-tab-header">
        <div>
          <h2 className="db-tab-title">Connected Database</h2>
          {schema && (
            <p className="db-tab-subtitle">
              <strong>{schema.db_name}</strong> — {schema.table_count} table{schema.table_count !== 1 ? 's' : ''}
              {grouped.length > 1 && ` across ${grouped.length} databases`}
              {dbScope && <> · answering from <strong>{dbScope}</strong></>}
            </p>
          )}
        </div>
        <button className="icon-btn" onClick={load} title="Refresh schema" disabled={loading}>
          <RefreshCw size={14} className={loading ? 'spin' : ''} />
        </button>
      </div>

      {loading && (
        <div className="db-tab-loading">
          <div className="typing-indicator"><span /><span /><span /></div>
          <span>Loading schema…</span>
        </div>
      )}

      {error && (
        <div className="db-tab-error">
          {error}
          <button onClick={load}>Retry</button>
        </div>
      )}

      {!loading && !error && (
        <IndexSyncPanel onSynced={handleSynced} message={syncMessage} />
      )}

      {schema && !loading && (
        <div className="schema-browser">
          {schema.tables.length === 0 ? (
            <p className="db-tab-empty">No tables found in the database.</p>
          ) : grouped.length > 1 ? (
            grouped.map(([domain, tables]) => (
              <DomainGroup
                key={domain}
                domain={domain}
                tables={tables}
                active={domain === dbScope}
              />
            ))
          ) : (
            // One domain (or none tagged) — grouping would add a level of
            // nesting that says nothing.
            schema.tables.map((t) => <TableRow key={t.table_name} table={t} />)
          )}
        </div>
      )}
    </div>
  );
}
