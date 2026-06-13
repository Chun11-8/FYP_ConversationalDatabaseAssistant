import { useEffect, useState } from 'react';
import { ChevronRight, ChevronDown, Key, Link, RefreshCw } from 'lucide-react';
import type { SchemaInfo, SchemaTable } from '../types';
import { getSchema } from '../api/client';

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
            <div key={col.name} className="schema-column">
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
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function DatabaseTab() {
  const [schema, setSchema] = useState<SchemaInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    getSchema()
      .then(setSchema)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Failed to load schema'))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  return (
    <div className="db-tab">
      <div className="db-tab-header">
        <div>
          <h2 className="db-tab-title">Connected Database</h2>
          {schema && (
            <p className="db-tab-subtitle">
              <strong>{schema.db_name}</strong> — {schema.table_count} table{schema.table_count !== 1 ? 's' : ''}
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

      {schema && !loading && (
        <div className="schema-browser">
          {schema.tables.length === 0 ? (
            <p className="db-tab-empty">No tables found in the database.</p>
          ) : (
            schema.tables.map((t) => <TableRow key={t.table_name} table={t} />)
          )}
        </div>
      )}
    </div>
  );
}
