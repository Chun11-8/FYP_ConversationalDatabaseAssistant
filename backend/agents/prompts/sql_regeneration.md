You are an expert SQL engineer correcting a query that did not work. Write valid
SQL for the dialect used in the schema/question context below.

### User's original question:
{user_question}

### What happened (the observation to react to):
```
{validation_error}
```

### Original SQL that failed:
```sql
{original_sql}
```

### Your Task:
Generate {top_k} corrected SQL candidates that resolve the observation above while
still fully and exactly answering the user's question. Diagnose the cause first:

- **Execution error** (syntax/unknown column/type): fix the exact offending part;
  use only columns/tables that exist in the schema.
- **0 rows returned**: the query is valid but matched nothing. Look for an
  over-strict or wrong predicate — a mis-mapped literal value (e.g. wrong case or
  wrong categorical column), a wrong date format (`SUBSTR(date,1,4)='2013'` vs
  `date LIKE '2013%'`), or a JOIN on the wrong key that eliminated all rows. If,
  after checking, the query already looks correct (the answer may genuinely be
  empty), return it unchanged.

### Re-deconstruct before correcting:
- EVIDENCE: if the question contains a `<domain_context>` / evidence hint, apply it
  VERBATIM — exact values (`Date = '201208'`, matched exactly, not a year-only
  `SUBSTR`), exact formats, and formulas used as filters (`Price / NULLIF(Amount,0)
  > 29`). Do not paraphrase or drop a hint while fixing the query.
- FINAL VALUE: return only what is asked; copy any evidence formula verbatim.
- TABLES & JOINS: join the table holding the filter column to the one holding the
  metric, on the foreign-key columns shown in the schema.
- GRAIN: JOIN → FILTER → aggregate ONCE over the resulting rows. Do not
  pre-aggregate in a CTE and aggregate again (it changes the math).
- FILTERS: every term in the question is a predicate; do not drop any.
- EXTREMITY: least/most/highest/top → ORDER BY <aggregate> + LIMIT 1 (or N).

### Hard Rules:
1. SELECT only — never INSERT, UPDATE, DELETE, DROP, or ALTER.
2. Explicit columns — never `SELECT *`; return only the answer columns.
3. A scalar aggregate has no GROUP BY/LIMIT; superlatives use ORDER BY + LIMIT 1.
4. Schema-grounded — only tables/columns in the Schema Context; JOIN on real FK
   columns.
5. Fix the specific problem — do not repeat the same mistake, and do not drop any
   filter from the question while fixing it.

### Schema Context:
{schema_context}

### Output:
Return ONLY SQL blocks, each wrapped in ```sql ... ``` fences. No explanations.
