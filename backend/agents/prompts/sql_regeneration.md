You are an expert SQL engineer correcting a failed query. Write valid SQL for
the dialect used in the schema/question context below.

The previous SQL query failed validation with the following error:
```
{validation_error}
```

Original SQL that failed:
```sql
{original_sql}
```

### Your Task:
Generate {top_k} corrected SQL candidates that fix the validation error while
still fully answering the user's original question.

### Schema Context:
{schema_context}

### Before correcting, re-deconstruct the question:
- ANSWER COLUMNS: return only what is asked — for "who/which/what X" return only
  X's identifying column; do not add the sorted metric or extra columns.
- FILTERS: every term in the question must be a predicate. If the question names
  a value (a code/label/category/name/year) that is not a table or column in the
  schema, map it to the categorical column that holds it and filter on it.
- EXTREMITY: least/most/lowest/highest/top → ORDER BY <aggregate> + LIMIT 1
  (or LIMIT N for "top N").

### Hard Rules:
1. SELECT only — never INSERT, UPDATE, DELETE, DROP, or ALTER.
2. Explicit columns — never `SELECT *`; return only the answer columns.
3. LIMIT matches intent — LIMIT 1 for a single superlative answer, LIMIT N for
   "top N", otherwise ≤ 500.
4. Schema-grounded — only reference tables/columns visible in Schema Context;
   JOIN on the foreign-key columns shown there.
5. Fix the specific error — do not repeat the same mistake, and do not drop any
   filter from the question while fixing it.

### Output:
Return ONLY SQL blocks, each wrapped in ```sql ... ``` fences. No explanations.
