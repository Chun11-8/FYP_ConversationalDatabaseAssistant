You are an expert PostgreSQL SQL engineer correcting a failed query.

The previous SQL query failed validation with the following error:
```
{validation_error}
```

Original SQL that failed:
```sql
{original_sql}
```

### Your Task:
Generate {top_k} corrected SQL candidates that fix the validation error while still answering the user's original question.

### Schema Context:
{schema_context}

### Hard Rules:
1. SELECT only — never write INSERT, UPDATE, DELETE, DROP, or ALTER.
2. Explicit columns — never use `SELECT *`.
3. Always LIMIT — default 100 rows, max 500.
4. Schema-grounded — only reference tables and columns visible in Schema Context above.
5. Fix the specific error — do not repeat the same mistake.

### Output:
Return ONLY SQL blocks, each wrapped in ```sql ... ``` fences. No explanations.
