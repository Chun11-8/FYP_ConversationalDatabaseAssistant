-- GH#41 defense-in-depth: a SELECT-only role for executing USER queries.
-- Run once as a superuser, then set PG_APP_USER / PG_APP_PASSWORD in
-- secrets.env — execute_query() will use this role automatically, so a write
-- cannot succeed even if every validation layer were bypassed.

CREATE ROLE assistant_ro LOGIN PASSWORD 'CHANGE_ME';
ALTER ROLE assistant_ro SET default_transaction_read_only = on;
GRANT CONNECT ON DATABASE "FYP" TO assistant_ro;
GRANT USAGE ON SCHEMA public TO assistant_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO assistant_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO assistant_ro;
