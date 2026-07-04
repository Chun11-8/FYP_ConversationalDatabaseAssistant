"""
smoke_examples.py — end-to-end agent check on three representative BIRD questions.

Covers the exact questions that motivated Issues #17–#20:
    q1533 (debit_card_specializing) — exact date + computed evidence filter
    q1317 (student_club)            — 3-table bridge join + exact value casing
    q1323 (student_club)            — per-group HAVING threshold + EXCEPT exclusion

It builds faithful *reconstructed* mini-databases (same tables / columns / FKs /
categorical values as BIRD) so it runs without the multi-GB BIRD download. For
each question it runs the SAME path as backend/eval/bird.py:

    extract_sqlite_schema → SchemaLinkingAgent.analyze → SQLCodeAgent.generate
    → rerank → execute predicted vs gold → execution-accuracy (set equality)

Run from repo root (needs OPENAI_API_KEY reachable, i.e. api.openai.com allowed):
    python -m backend.eval.smoke_examples

Use --db-dir to point at the REAL BIRD databases instead of the reconstructions:
    python -m backend.eval.smoke_examples --db-dir path/to/dev_databases
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from backend.eval.bird import extract_sqlite_schema, execute_sqlite, compute_ex


# ── The three motivating questions (verbatim from the user) ───────────────────

EXAMPLES = [
    {
        "qid": 1533,
        "db_id": "debit_card_specializing",
        "question": "For all the people who paid more than 29.00 per unit of product "
                    "id No.5. Give their consumption status in the August of 2012.",
        "evidence": "August of 2012 refers to the Date value = '201208' ; "
                    "Price per unit of product = Price / Amount;",
        "gold": "SELECT T2.Consumption FROM transactions_1k AS T1 INNER JOIN yearmonth "
                "AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Price / "
                "NULLIF(T1.Amount, 0) > 29.00 AND T1.ProductID = 5 AND T2.Date = '201208'",
    },
    {
        "qid": 1317,
        "db_id": "student_club",
        "question": "Among the students from the Student_Club who attended the event "
                    "\"Women's Soccer\", how many of them want a T-shirt that's in medium size?",
        "evidence": "Women's Soccer is an event name; T-shirt that is in medium size "
                    "refers to t_shirt_size = 'Medium'",
        "gold": "SELECT COUNT(T1.event_id) FROM event AS T1 INNER JOIN attendance AS T2 "
                "ON T1.event_id = T2.link_to_event INNER JOIN member AS T3 ON "
                "T2.link_to_member = T3.member_id WHERE T1.event_name = 'Women''s Soccer' "
                "AND T3.t_shirt_size = 'Medium'",
    },
    {
        "qid": 1323,
        "db_id": "student_club",
        "question": "List all the names of events that had an attendance of over 20 "
                    "students but were not fundraisers.",
        "evidence": "name of events refers to event_name; attendance of over 20 students "
                    "COUNT(event_id) > 20.",
        "gold": "SELECT T1.event_name FROM event AS T1 INNER JOIN attendance AS T2 ON "
                "T1.event_id = T2.link_to_event GROUP BY T1.event_id HAVING "
                "COUNT(T2.link_to_event) > 20 EXCEPT SELECT T1.event_name FROM event AS "
                "T1 WHERE T1.type = 'Fundraiser'",
    },
]


# ── Reconstructed mini-databases ──────────────────────────────────────────────

def _build_debit_card(path: Path) -> None:
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.executescript(
        """
        CREATE TABLE customers (CustomerID INTEGER PRIMARY KEY, Segment TEXT, Currency TEXT);
        CREATE TABLE transactions_1k (
            TransactionID INTEGER PRIMARY KEY, CustomerID INTEGER, Date TEXT,
            ProductID INTEGER, Amount REAL, Price REAL,
            FOREIGN KEY (CustomerID) REFERENCES customers(CustomerID)
        );
        CREATE TABLE yearmonth (
            CustomerID INTEGER, Date TEXT, Consumption REAL,
            FOREIGN KEY (CustomerID) REFERENCES customers(CustomerID)
        );
        """
    )
    c.executemany("INSERT INTO customers VALUES (?,?,?)", [
        (1, "SME", "EUR"), (2, "LAM", "EUR"), (3, "KAM", "CZK"),
    ])
    c.executemany("INSERT INTO transactions_1k VALUES (?,?,?,?,?,?)", [
        # cust 1: ProductID 5, 60/2 = 30 > 29  → qualifies
        (101, 1, "2012-08-24", 5, 2.0, 60.0),
        # cust 2: ProductID 5, 29/2 = 14.5      → does NOT qualify
        (102, 2, "2012-08-24", 5, 2.0, 29.0),
        # cust 3: ProductID 5, amount 0 (NULLIF guard) → NULL, excluded
        (103, 3, "2012-08-24", 5, 0.0, 100.0),
        # cust 1: different product, ignored by ProductID=5
        (104, 1, "2012-08-24", 7, 1.0, 99.0),
    ])
    c.executemany("INSERT INTO yearmonth VALUES (?,?,?)", [
        (1, "201208", 100.0),   # expected answer
        (1, "201207", 50.0),    # wrong month
        (2, "201208", 200.0),   # cust 2 not qualifying
        (3, "201208", 300.0),   # cust 3 not qualifying
    ])
    conn.commit()
    conn.close()


def _build_student_club(path: Path) -> None:
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.executescript(
        """
        CREATE TABLE event (event_id INTEGER PRIMARY KEY, event_name TEXT, event_date TEXT, type TEXT);
        CREATE TABLE member (member_id INTEGER PRIMARY KEY, first_name TEXT, t_shirt_size TEXT);
        CREATE TABLE attendance (
            link_to_event INTEGER, link_to_member INTEGER,
            FOREIGN KEY (link_to_event) REFERENCES event(event_id),
            FOREIGN KEY (link_to_member) REFERENCES member(member_id)
        );
        """
    )
    c.executemany("INSERT INTO event VALUES (?,?,?,?)", [
        (1, "Women's Soccer", "2019-09-01", "Game"),
        (2, "Fundraiser Gala", "2019-10-01", "Fundraiser"),
        (3, "Open Day",        "2019-11-01", "Meeting"),
    ])
    members = [(i, f"M{i}", ["Small", "Medium", "Large"][i % 3]) for i in range(1, 31)]
    # Force m1, m2 to be Medium so Women's Soccer has exactly 2 medium attendees.
    members[0] = (1, "M1", "Medium")
    members[1] = (2, "M2", "Medium")
    members[2] = (3, "M3", "Small")
    c.executemany("INSERT INTO member VALUES (?,?,?)", members)

    att: List[Tuple[int, int]] = []
    # event 1 (Women's Soccer): m1(M), m2(M), m3(S) → 2 medium attendees
    att += [(1, 1), (1, 2), (1, 3)]
    # event 2 (Fundraiser Gala): 25 attendance rows → >20 but IS a fundraiser
    att += [(2, (i % 30) + 1) for i in range(25)]
    # event 3 (Open Day): 22 attendance rows → >20 and NOT a fundraiser → answer
    att += [(3, (i % 30) + 1) for i in range(22)]
    c.executemany("INSERT INTO attendance VALUES (?,?)", att)
    conn.commit()
    conn.close()


_BUILDERS = {
    "debit_card_specializing": _build_debit_card,
    "student_club": _build_student_club,
}


def _prepare_reconstructed_dbs(root: Path) -> None:
    for db_id, builder in _BUILDERS.items():
        db_dir = root / db_id
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / f"{db_id}.sqlite"
        if db_path.exists():
            db_path.unlink()
        builder(db_path)


# ── Runner ────────────────────────────────────────────────────────────────────

def _find_db(db_dir: Path, db_id: str) -> str:
    for ext in (".sqlite", ".db"):
        p = db_dir / db_id / f"{db_id}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No database for {db_id} under {db_dir}")


def run(db_dir: Path) -> int:
    # Build the agents exactly like bird.py (dialect → sqlite).
    from backend.config.config_loader import load_config
    from backend.agents.sql_code_agent import SQLCodeAgent
    from backend.agents.sql_validator_agent import SQLValidatorAgent
    from backend.agents.schema_linking_agent import SchemaLinkingAgent
    from langchain_openai import ChatOpenAI

    config = load_config()
    bird_config = {**config, "db": {**config.get("db", {}), "dialect": "sqlite"}}
    llm = ChatOpenAI(model=config["llm"]["model"], temperature=0,
                     api_key=config["llm"]["api_key"], max_retries=1)
    sql_agent = SQLCodeAgent(bird_config, llm)
    validator = SQLValidatorAgent(bird_config)
    linker = SchemaLinkingAgent(bird_config, llm)

    n_pass = 0
    for ex in EXAMPLES:
        db_path = _find_db(db_dir, ex["db_id"])
        schema = extract_sqlite_schema(db_path)
        full_query = f"<domain_context>{ex['evidence']}</domain_context>\n\n{ex['question']}"

        print("\n" + "=" * 78)
        print(f"  q{ex['qid']}  ({ex['db_id']})")
        print("=" * 78)
        print(f"Question : {ex['question']}")
        print(f"Evidence : {ex['evidence']}")
        print("\n── Schema context fed to the agent ──")
        print(schema)

        # Gold result (ground truth on the reconstructed DB)
        gold_rows, gold_err, _ = execute_sqlite(ex["gold"], db_path)
        print(f"\nGold SQL : {ex['gold']}")
        print(f"Gold rows: {gold_rows}  {'(error: ' + gold_err + ')' if gold_err else ''}")

        # ── Live agent run ──────────────────────────────────────────────────
        try:
            plan = linker.analyze(full_query, schema)
            sqls = sql_agent.generate(query=full_query, schema_context=schema,
                                      few_shots=[], history=[], query_plan=plan)
            pred = sql_agent.rerank(sqls, full_query, schema) if sqls else ""
        except Exception as exc:
            print(f"\n[LLM UNAVAILABLE] {type(exc).__name__}: {str(exc)[:160]}")
            print("Cannot generate predicted SQL in this environment "
                  "(api.openai.com must be reachable). Schema/gold above are valid.")
            continue

        print(f"\nQuery plan: {plan}")
        print(f"Predicted : {pred}")
        validation = validator.validate(pred, schema) if pred else {"valid": False}
        pred_rows, pred_err, _ = execute_sqlite(pred, db_path) if pred else ([], "no SQL", 0)
        ex_score = compute_ex(pred_rows, gold_rows) if (pred and not pred_err) else 0.0
        print(f"Validation: valid={validation.get('valid')} error={validation.get('error')}")
        print(f"Pred rows : {pred_rows}  {'(error: ' + pred_err + ')' if pred_err else ''}")
        print(f"EX        : {ex_score}  {'✅ MATCH' if ex_score == 1.0 else '❌ MISMATCH'}")
        n_pass += int(ex_score == 1.0)

    print("\n" + "=" * 78)
    print(f"  Result: {n_pass}/{len(EXAMPLES)} match gold (EX=1.0)")
    print("=" * 78)
    return 0 if n_pass == len(EXAMPLES) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-dir", default=None,
                        help="Use real BIRD dev_databases instead of the reconstructions")
    args = parser.parse_args(argv)

    if args.db_dir:
        db_dir = Path(args.db_dir)
    else:
        scratch = Path(__file__).parent / "_smoke_dbs"
        _prepare_reconstructed_dbs(scratch)
        db_dir = scratch
        print(f"[INFO] Using reconstructed mini-databases under {db_dir}")
    return run(db_dir)


if __name__ == "__main__":
    sys.exit(main())
