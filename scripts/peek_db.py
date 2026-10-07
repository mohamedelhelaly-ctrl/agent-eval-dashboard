import sqlite3
from pathlib import Path

# Find data/saas.db relative to this script, and open it read-only (mode=ro).
db = Path(__file__).resolve().parent.parent / "data" / "saas.db"
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

# List every user table (skip SQLite's internal ones).
tables = [r[0] for r in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]

for t in tables:
    # 1) Row count for the table.
    print(f"\n=== {t}: {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} rows ===")
    # 2) Column names and declared types (PRAGMA table_info rows: cid, name, type, ...).
    print("columns:", ", ".join(f"{c[1]} ({c[2]})" for c in con.execute(f"PRAGMA table_info({t})")))
    # 3) Three sample rows.
    for row in con.execute(f"SELECT * FROM {t} LIMIT 3"):
        print("  ", row)

# 4) Labeled counts that show how "customer" and "active" can be counted differently.
D = "2026-01-01"
print("\n=== counts ===")
q = lambda sql, *a: con.execute(sql, a).fetchall()
print("customers total:", q("SELECT COUNT(*) FROM customers")[0][0])
print("test accounts:", q("SELECT COUNT(*) FROM customers WHERE is_test_account=1")[0][0])
print("subscriptions by status:", dict(q("SELECT status, COUNT(*) FROM subscriptions GROUP BY status")))
# Active on a date by dates: started on/before it, not ended by it, not paused, real customers only.
print(f"customers active on {D} (by dates, non-paused, non-test):", q(
    """SELECT COUNT(DISTINCT s.customer_id) FROM subscriptions s
       JOIN customers c USING (customer_id)
       WHERE s.start_date <= ? AND (s.end_date IS NULL OR s.end_date > ?)
         AND s.status != 'paused' AND c.is_test_account = 0""", D, D)[0][0])
# Active by the status label only, ignoring dates (and test accounts).
print("customers with any subscription status='active':", q(
    "SELECT COUNT(DISTINCT customer_id) FROM subscriptions WHERE status='active'")[0][0])
