#!/usr/bin/env python3
"""
Generate the synthetic SaaS database used by the agent evaluation project.

Deterministic: the same SEED always produces the same database.
Usage:  python generate_db.py [output_path]

Conventions baked into the data (the glossary repeats the ones that matter):
  * subscriptions.end_date is EXCLUSIVE: the first day the subscription is no longer active.
  * A subscription is active on date D if start_date <= D, (end_date IS NULL OR end_date > D),
    and status != 'paused'.
  * A plan change ends the old subscription (status 'cancelled', cancel_reason 'plan_change')
    and starts a new one on the same day.
  * Prices are constant over the whole period.
  * "Today" for all questions is AS_OF_DATE (2026-01-01); data runs to DATA_END.
"""
import calendar
import hashlib
import json
import math
import random
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 42
AS_OF_DATE = date(2026, 1, 1)
DATA_START = date(2023, 1, 1)
DATA_END = date(2025, 12, 31)

N_CUSTOMERS = 500
N_TEST_ACCOUNTS = 15
N_TICKETS = 1500
N_PAUSED = 8

PLANS = [
    # (plan_id, plan_name, legacy_name, monthly_price, annual_price)
    (1, "Starter", None, 29.0, 290.0),
    (2, "Pro", "Growth", 99.0, 990.0),  # renamed plan: old name kept in legacy_name
    (3, "Business", None, 299.0, 2990.0),
    (4, "Enterprise", None, 999.0, 9990.0),
]
PLAN_WEIGHTS = [40, 35, 18, 7]

NAME_A = ["Blue", "Nova", "Apex", "Summit", "Orbit", "Pixel", "Cedar", "Lumen", "Harbor", "Quartz",
          "Vertex", "Maple", "Ember", "Atlas", "Falcon", "Iron", "Silver", "Zenith", "Delta", "Echo",
          "Northwind", "Bright", "Swift", "Clear", "Prime"]
NAME_B = ["Labs", "Systems", "Logistics", "Analytics", "Health", "Foods", "Works", "Digital", "Partners",
          "Networks", "Media", "Robotics", "Capital", "Studio", "Energy", "Retail", "Cloud", "Dynamics",
          "Supply", "Learning"]
SUFFIXES = ["Inc", "Ltd", "LLC", "GmbH", "SA", ""]
TEST_NAMES = [f"Internal QA Sandbox {i}" for i in range(1, 9)] + [f"Demo Account {i}" for i in range(1, 8)]

COUNTRIES = [("United States", 30), ("United Kingdom", 12), ("Germany", 8), ("France", 6), ("Canada", 6),
             ("Egypt", 6), ("United Arab Emirates", 6), ("Australia", 5), ("India", 5), ("Brazil", 4),
             ("Netherlands", 4), ("Spain", 3), ("Singapore", 3), ("Saudi Arabia", 2)]
INDUSTRIES = ["Software", "Healthcare", "Retail", "Finance", "Education", "Logistics", "Media",
              "Manufacturing", "Real Estate", "Hospitality"]
CHANNELS = ["organic_search", "paid_ads", "referral", "partner", "outbound_sales", "events", "social"]

STORIES = ["steady", "churned", "upgrade", "downgrade", "resub", "multi"]
STORY_WEIGHTS = [33, 20, 13, 5, 22, 7]  # resub is high because many resub stories run out of room
CANCEL_REASONS = [("price", 25), ("missing_features", 20), ("switched_competitor", 20),
                  ("business_closed", 10), ("no_longer_needed", 15), ("poor_support", 10)]

PRIORITY = [("low", 35), ("medium", 40), ("high", 20), ("urgent", 5)]
MEDIAN_RESOLUTION_HOURS = {"low": 60, "medium": 30, "high": 12, "urgent": 4}
CATEGORY = [("billing", 22), ("bug", 28), ("how_to", 30), ("feature_request", 12), ("account", 8)]
CSAT = [(1, 5), (2, 8), (3, 17), (4, 30), (5, 40)]

SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    company_name TEXT NOT NULL,
    country TEXT NOT NULL,
    industry TEXT,
    signup_date TEXT NOT NULL,
    acquisition_channel TEXT,
    is_test_account INTEGER NOT NULL CHECK (is_test_account IN (0, 1))
);
CREATE TABLE plans (
    plan_id INTEGER PRIMARY KEY,
    plan_name TEXT NOT NULL,
    legacy_name TEXT,
    monthly_price REAL NOT NULL,
    annual_price REAL NOT NULL
);
CREATE TABLE subscriptions (
    subscription_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    plan_id INTEGER NOT NULL REFERENCES plans(plan_id),
    billing_cycle TEXT NOT NULL CHECK (billing_cycle IN ('monthly', 'annual')),
    start_date TEXT NOT NULL,
    end_date TEXT,
    status TEXT NOT NULL CHECK (status IN ('active', 'paused', 'cancelled')),
    cancel_reason TEXT
);
CREATE TABLE invoices (
    invoice_id INTEGER PRIMARY KEY,
    subscription_id INTEGER NOT NULL REFERENCES subscriptions(subscription_id),
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    invoice_date TEXT NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('paid', 'failed', 'refunded')),
    refunded_amount REAL NOT NULL DEFAULT 0
);
CREATE TABLE support_tickets (
    ticket_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    priority TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    category TEXT NOT NULL CHECK (category IN ('billing', 'bug', 'how_to', 'feature_request', 'account')),
    csat_score INTEGER CHECK (csat_score BETWEEN 1 AND 5)
);
CREATE INDEX idx_subscriptions_customer ON subscriptions(customer_id);
CREATE INDEX idx_invoices_customer ON invoices(customer_id);
CREATE INDEX idx_invoices_subscription ON invoices(subscription_id);
CREATE INDEX idx_tickets_customer ON support_tickets(customer_id);
"""


# ---------------------------------------------------------------- helpers

def pick(rng, pairs):
    """Weighted choice from a list of (value, weight) pairs."""
    return rng.choices([p[0] for p in pairs], [p[1] for p in pairs])[0]


def add_months(d, n):
    idx = d.year * 12 + (d.month - 1) + n
    y, m = divmod(idx, 12)
    m += 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def quarter_boundaries():
    """Last day of each quarter and the first day of the next, within the data range."""
    out = []
    for y in range(2023, 2026):
        for m in (3, 6, 9, 12):
            last = date(y, m, calendar.monthrange(y, m)[1])
            out.append(last)
            out.append(last + timedelta(days=1))
    return [d for d in out if DATA_START <= d <= DATA_END]


BOUNDARIES = quarter_boundaries()


def snap_to_boundary(rng, d, lo, p):
    """With probability p, move d to the nearest quarter-boundary date within 15 days.
    The result always stays after `lo` and on or before DATA_END."""
    if rng.random() >= p:
        return d
    nearest = min(BOUNDARIES, key=lambda b: abs((b - d).days))
    if abs((nearest - d).days) <= 15 and lo < nearest <= DATA_END:
        return nearest
    return d


# ---------------------------------------------------------------- customers

def build_customers(rng):
    span = (DATA_END - DATA_START).days - 120
    combos = [(a, b) for a in NAME_A for b in NAME_B]
    rng.shuffle(combos)
    n_real = N_CUSTOMERS - N_TEST_ACCOUNTS
    customers = []
    for i in range(N_CUSTOMERS):
        is_test = i >= n_real
        if is_test:
            name = TEST_NAMES[i - n_real]
            offset = int(span * rng.random() * 0.35)  # internal accounts were created early
        else:
            a, b = combos[i]
            name = f"{a} {b} {rng.choice(SUFFIXES)}".strip()
            offset = int(span * rng.betavariate(1.3, 1.0))  # signups skew later: a growing company
        signup = DATA_START + timedelta(days=offset)
        signup = snap_to_boundary(rng, signup, DATA_START - timedelta(days=1), 0.06)
        customers.append({
            "company_name": name,
            "country": pick(rng, COUNTRIES),
            "industry": None if rng.random() < 0.04 else rng.choice(INDUSTRIES),
            "signup_date": signup,
            "acquisition_channel": None if rng.random() < 0.03 else rng.choice(CHANNELS),
            "is_test_account": 1 if is_test else 0,
        })
    customers.sort(key=lambda c: (c["signup_date"], c["company_name"]))
    for i, c in enumerate(customers, 1):
        c["customer_id"] = i
    return customers


# ---------------------------------------------------------------- subscriptions

def new_cycle(rng):
    return "annual" if rng.random() < 0.30 else "monthly"


def shifted_plan(plan_i, direction):
    """Move one tier up or down; at the top or bottom tier the direction flips."""
    if direction == "up":
        return plan_i + 1 if plan_i < len(PLANS) - 1 else plan_i - 1
    return plan_i - 1 if plan_i > 0 else plan_i + 1


def seg_end(rng, start, min_len=75, tail=45):
    """End date for a subscription segment, or None if there is no room before DATA_END."""
    room = (DATA_END - start).days - tail
    if room <= min_len:
        return None
    end = start + timedelta(days=rng.randint(min_len, room))
    return snap_to_boundary(rng, end, start + timedelta(days=14), 0.12)


def events_for(rng, story):
    if story == "steady":
        return []
    if story == "churned":
        return ["cancel"]
    if story == "upgrade":
        return ["up"] + (["cancel"] if rng.random() < 0.3 else [])
    if story == "downgrade":
        return ["down"]
    if story == "resub":
        return ["cancel", "resub"] + (["cancel"] if rng.random() < 0.25 else [])
    return rng.choice([["up", "cancel"], ["up", "down"], ["down", "up"]])  # multi


def build_segments(rng, signup, is_test):
    """Subscription history for one customer. end=None means still running."""
    plan_i = 3 if is_test else rng.choices(range(len(PLANS)), PLAN_WEIGHTS)[0]
    story = "steady" if is_test else rng.choices(STORIES, STORY_WEIGHTS)[0]
    events = events_for(rng, story)

    cur = {"plan_i": plan_i, "cycle": new_cycle(rng), "start": signup, "end": None, "reason": None}
    segs = [cur]
    last_end, last_plan = None, None
    for ev in events:
        if ev == "resub":
            if cur is not None:
                continue
            start2 = last_end + timedelta(days=rng.randint(30, 200))
            start2 = snap_to_boundary(rng, start2, last_end, 0.08)
            if (DATA_END - start2).days < 90:
                break
            if rng.random() < 0.6:
                plan2 = last_plan
            else:
                plan2 = shifted_plan(last_plan, rng.choice(["up", "down"]))
            cur = {"plan_i": plan2, "cycle": new_cycle(rng), "start": start2, "end": None, "reason": None}
            segs.append(cur)
            continue
        if cur is None:
            break
        end = seg_end(rng, cur["start"])
        if end is None:
            break
        cur["end"] = end
        if ev == "cancel":
            cur["reason"] = pick(rng, CANCEL_REASONS)
            last_end, last_plan = end, cur["plan_i"]
            cur = None
        else:  # plan change: the new subscription starts the day the old one ends
            cur["reason"] = "plan_change"
            if rng.random() < 0.2:
                next_cycle = "annual" if cur["cycle"] == "monthly" else "monthly"
            else:
                next_cycle = cur["cycle"]
            cur = {"plan_i": shifted_plan(cur["plan_i"], ev), "cycle": next_cycle,
                   "start": end, "end": None, "reason": None}
            segs.append(cur)
    return segs


def build_subscriptions(rng, customers):
    subs = []
    for c in customers:
        for seg in build_segments(rng, c["signup_date"], c["is_test_account"]):
            subs.append({"customer_id": c["customer_id"], "is_test": c["is_test_account"], **seg})

    # A few ongoing subscriptions are paused: they stop invoicing and never count as active.
    eligible = [s for s in subs
                if s["end"] is None and not s["is_test"] and s["start"] < date(2025, 6, 1)]
    for s in rng.sample(eligible, N_PAUSED):
        s["paused"] = True
        s["invoice_until"] = DATA_END - timedelta(days=rng.randint(60, 120))

    subs.sort(key=lambda s: (s["start"], s["customer_id"]))
    for i, s in enumerate(subs, 1):
        s["subscription_id"] = i
        if s.get("paused"):
            s["status"] = "paused"
        else:
            s["status"] = "active" if s["end"] is None else "cancelled"
    return subs


# ---------------------------------------------------------------- invoices

def build_invoices(rng, subs):
    rows = []
    for s in subs:
        plan = PLANS[s["plan_i"]]
        annual = s["cycle"] == "annual"
        price = plan[4] if annual else plan[3]
        limit = s.get("invoice_until", DATA_END)
        k = 0
        while True:
            d = add_months(s["start"], k * (12 if annual else 1))
            if d > limit or (s["end"] is not None and d >= s["end"]):
                break
            if s["is_test"]:
                status, refunded = "paid", 0.0
            else:
                roll = rng.random()
                if roll < 0.05:
                    status, refunded = "failed", 0.0
                elif roll < 0.08:
                    status = "refunded"
                    if rng.random() < 0.5:
                        refunded = price
                    else:
                        refunded = round(price * rng.choice([0.25, 0.5, 0.75]), 2)
                else:
                    status, refunded = "paid", 0.0
            rows.append({"subscription_id": s["subscription_id"], "customer_id": s["customer_id"],
                         "invoice_date": d, "amount": price, "status": status,
                         "refunded_amount": refunded})
            k += 1
    rows.sort(key=lambda r: (r["invoice_date"], r["subscription_id"]))
    for i, r in enumerate(rows, 1):
        r["invoice_id"] = i
    return rows


# ---------------------------------------------------------------- tickets

def build_tickets(rng, customers, subs):
    by_customer = {}
    for s in subs:
        by_customer.setdefault(s["customer_id"], []).append(s)

    def span_days(s):
        return ((s["end"] or (DATA_END + timedelta(days=1))) - s["start"]).days

    cust_ids = [c["customer_id"] for c in customers]
    weights = [sum(span_days(s) for s in by_customer[cid]) for cid in cust_ids]

    rows = []
    for _ in range(N_TICKETS):
        cid = rng.choices(cust_ids, weights)[0]
        periods = by_customer[cid]
        s = rng.choices(periods, [span_days(p) for p in periods])[0]
        d = s["start"] + timedelta(days=rng.randrange(span_days(s)))
        created = datetime(d.year, d.month, d.day) + timedelta(
            hours=rng.randint(8, 19), minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))
        priority = pick(rng, PRIORITY)
        category = pick(rng, CATEGORY)

        recent = (DATA_END - d).days < 14
        resolved_at = None
        if rng.random() < (0.6 if recent else 0.93):
            hours = rng.lognormvariate(math.log(MEDIAN_RESOLUTION_HOURS[priority]), 0.6)
            r = (created + timedelta(hours=hours)).replace(microsecond=0)
            if r.date() <= DATA_END:
                resolved_at = r
        csat = pick(rng, CSAT) if resolved_at and rng.random() < 0.55 else None
        rows.append({"customer_id": cid, "created_at": created, "resolved_at": resolved_at,
                     "priority": priority, "category": category, "csat_score": csat})
    rows.sort(key=lambda r: (r["created_at"], r["customer_id"]))
    for i, r in enumerate(rows, 1):
        r["ticket_id"] = i
    return rows


# ---------------------------------------------------------------- output

def iso(d):
    return d.isoformat() if d else None


def ts(t):
    return t.strftime("%Y-%m-%d %H:%M:%S") if t else None


def write_db(path, customers, subs, invoices, tickets):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO plans VALUES (?,?,?,?,?)", PLANS)
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?)", [
        (c["customer_id"], c["company_name"], c["country"], c["industry"], iso(c["signup_date"]),
         c["acquisition_channel"], c["is_test_account"]) for c in customers])
    conn.executemany("INSERT INTO subscriptions VALUES (?,?,?,?,?,?,?,?)", [
        (s["subscription_id"], s["customer_id"], s["plan_i"] + 1, s["cycle"], iso(s["start"]),
         iso(s["end"]), s["status"], s["reason"]) for s in subs])
    conn.executemany("INSERT INTO invoices VALUES (?,?,?,?,?,?,?)", [
        (r["invoice_id"], r["subscription_id"], r["customer_id"], iso(r["invoice_date"]),
         r["amount"], r["status"], r["refunded_amount"]) for r in invoices])
    conn.executemany("INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?)", [
        (r["ticket_id"], r["customer_id"], ts(r["created_at"]), ts(r["resolved_at"]),
         r["priority"], r["category"], r["csat_score"]) for r in tickets])
    conn.commit()

    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("customers", "plans", "subscriptions", "invoices", "support_tickets")}
    digest = hashlib.sha256("\n".join(conn.iterdump()).encode()).hexdigest()
    conn.close()

    meta = {"seed": SEED, "as_of_date": iso(AS_OF_DATE), "data_start": iso(DATA_START),
            "data_end": iso(DATA_END), "row_counts": counts, "content_sha256": digest}
    path.with_name("db_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "saas.db"
    rng = random.Random(SEED)
    customers = build_customers(rng)
    subs = build_subscriptions(rng, customers)
    invoices = build_invoices(rng, subs)
    tickets = build_tickets(rng, customers, subs)
    meta = write_db(out, customers, subs, invoices, tickets)
    print(f"Wrote {out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
