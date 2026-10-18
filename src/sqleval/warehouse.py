"""Build the benchmark's warehouse: a retail business in DuckDB, generated deterministically from a seed.

The data is synthetic so the benchmark is reproducible and free to share, but it is built to have the traps
real warehouses have: cancelled orders that must not count as revenue, partial refunds, customers who never
ordered, discontinued products, nullable referral and promo columns, weekend seasonality and a holiday peak.

    python -m sqleval.warehouse build/warehouse.duckdb
"""

from __future__ import annotations

import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

SEED = 20260630
START, END = date(2024, 1, 1), date(2026, 6, 30)  # "today" for the benchmark is 2026-06-30

STATES = {
    "West": [("Seattle", "WA"), ("Portland", "OR"), ("San Jose", "CA"), ("Los Angeles", "CA"), ("Denver", "CO")],
    "South": [("Austin", "TX"), ("Dallas", "TX"), ("Atlanta", "GA"), ("Miami", "FL"), ("Nashville", "TN")],
    "Midwest": [("Chicago", "IL"), ("Columbus", "OH"), ("Minneapolis", "MN"), ("Detroit", "MI")],
    "Northeast": [("New York", "NY"), ("Boston", "MA"), ("Albany", "NY"), ("Philadelphia", "PA")],
}
CATALOG = {
    "Electronics": {"Audio": (25, 350), "Accessories": (8, 60), "Wearables": (60, 420)},
    "Home": {"Kitchen": (12, 240), "Decor": (10, 150), "Bedding": (25, 260)},
    "Outdoors": {"Camping": (20, 380), "Cycling": (15, 900)},
    "Office": {"Stationery": (3, 40), "Furniture": (80, 700)},
}
FIRST = [
    "Ava",
    "Liam",
    "Mia",
    "Noah",
    "Zoe",
    "Ethan",
    "Isla",
    "Arjun",
    "Priya",
    "Mateo",
    "Chloe",
    "Omar",
    "Lena",
    "Kai",
    "Nora",
    "Ravi",
    "Sofia",
    "Leo",
    "Maya",
    "Jonah",
    "Elena",
    "Sam",
    "Aria",
    "Theo",
]
LAST = [
    "Patel",
    "Nguyen",
    "Smith",
    "Garcia",
    "Kim",
    "Johnson",
    "Reddy",
    "Brown",
    "Lopez",
    "Chen",
    "Davis",
    "Khan",
    "Wilson",
    "Moore",
    "Singh",
    "Clark",
    "Lee",
    "Walker",
    "Hall",
    "Young",
]
ADJ = ["Aero", "Nova", "Terra", "Lumen", "Pulse", "Summit", "Harbor", "Cedar", "Echo", "Atlas", "Vista", "Orbit"]
NOUN = {
    "Audio": ["Earbuds", "Speaker", "Headphones"],
    "Accessories": ["Cable", "Charger", "Case"],
    "Wearables": ["Watch", "Band", "Ring"],
    "Kitchen": ["Kettle", "Blender", "Pan Set"],
    "Decor": ["Lamp", "Vase", "Frame"],
    "Bedding": ["Duvet", "Pillow", "Sheet Set"],
    "Camping": ["Tent", "Lantern", "Sleeping Bag"],
    "Cycling": ["Helmet", "Bike", "Pump"],
    "Stationery": ["Notebook", "Pen Pack", "Planner"],
    "Furniture": ["Desk", "Chair", "Shelf"],
}

SCHEMA = """
CREATE TABLE customers (
  customer_id   INTEGER PRIMARY KEY,
  name          VARCHAR NOT NULL,
  email         VARCHAR NOT NULL,
  city          VARCHAR NOT NULL,
  state         VARCHAR NOT NULL,     -- two-letter US state
  region        VARCHAR NOT NULL,     -- West | South | Midwest | Northeast
  segment       VARCHAR NOT NULL,     -- consumer | small_business | enterprise
  signup_date   DATE NOT NULL,
  referred_by   INTEGER               -- customer_id of the referrer, NULL if not referred
);
CREATE TABLE products (
  product_id    INTEGER PRIMARY KEY,
  sku           VARCHAR NOT NULL,
  name          VARCHAR NOT NULL,
  category      VARCHAR NOT NULL,
  subcategory   VARCHAR NOT NULL,
  unit_cost     DECIMAL(10,2) NOT NULL,   -- what the business pays per unit
  list_price    DECIMAL(10,2) NOT NULL,
  launched_on   DATE NOT NULL,
  discontinued  BOOLEAN NOT NULL
);
CREATE TABLE orders (
  order_id      INTEGER PRIMARY KEY,
  customer_id   INTEGER NOT NULL,
  order_date    DATE NOT NULL,
  status        VARCHAR NOT NULL,     -- placed | shipped | delivered | cancelled | returned
  channel       VARCHAR NOT NULL,     -- web | app | store | phone
  ship_date     DATE,                 -- NULL until shipped; NULL for cancelled orders
  promo_code    VARCHAR               -- NULL when no promotion was used
);
CREATE TABLE order_items (
  order_id      INTEGER NOT NULL,
  line_no       INTEGER NOT NULL,
  product_id    INTEGER NOT NULL,
  quantity      INTEGER NOT NULL,
  unit_price    DECIMAL(10,2) NOT NULL,   -- price charged per unit before discount
  discount      DECIMAL(4,2) NOT NULL,    -- fraction, e.g. 0.10 = 10% off the line
  PRIMARY KEY (order_id, line_no)
);
CREATE TABLE payments (
  payment_id    INTEGER PRIMARY KEY,
  order_id      INTEGER NOT NULL,
  paid_at       TIMESTAMP NOT NULL,
  method        VARCHAR NOT NULL,     -- card | paypal | gift_card | bank_transfer
  amount        DECIMAL(12,2) NOT NULL
);
CREATE TABLE returns (
  return_id     INTEGER PRIMARY KEY,
  order_id      INTEGER NOT NULL,
  product_id    INTEGER NOT NULL,
  returned_at   DATE NOT NULL,
  reason        VARCHAR NOT NULL,     -- damaged | wrong_item | not_as_described | changed_mind
  refund_amount DECIMAL(12,2) NOT NULL
);
"""


COLUMNS = {
    "customers": ["customer_id", "name", "email", "city", "state", "region", "segment", "signup_date", "referred_by"],
    "products": [
        "product_id",
        "sku",
        "name",
        "category",
        "subcategory",
        "unit_cost",
        "list_price",
        "launched_on",
        "discontinued",
    ],
    "orders": ["order_id", "customer_id", "order_date", "status", "channel", "ship_date", "promo_code"],
    "order_items": ["order_id", "line_no", "product_id", "quantity", "unit_price", "discount"],
    "payments": ["payment_id", "order_id", "paid_at", "method", "amount"],
    "returns": ["return_id", "order_id", "product_id", "returned_at", "reason", "refund_amount"],
}


def _insert(con: duckdb.DuckDBPyConnection, table: str, rows: list[tuple]) -> None:  # type: ignore[type-arg]
    frame = pd.DataFrame(rows, columns=COLUMNS[table])  # noqa: F841 - read by DuckDB's replacement scan
    con.execute(f"INSERT INTO {table} SELECT * FROM frame")


def _rng() -> random.Random:
    return random.Random(SEED)


def generate(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    rng = _rng()
    con.execute(SCHEMA)

    # customers ---------------------------------------------------------------
    customers = []
    days = (END - START).days
    for cid in range(1, 3001):
        region = rng.choices(list(STATES), weights=[34, 30, 18, 18])[0]
        city, state = rng.choice(STATES[region])
        first, last = rng.choice(FIRST), rng.choice(LAST)
        signup = START + timedelta(days=int(days * rng.random() ** 1.3))
        referred = rng.randint(1, cid - 1) if cid > 50 and rng.random() < 0.18 else None
        seg = rng.choices(["consumer", "small_business", "enterprise"], weights=[78, 17, 5])[0]
        customers.append(
            (
                cid,
                f"{first} {last}",
                f"{first.lower()}.{last.lower()}{cid}@example.com",
                city,
                state,
                region,
                seg,
                signup,
                referred,
            )
        )
    _insert(con, "customers", customers)

    # products ----------------------------------------------------------------
    products = []
    pid = 0
    for cat, subs in CATALOG.items():
        for sub, (lo, hi) in subs.items():
            for k in range(12 if cat != "Office" else 9):
                pid += 1
                price = round(rng.uniform(lo, hi), 2)
                name = f"{rng.choice(ADJ)} {NOUN[sub][k % 3]}" + (f" {k // 3 + 1}" if k >= 3 else "")
                launched = START - timedelta(days=200) + timedelta(days=rng.randint(0, days - 60))
                products.append(
                    (
                        pid,
                        f"{cat[:3].upper()}-{sub[:3].upper()}-{pid:04d}",
                        name,
                        cat,
                        sub,
                        round(price * rng.uniform(0.38, 0.7), 2),
                        price,
                        launched,
                        rng.random() < 0.12,
                    )
                )
    _insert(con, "products", products)
    popularity = [min(rng.paretovariate(2.2), 8.0) / float(p[6]) ** 0.45 for p in products]  # cheaper sells more

    # orders, items, payments, returns ----------------------------------------
    orders, items, payments, returns = [], [], [], []
    oid = pay_id = ret_id = 0
    by_signup = sorted(customers, key=lambda c: c[7])
    activity = {c[0]: rng.paretovariate(1.6) for c in customers}
    never = set(rng.sample([c[0] for c in customers], 260))  # customers who sign up and never buy
    d = START
    while d <= END:
        season = 1.0 + 0.9 * (d.month == 11 and d.day >= 20) + 1.2 * (d.month == 12 and d.day <= 22)
        weekend = 1.25 if d.weekday() >= 5 else 1.0
        growth = 1 + (d - START).days / days * 0.8
        n = int(rng.gauss(22, 4) * season * weekend * growth)
        eligible = [c for c in by_signup if c[7] <= d and c[0] not in never]
        if not eligible:
            d += timedelta(days=1)
            continue
        weights = [activity[c[0]] for c in eligible]
        for cust in rng.choices(eligible, weights=weights, k=max(n, 0)):
            oid += 1
            age = (END - d).days
            r = rng.random()
            if r < 0.06:
                status = "cancelled"
            elif age < 3:
                status = "placed"
            elif age < 8 and r < 0.6:
                status = "shipped"
            else:
                status = "delivered"
            channel = rng.choices(["web", "app", "store", "phone"], weights=[46, 30, 18, 6])[0]
            ship = (
                None if status in ("cancelled", "placed") else d + timedelta(days=rng.choice([0, 1, 1, 2, 2, 3, 4, 6]))
            )
            promo = rng.choice(["SPRING10", "WELCOME15", "HOLIDAY20", "FREESHIP"]) if rng.random() < 0.22 else None
            if promo == "HOLIDAY20" and d.month not in (11, 12):
                promo = "WELCOME15"
            lines = rng.choices([1, 2, 3, 4, 5], weights=[45, 28, 15, 8, 4])[0]
            chosen = rng.choices(products, weights=popularity, k=lines)
            total = 0.0
            seen = set()
            ln = 0
            for prod in chosen:
                if prod[0] in seen:
                    continue
                seen.add(prod[0])
                ln += 1
                qty = rng.choices([1, 2, 3, 4, 6], weights=[62, 22, 9, 5, 2])[0]
                if cust[6] == "enterprise":
                    qty *= rng.choice([1, 2, 5])
                disc = {"SPRING10": 0.10, "WELCOME15": 0.15, "HOLIDAY20": 0.20}.get(promo or "", 0.0)
                if not disc and rng.random() < 0.08:
                    disc = 0.05
                price = float(prod[6]) if rng.random() < 0.85 else round(float(prod[6]) * rng.uniform(0.85, 1.0), 2)
                items.append((oid, ln, prod[0], qty, price, disc))
                total += qty * price * (1 - disc)
            orders.append((oid, cust[0], d, status, channel, ship, promo))
            if status != "cancelled":
                pay_id += 1
                paid = datetime.combine(d, datetime.min.time()) + timedelta(minutes=rng.randint(6 * 60, 23 * 60))
                method = rng.choices(["card", "paypal", "gift_card", "bank_transfer"], weights=[64, 22, 7, 7])[0]
                if method == "gift_card" and total > 60 and rng.random() < 0.7:  # split tender
                    payments.append((pay_id, oid, paid, "gift_card", 50.0))
                    pay_id += 1
                    payments.append((pay_id, oid, paid, "card", round(total - 50.0, 2)))
                else:
                    payments.append((pay_id, oid, paid, method, round(total, 2)))
            if status == "delivered" and rng.random() < 0.07:
                line = rng.choice([x for x in items[-ln:]])
                ret_id += 1
                full = line[3] * line[4] * (1 - line[5])
                refund = round(full if rng.random() < 0.8 else full * 0.5, 2)
                returns.append(
                    (
                        ret_id,
                        oid,
                        line[2],
                        (ship or d) + timedelta(days=rng.randint(3, 28)),
                        rng.choices(
                            ["damaged", "wrong_item", "not_as_described", "changed_mind"], weights=[25, 15, 20, 40]
                        )[0],
                        refund,
                    )
                )
        d += timedelta(days=1)
    # orders with every line returned are marked returned
    lines_per_order: dict[int, int] = {}
    for it in items:
        lines_per_order[it[0]] = lines_per_order.get(it[0], 0) + 1
    returned_orders = {r[1] for r in returns if lines_per_order[r[1]] == 1}
    orders = [(o[0], o[1], o[2], "returned" if o[0] in returned_orders else o[3], *o[4:]) for o in orders]
    _insert(con, "orders", orders)
    _insert(con, "order_items", items)
    _insert(con, "payments", payments)
    _insert(con, "returns", [r for r in returns if r[3] <= END])
    return {
        t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]  # type: ignore[index]
        for t in ("customers", "products", "orders", "order_items", "payments", "returns")
    }


def build(path: str | Path) -> dict[str, int]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.unlink(missing_ok=True)
    con = duckdb.connect(str(p))
    try:
        counts = generate(con)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return counts


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "build/warehouse.duckdb"))
