"""Runs every query in sql/analysis.sql against the telemetry database and
prints the results as tables.   Usage:  python sql/run_analysis.py [db_path]"""
import os
import re
import sqlite3
import sys

here = os.path.dirname(os.path.abspath(__file__))
db_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TELEMETRY_DB_PATH", "rag_telemetry.db")

sql = open(os.path.join(here, "analysis.sql"), encoding="utf-8").read()
blocks = re.split(r"^-- name: (.+)$", sql, flags=re.M)[1:]   # [name, body, name, body, ...]

conn = sqlite3.connect(db_path)
for name, body in zip(blocks[0::2], blocks[1::2]):
    cur = conn.execute(body)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    widths = [max(len(str(x)) for x in [c] + [r[i] for r in rows]) for i, c in enumerate(cols)]
    print(f"\n=== {name.strip()} ===")
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(("" if v is None else str(v)).ljust(w) for v, w in zip(r, widths)))
conn.close()
