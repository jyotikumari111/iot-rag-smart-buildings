import os
import psycopg
from typing import List, Dict, Any

PG_URL = os.getenv("PG_URL", "postgresql://postgres:postgres@timescaledb:5432/smart_building")

def fetch_kpis(site_id: str | None, equip_id: str | None) -> List[Dict[str, Any]]:
    if not site_id or not equip_id:
        return []
    q = '''
    with rollups as (
      select time_bucket('5 minutes', ts) as tb,
             avg(value) as mean_v, stddev_pop(value) as std_v
      from telemetry
      where site_id=%s and equip_id=%s and ts> now() - interval '1 hour'
      group by tb order by tb desc limit 12
    )
    select tb, mean_v, std_v from rollups order by tb desc;
    '''
    with psycopg.connect(PG_URL) as conn:
        rows = conn.execute(q, (site_id, equip_id)).fetchall()
    return [dict(ts=r[0].isoformat(), mean=float(r[1] or 0.0), std=float(r[2] or 0.0)) for r in rows]
