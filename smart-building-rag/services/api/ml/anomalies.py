from typing import List, Dict, Any, Optional
import os, psycopg

PG_URL = os.getenv("PG_URL", "postgresql://postgres:postgres@timescaledb:5432/smart_building")

def evaluate_point(site_id: str, equip_id: str, sensor: str, window_minutes: int = 15) -> Dict[str, Any]:
    q = '''
    select avg(value), stddev_pop(value)
    from telemetry
    where site_id=%s and equip_id=%s and sensor=%s and ts> now() - interval %s;
    '''
    with psycopg.connect(PG_URL) as conn:
        row = conn.execute(q, (site_id, equip_id, sensor, f"{window_minutes} minutes")).fetchone()
    mean_v = float(row[0] or 0.0)
    std_v = float(row[1] or 0.0)
    severity = "OK"
    if std_v > 0 and abs(mean_v) > 3 * std_v:
        severity = "WARN"
    return {"site_id": site_id, "equip_id": equip_id, "sensor": sensor, "mean": mean_v, "std": std_v, "severity": severity}

def recent_alerts(site_id: Optional[str], equip_id: Optional[str]) -> List[Dict[str, Any]]:
    # For demo, no persisted alerts; a real build would read from an alerts table.
    return []
