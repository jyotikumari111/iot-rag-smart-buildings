import os, time, random, psycopg
from datetime import datetime, timezone, timedelta

PG_URL = os.getenv("PG_URL", "postgresql://postgres:postgres@timescaledb:5432/smart_building")

site = "BLDG_A"
equip = "AHU_03"
sensors = [
    ("supply_air_temp", "C", 12.0, 0.6),
    ("return_air_temp", "C", 24.0, 0.7),
    ("fan_power_kw", "kW", 5.0, 0.8)
]

def insert_point(conn, site_id, equip_id, sensor, ts, value, unit, status="OK"):
    conn.execute(
        "INSERT INTO telemetry(site_id, equip_id, sensor, ts, value, unit, status) VALUES (%s,%s,%s,%s,%s,%s,%s)",
        (site_id, equip_id, sensor, ts, value, unit, status)
    )

def main():
    with psycopg.connect(PG_URL, autocommit=True) as conn:
        while True:
            ts = datetime.now(timezone.utc)
            for s, unit, mu, sigma in sensors:
                val = random.gauss(mu, sigma)
                insert_point(conn, site, equip, s, ts, val, unit, "OK")
            time.sleep(2)

if __name__ == "__main__":
    main()
