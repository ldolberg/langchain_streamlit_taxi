import argparse
import io
import os
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

# TLC base URL for yellow taxi tripdata parquet files
# Example: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet
TLC_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NYC TLC Yellow Taxi data and load into Postgres")
    parser.add_argument("--month", required=True, help="Month in YYYY-MM format, e.g., 2019-01")
    parser.add_argument("--limit", type=int, default=50000, help="Limit rows to load (for POC)")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/taxi"),
        help="SQLAlchemy database URL",
    )
    return parser.parse_args()


def ensure_schema(engine) -> None:
    create_sql = """
    CREATE TABLE IF NOT EXISTS yellow_trips (
        trip_id SERIAL PRIMARY KEY,
        pickup_datetime TIMESTAMP NOT NULL,
        dropoff_datetime TIMESTAMP NOT NULL,
        passenger_count INTEGER,
        trip_distance NUMERIC(8,2),
        pickup_location_id INTEGER,
        dropoff_location_id INTEGER,
        fare_amount NUMERIC(10,2),
        tip_amount NUMERIC(10,2),
        total_amount NUMERIC(10,2),
        payment_type TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_yellow_trips_pickup_datetime ON yellow_trips (pickup_datetime);
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Map TLC columns to our schema
    rename_map = {
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
        "passenger_count": "passenger_count",
        "trip_distance": "trip_distance",
        "PULocationID": "pickup_location_id",
        "DOLocationID": "dropoff_location_id",
        "fare_amount": "fare_amount",
        "tip_amount": "tip_amount",
        "total_amount": "total_amount",
        "payment_type": "payment_type",
    }
    keep_cols = list(rename_map.values())
    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    missing = set(keep_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns from TLC data: {missing}")

    # Cast dtypes and clean
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")
    for col in ["passenger_count", "pickup_location_id", "dropoff_location_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["trip_distance", "fare_amount", "tip_amount", "total_amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["payment_type"] = df["payment_type"].astype("string")

    # Drop rows with invalid datetimes
    df = df.dropna(subset=["pickup_datetime", "dropoff_datetime"])  # type: ignore

    return df[keep_cols]


def load_month(engine, month: str, limit: int) -> int:
    url = f"{TLC_BASE}/yellow_tripdata_{month}.parquet"
    print(f"Downloading {url}")
    df = pd.read_parquet(url)
    if limit:
        df = df.head(limit)
    df = normalize_dataframe(df)

    # Insert into DB
    with engine.begin() as conn:
        df.to_sql("yellow_trips", con=conn, if_exists="append", index=False, method="multi", chunksize=5000)
    return len(df)


def main():
    args = parse_args()
    engine = create_engine(args.database_url)
    ensure_schema(engine)
    inserted = load_month(engine, args.month, args.limit)
    print(f"Inserted rows: {inserted}")


if __name__ == "__main__":
    main()
