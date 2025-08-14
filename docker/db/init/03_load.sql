-- Seed few rows if table empty (first boot safety)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM yellow_trips) THEN
        COPY yellow_trips (
            pickup_datetime,
            dropoff_datetime,
            passenger_count,
            trip_distance,
            pickup_location_id,
            dropoff_location_id,
            fare_amount,
            tip_amount,
            total_amount,
            payment_type
        )
        FROM '/docker-entrypoint-initdb.d/02_trips.csv'
        WITH (FORMAT csv, HEADER true);
    END IF;
END $$;
