-- Create a simple Yellow Taxi trips table
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

-- Helpful index for time filtering
CREATE INDEX IF NOT EXISTS idx_yellow_trips_pickup_datetime
ON yellow_trips (pickup_datetime);
