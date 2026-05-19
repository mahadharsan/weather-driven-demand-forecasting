# ingestion/schema.py
import pyarrow as pa

# Explicitly define fields with strict nullability constraints
WEATHER_SCHEMA = pa.schema([
    pa.field('date', pa.string(), nullable=False),              # Temporal anchor; must exist
    pa.field('city', pa.string(), nullable=False),              # Partition anchor; must exist
    pa.field('temperature_max', pa.float32(), nullable=False),   # Core metric; null means data failure
    pa.field('temperature_min', pa.float32(), nullable=False),   # Core metric; null means data failure
    pa.field('precipitation_sum', pa.float32(), nullable=True),  # Acceptable null (e.g., sensor malfunction/no rain data)
    pa.field('wind_speed_max', pa.float32(), nullable=True),     # Acceptable null
    pa.field('weather_code', pa.int32(), nullable=True)          # Acceptable null
])