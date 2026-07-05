"""
generate_data.py
-----------------
Creates a synthetic (but realistic) historical dataset of campus events and
their actual attendance. In a real deployment this file would be replaced by
your college's real event-attendance records (from event registration forms,
gate-entry counts, feedback forms, etc). The synthetic generator below encodes
sensible real-world relationships (e.g. Fests draw bigger crowds than Seminars,
rain reduces outdoor turnout, better marketing increases attendance) plus
random noise, so the dataset "behaves" like real data would.

Run directly to (re)create data/campus_events.csv:
    python generate_data.py
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES = 1800

np.random.seed(RANDOM_SEED)

EVENT_TYPES = ["Technical", "Cultural", "Sports", "Workshop", "Seminar", "Fest"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIME_SLOTS = ["Morning", "Afternoon", "Evening"]
WEATHER = ["Sunny", "Cloudy", "Rainy"]
MARKETING = ["Low", "Medium", "High"]

# Relative "pull factor" of each event type (Fests pull the biggest crowds)
EVENT_TYPE_FACTOR = {
    "Technical": 0.85,
    "Cultural": 1.10,
    "Sports": 0.95,
    "Workshop": 0.55,
    "Seminar": 0.45,
    "Fest": 1.55,
}

DAY_FACTOR = {
    "Monday": 0.75, "Tuesday": 0.80, "Wednesday": 0.85,
    "Thursday": 0.90, "Friday": 1.10, "Saturday": 1.25, "Sunday": 1.05,
}

TIME_FACTOR = {"Morning": 0.85, "Afternoon": 1.05, "Evening": 1.20}
WEATHER_FACTOR = {"Sunny": 1.10, "Cloudy": 1.00, "Rainy": 0.70}
MARKETING_FACTOR = {"Low": 0.75, "Medium": 1.00, "High": 1.35}


def generate_row():
    event_type = np.random.choice(EVENT_TYPES)
    day = np.random.choice(DAYS)
    time_slot = np.random.choice(TIME_SLOTS)
    weather = np.random.choice(WEATHER, p=[0.55, 0.30, 0.15])
    marketing = np.random.choice(MARKETING, p=[0.30, 0.45, 0.25])

    venue_capacity = int(np.random.choice(
        [100, 150, 200, 300, 500, 750, 1000, 1500, 2000]
    ))

    social_media_reach = int(np.clip(np.random.normal(3500, 2000), 100, 15000))
    is_holiday = np.random.choice([0, 1], p=[0.85, 0.15])
    competing_event = np.random.choice([0, 1], p=[0.75, 0.25])

    # Past average attendance for this "type of event" (what an organizer would
    # typically know from experience / last year's records)
    past_avg_attendance = np.clip(
        np.random.normal(venue_capacity * 0.5 * EVENT_TYPE_FACTOR[event_type], 60),
        20, venue_capacity * 1.1
    )

    # --- Core formula that "drives" real attendance ---
    base = past_avg_attendance
    base *= EVENT_TYPE_FACTOR[event_type]
    base *= DAY_FACTOR[day]
    base *= TIME_FACTOR[time_slot]
    base *= WEATHER_FACTOR[weather]
    base *= MARKETING_FACTOR[marketing]

    # Social media reach has a diminishing-returns boost
    base += np.log1p(social_media_reach) * 12

    # Holiday boosts turnout a little (students have free time)
    if is_holiday:
        base *= 1.08

    # A competing event on campus/nearby pulls the crowd away
    if competing_event:
        base *= 0.80

    # Random real-world noise
    noise = np.random.normal(0, base * 0.10)
    attendance = base + noise

    # Physically, attendance cannot exceed venue capacity (people get turned away)
    attendance = int(np.clip(attendance, 10, venue_capacity))

    return {
        "event_type": event_type,
        "day_of_week": day,
        "time_slot": time_slot,
        "weather": weather,
        "marketing_effort": marketing,
        "social_media_reach": social_media_reach,
        "venue_capacity": venue_capacity,
        "is_holiday": is_holiday,
        "competing_event": competing_event,
        "past_avg_attendance": round(past_avg_attendance, 1),
        "attendance": attendance,
    }


def main():
    rows = [generate_row() for _ in range(N_SAMPLES)]
    df = pd.DataFrame(rows)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "campus_events.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} historical event records -> {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
