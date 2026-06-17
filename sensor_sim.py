import time
import random
import requests

API_URL = "http://127.0.0.1:5000/api/sensor/upload"
SYNC_URL = "http://127.0.0.1:5000/api/farms/active"

farm_states = {}

print("🌱 Starting Smart IoT Simulator...")
print("Listening for new farms drawn on the dashboard...\n")

while True:
    # 1. Sync with the database to find all active farms
    try:
        active_ids = requests.get(SYNC_URL).json()
    except Exception as e:
        print("⚠️ Cannot reach Flask server. Is it running?")
        time.sleep(5)
        continue
        
    # 2. If a new farm was drawn, initialize a sensor state for it!
    for f_id in active_ids:
        if f_id not in farm_states:
            farm_states[f_id] = {
                "moisture": random.uniform(40.0, 60.0), # Random start moisture
                "ndvi": random.uniform(0.55, 0.70)      # Random start health
            }
            print(f"✅ New Hardware Provisioned! Simulating Farm ID: {f_id}")

    # 3. Generate and transmit data for ALL active farms
    for f_id in active_ids:
        state = farm_states[f_id]
        
        # Physics fluctuation
        state["moisture"] += random.uniform(-1.5, 1.0) 
        state["moisture"] = max(10, min(90, state["moisture"])) 
        state["ndvi"] += random.uniform(-0.01, 0.01)
        state["ndvi"] = max(0.1, min(0.9, state["ndvi"]))

        payload = {
            "farm_id": f_id,
            "moisture": round(state["moisture"], 1),
            "ndvi": round(state["ndvi"], 3),
            "n": int(random.uniform(30, 60)),
            "p": int(random.uniform(15, 30)),
            "k": int(random.uniform(20, 40))
        }

        try:
            requests.post(API_URL, json=payload)
            print(f"📡 Farm {f_id} Transmitted | Moisture: {payload['moisture']}%")
        except:
            pass

    print("--- Waiting 5 seconds ---")
    time.sleep(5)