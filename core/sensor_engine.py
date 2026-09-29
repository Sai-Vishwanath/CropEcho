import random
import requests
from datetime import datetime, timezone

class CropEchoEngine:
    def __init__(self, lat, lon, crop_type="Wheat", agro_api_key=None, weather_api_key=None):
        self.lat = lat
        self.lon = lon
        self.crop_type = crop_type
        
        # 1. Store the keys correctly
        self.agro_api_key = agro_api_key
        self.weather_api_key = weather_api_key 
        
        # 2. Fetch base data immediately on init
        self.weather_data = self._get_weather()
        self.ground_truth = self._get_satellite_ground_truth()

    def _get_weather(self):
        """Fetches live weather using the OpenWeather key."""
        # FIX: Using self.weather_api_key instead of self.api_key
        if not self.weather_api_key:
            return {"temp": 32, "humidity": 45, "rain": 0, "wind_speed": 10, "uvi": 8}
            
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={self.lat}&lon={self.lon}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url, timeout=5).json()
            return {
                "temp": response['main']['temp'],
                "humidity": response['main']['humidity'],
                "rain": response.get('rain', {}).get('1h', 0),
                "wind_speed": response['wind']['speed'] * 3.6, # Converted to km/h for alerts
                "uvi": 5 # Fallback if OneCall isn't enabled
            }
        except Exception as e:
            print(f"Weather Fetch Error: {e}")
            return {"temp": 28, "humidity": 50, "rain": 0, "wind_speed": 10, "uvi": 5}

    def _get_satellite_ground_truth(self):
        """Connects to Sentinel-2 data via Agro API."""
        # FIX: Using self.agro_api_key instead of self.api_key
        if not self.agro_api_key: return None
        
        try:
            url = f"http://api.agromonitoring.com/agro/1.0/soil?lat={self.lat}&lon={self.lon}&appid={self.agro_api_key}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                res = response.json()
                return {
                    "moisture": res.get('moisture', 0.4) * 100,
                    "surface_temp": res.get('t0', 300) - 273.15,
                }
        except Exception as e:
            print(f"Satellite Data Error (using physics fallback): {e}")
            
        return {"moisture": 40.0, "surface_temp": 25.0}

    def fetch_weather_alerts(self):
        """NEW: Evaluates the cached weather data for UI Dashboard Toasts."""
        data = self.weather_data
        if not data:
            return {"active": False}
            
        temp = data.get('temp', 25)
        wind = data.get('wind_speed', 0)
        rain = data.get('rain', 0)
        
        # --- AGRONOMIC THRESHOLDS FOR WHEAT ---
        if temp < 4.0:
            return {"active": True, "title": "Frost Warning", "message": f"Local temp dropped to {temp}°C. High risk of frost damage.", "type": "warning"}
        elif temp > 35.0:
            return {"active": True, "title": "Heat Stress Alert", "message": f"Temperature spiked to {temp}°C. Monitor soil moisture.", "type": "danger"}
        elif wind > 40.0:
            return {"active": True, "title": "High Wind Alert", "message": f"Winds detected at {round(wind, 1)} km/h. Risk of crop lodging.", "type": "warning"}
        elif rain > 15.0:
            return {"active": True, "title": "Heavy Rain Warning", "message": f"Intense rainfall ({rain}mm/hr). Check drainage.", "type": "danger"}
            
        return {"active": False}

    def get_growth_stage(self, sowing_date_str="2026-01-15"):
        try:
            sowing_date = datetime.strptime(sowing_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_since_sowing = max(0, (now - sowing_date).days)
        except:
            days_since_sowing = 0 
            
        # --- THE GDD PHYSICS UPGRADE ---
        T_base = 5.0  # Wheat stops growing below 5°C
        
        # Get the live temperature from your OpenWeather cache
        current_temp = self.weather_data.get('temp', 25.0)
        
        # Estimate the daily average (Current temp is usually the daytime high, so we subtract 4°C to estimate the 24hr average)
        estimated_daily_avg = max(T_base, current_temp - 4.0) 
        
        # Calculate daily heat accumulation
        daily_gdd = estimated_daily_avg - T_base
        
        # Total heat accumulated since the seeds were planted
        accumulated_gdd = days_since_sowing * daily_gdd
        
        # Wheat Growth Stages mapped to GDD Thermodynamics instead of pure days
        if accumulated_gdd < 150: return "Germination", 1
        if accumulated_gdd < 500: return "Leaf Development", 2
        if accumulated_gdd < 1000: return "Tillering", 3
        if accumulated_gdd < 1500: return "Stem Elongation", 4
        if accumulated_gdd < 2100: return "Flowering", 5
        
        return "Ripening", 6


    # --- NEW: THE YIELD PREDICTION ENGINE ---
    def predict_yield(self, moisture, ndvi):
        """Calculates estimated yield (tonnes/hectare) based on environmental stress factors."""
        if self.crop_type != "Wheat":
            return "N/A" # We will unlock Cotton and Paddy in V2

        baseline_yield = 4.5 # Optimal tonnes per hectare for Indian Wheat

        # 1. Heat Penalty: Wheat loses yield potential above 30 degrees Celsius
        temp = self.weather_data.get('temp', 25.0)
        heat_penalty = max(0, (temp - 30.0) * 0.12)

        # 2. Drought Penalty: Soil moisture dropping below 40% hurts grain filling
        moisture_penalty = 0
        if moisture < 40.0:
            moisture_penalty = (40.0 - moisture) * 0.05

        # 3. Biological Health Multiplier (NDVI)
        health_factor = min(1.2, max(0.3, ndvi / 0.65))

        # Final Calculation
        projected = (baseline_yield - heat_penalty - moisture_penalty) * health_factor
        
        # Clamp to realistic agronomic extremes (0.5 minimum to 6.5 absolute maximum)
        final_yield = max(0.5, min(6.5, projected))

        return round(final_yield, 2)

    # --- NEW: SUSTAINABILITY & ESG TRACKER ---
    def get_sustainability_metrics(self, projected_yield):
        """Calculates Water Use Efficiency (WUE) and generates an ESG sustainability score."""
        if self.crop_type != "Wheat" or projected_yield == "N/A":
            return {"wue_score": "N/A", "status": "Pending", "water_used": "N/A"}

        # 1. Estimate Water Consumption (Cubic Meters per Hectare)
        # Base water needed for healthy wheat is ~4000 m³/ha (4 million liters)
        base_water = 4000
        
        # Thermodynamics: Extreme heat increases crop transpiration and soil evaporation
        temp = self.weather_data.get('temp', 25.0)
        heat_evaporation_factor = max(0, (temp - 25.0) * 85) # Adds 85 m³ of water loss per degree over 25°C
        
        total_water_used = base_water + heat_evaporation_factor

        # 2. Calculate WUE: Kilograms of wheat produced per 1,000 Liters (1 m³) of water
        # projected_yield is in tonnes (1 tonne = 1000 kg)
        yield_kg = projected_yield * 1000
        wue = yield_kg / total_water_used

        # 3. Benchmark against Indian Regional Average (approx 1.15 kg/m³)
        if wue >= 1.2:
            status = "High Efficiency"
        elif wue >= 0.85:
            status = "Average"
        else:
            status = "Low Efficiency"

        return {
            "wue_score": round(wue, 2),
            "status": status,
            "water_used_m3": int(total_water_used)
        }


    def generate_sensor_data(self):
        """Fuses Real Satellite Data with Physics Logic."""
        temp = self.weather_data['temp']
        uvi = self.weather_data['uvi']
        hum = self.weather_data['humidity']
        
        crop_baselines = {
            "Wheat": {"min": 35, "ideal": 60, "n": 15, "p": 12, "k": 20},
            "Paddy": {"min": 65, "ideal": 85, "n": 25, "p": 18, "k": 15},
            "Cotton": {"min": 25, "ideal": 50, "n": 12, "p": 10, "k": 25}
        }
        profile = crop_baselines.get(self.crop_type, crop_baselines["Wheat"])

        if self.ground_truth:
            base_moisture = self.ground_truth['moisture']
        else:
            evaporation = (temp * 0.15) + (uvi * 0.2) - (hum * 0.08)
            base_moisture = profile["ideal"] - evaporation

        final_moisture = max(10, min(98, base_moisture))

        return {
            "crop": self.crop_type,
            "soil_moisture": round(final_moisture, 2),
            "soil_temp": round(temp - 1.5, 2),
            "nitrogen": round(profile["n"] + random.uniform(-2, 2), 1),
            "phosphorus": round(profile["p"] + random.uniform(-1, 1), 1),
            "potassium": round(profile["k"] + random.uniform(-3, 3), 1),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

    def get_satellite_mock(self):
        data = self.generate_sensor_data()
        moisture = data["soil_moisture"]
        if moisture == 0: return 0.0 
        ndvi = 0.45 + (moisture / 200)
        return round(min(0.9, ndvi), 2)