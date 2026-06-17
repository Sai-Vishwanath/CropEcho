import random
import requests
from datetime import datetime, timezone

class CropEchoEngine:
    def __init__(self, lat, lon, crop_type="Wheat", api_key=None):
        self.lat = lat
        self.lon = lon
        self.crop_type = crop_type
        self.api_key = api_key # This is your OpenWeather/Agro key
        self.weather_data = self._get_weather()
        # Fetch Satellite Ground Truth immediately on init
        self.ground_truth = self._get_satellite_ground_truth()

    def _get_weather(self):
        """Stayed mostly the same - keep your local context logic!"""
        if not self.api_key:
            return {"temp": 32, "humidity": 45, "rain": 0, "uvi": 8}
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={self.lat}&lon={self.lon}&appid={self.api_key}&units=metric"
            response = requests.get(url).json()
            return {
                "temp": response['main']['temp'],
                "humidity": response['main']['humidity'],
                "rain": response.get('rain', {}).get('1h', 0),
                "uvi": 5 # Fallback if OneCall 3.0 isn't enabled
            }
        except:
            return {"temp": 28, "humidity": 50, "rain": 0, "uvi": 5}

    def _get_satellite_ground_truth(self):
        """The New Bridge: Connects to Sentinel-2 data via Agro API."""
        if not self.api_key: return None
        try:
            url = f"http://api.agromonitoring.com/agro/1.0/soil?lat={self.lat}&lon={self.lon}&appid={self.api_key}"
            response = requests.get(url, timeout=5) # Added a timeout so the UI doesn't hang!
            if response.status_code == 200:
                res = response.json()
                return {
                    "moisture": res.get('moisture', 0.4) * 100,
                    "surface_temp": res.get('t0', 300) - 273.15,
                }
        except Exception as e:
            print(f"Satellite Data Error (using physics fallback): {e}")
        
        # ALWAYS return a safe default instead of None
        return {"moisture": 40.0, "surface_temp": 25.0}
        

    def get_growth_stage(self, sowing_date_str="2026-01-15"):
        try:
            # Use timezone-aware UTC datetime
            sowing_date = datetime.strptime(sowing_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_since_sowing = max(0, (now - sowing_date).days)
        except:
            days_since_sowing = 0 
    
        # Logic for Wheat
        if days_since_sowing < 30: return "Germination", 1
        if days_since_sowing < 60: return "Leaf Development", 2
        if days_since_sowing < 90: return "Tillering", 3
        if days_since_sowing < 120: return "Stem Elongation", 4
        if days_since_sowing < 150: return "Flowering", 5
    
        return "Ripening", 6

    def generate_sensor_data(self):
        """Fuses Real Satellite Data with your Physics Logic."""
        # 1. Physics Calculations (Your existing logic)
        temp = self.weather_data['temp']
        uvi = self.weather_data['uvi']
        hum = self.weather_data['humidity']
        
        crop_baselines = {
            "Wheat": {"min": 35, "ideal": 60, "n": 15, "p": 12, "k": 20},
            "Paddy": {"min": 65, "ideal": 85, "n": 25, "p": 18, "k": 15},
            "Cotton": {"min": 25, "ideal": 50, "n": 12, "p": 10, "k": 25}
        }
        profile = crop_baselines.get(self.crop_type, crop_baselines["Wheat"])

        # 2. DATA FUSION: If we have satellite data, use it as the base.
        if self.ground_truth:
            base_moisture = self.ground_truth['moisture']
        else:
            # Fallback to your evaporation physics if satellite data is missing
            evaporation = (temp * 0.15) + (uvi * 0.2) - (hum * 0.08)
            base_moisture = profile["ideal"] - evaporation

        # Keep within bounds
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
    
        # In a real app, we handle the 'No Data' state first
        if moisture == 0:
            return 0.0 # Indicate a 'Null' state instead of stress
    
            # Scale NDVI to be more realistic for Indian agriculture
        # Most healthy Indian crops sit between 0.5 and 0.8
        ndvi = 0.45 + (moisture / 200)
    
        return round(min(0.9, ndvi), 2)