# 🌱 CropEcho 

**Tri-Modal AI Fusion: Satellite Telemetry, Ground Sensors, and Computer Vision for Precision Agronomy.**

CropEcho is an advanced, full-stack agricultural intelligence platform designed to simplify farm analytics, identify risks early, and optimize resources. It integrates live satellite data, a proprietary physics-based agronomy engine, and a RAG-enabled LLM to provide farmers and agronomists with highly actionable, localized insights.

## 🚀 Core Features

*   **🛰️ Live Satellite Analytics:** Integrates with AgroMonitoring (Sentinel-2) to pull live NDVI (Normalized Difference Vegetation Index) and simulate surface soil moisture.
*   **🌡️ Thermodynamic Yield Engine:** Replaces standard calendar-based growth tracking with Growing Degree Days (GDD). Predicts wheat yields based on biological age, penalized by live extreme heat or drought conditions via the OpenWeather API.
*   **💧 ESG & Sustainability Tracker:** Calculates real-time Water Use Efficiency (WUE) to help optimize irrigation and maximize harvest tonnage while minimizing resource waste.
*   **📸 True Fusion AI Diagnostics:**
    *   **Disease Check:** ResNet-50 powered leaf scanning.
    *   **Grain Quality Check:** YOLOv5 analysis for seed counting, quality grading, and impurity identification.
*   **🧠 RAG-Enabled LLM Consultant:** Uses ChromaDB and Groq (`llama-3.3-70b-versatile`) to cross-reference identified plant diseases and live IoT telemetry against an embedded agricultural knowledge base to generate instant, agronomist-grade advice.

## 🛠️ Tech Stack

*   **Backend:** Python, Flask, SQLAlchemy, SQLite
*   **AI/ML:** LangChain, Chroma, SentenceTransformers, Groq API, YOLOv5, ResNet-50
*   **APIs:** OpenWeather API, AgroMonitoring API
*   **Frontend:** HTML5, CSS3, Jinja2, Phosphor Icons

## ⚙️ Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR-USERNAME/CropEcho.git](https://github.com/YOUR-USERNAME/CropEcho.git)
cd CropEcho
