from flask import Flask, render_template, request, redirect, url_for
import os
import json
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime, timezone
from inference.predict_disease import predict_wheat_disease
from inference.predict_grain import predict_wheat_quality
from core.sensor_engine import CropEchoEngine
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask import jsonify 
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import flash # Add flash to your existing flask imports
from rag_engine import get_rag_advice

# Load the environment variables from the .env file
load_dotenv()

app = Flask(__name__)

# --- CONFIGURATION (SECURE VERSION) ---
# We use os.getenv to pull the keys from your hidden .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AGRO_API_KEY = os.getenv("AGRO_API_KEY")
app.secret_key = os.getenv("SECRET_KEY")
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cropecho.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- SECURITY SETUP ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # If someone tries to access the dashboard, redirect them here
login_manager.login_message_category = "error"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- NEW RELATIONAL MODELS ---

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False) # NEW: Encrypted password
    farms = db.relationship('Farm', backref='owner', lazy=True)

class Farm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    crop_type = db.Column(db.String(50))
    # boundary_coords stores the Polygon/GeoJSON string from the map
    boundary_coords = db.Column(db.Text, nullable=True) 
    center_lat = db.Column(db.Float, nullable=False)
    center_lon = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # Relationship: One farm can have many health records
    records = db.relationship('FieldRecord', backref='farm', lazy=True)

class ScanHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    farm_id = db.Column(db.Integer, db.ForeignKey('farm.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    disease = db.Column(db.String(100))
    advice = db.Column(db.Text)

class FieldRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    moisture = db.Column(db.Float)
    ndvi = db.Column(db.Float)
    advice = db.Column(db.Text)
    # Use the lambda function to ensure it generates a fresh UTC timestamp on every insert
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    farm_id = db.Column(db.Integer, db.ForeignKey('farm.id'), nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    # Only load farms if the user is actually logged in
    if current_user.is_authenticated:
        farms = current_user.farms
    else:
        farms = []
    return render_template('index.html', farms=farms)

# --- AUTHENTICATION ROUTES ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({"error": "Username already exists!"}), 400

        # Create new user with HASHED password
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password_hash=hashed_pw)
        
        db.session.add(new_user)
        db.session.commit()
        
        # Log them in automatically after registering
        login_user(new_user)
        return redirect(url_for('dashboard'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        # Verify the encrypted password
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return jsonify({"error": "Invalid username or password"}), 401

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/add-farm', methods=['POST'])
@login_required # <--- Lock this down!
def add_farm():
    data = request.form
    
    new_farm = Farm(
        name=data.get('farm_name'),
        crop_type=data.get('crop_type'),
        center_lat=float(data.get('lat')),
        center_lon=float(data.get('lon')),
        boundary_coords=data.get('boundary'), 
        user_id=current_user.id # <--- Assign it to the logged-in user!
    )
    db.session.add(new_farm)
    db.session.commit()
    return redirect(url_for('dashboard', farm_id=new_farm.id))

@app.route('/dashboard')
@login_required
def dashboard():
    # 1. Get the current user and their farms
    user = current_user 
    all_farms = user.farms if user.is_authenticated else []
    farm_id = request.args.get('farm_id')
    
    # 2. Check if loading a specific farm profile or utilizing Free Explorer
    if farm_id:
        current_farm = Farm.query.get(farm_id)
        lat = current_farm.center_lat
        lon = current_farm.center_lon
        crop = current_farm.crop_type
        # Extract sowing date if it exists, otherwise fall back to baseline default
        sowing_date_str = current_farm.sowing_date.strftime("%Y-%m-%d") if getattr(current_farm, 'sowing_date', None) else "2026-01-15"
    else:
        lat = round(float(request.args.get('lat', 16.3067)), 3)
        lon = round(float(request.args.get('lon', 80.4365)), 3)
        crop = request.args.get('crop', 'Wheat')
        current_farm = None
        sowing_date_str = "2026-01-15"

    # Update this line to include weather_api_key
    engine = CropEchoEngine(lat, lon, crop_type=crop, agro_api_key=AGRO_API_KEY, weather_api_key=WEATHER_API_KEY)
    
    # NEW: Run the weather scan!
    weather_alert = engine.fetch_weather_alerts()
    # Calculate Growth Stage dynamically
    stage_name, stage_num = engine.get_growth_stage(sowing_date_str) 

    # 3. Handle Telemetry & AI Logic without data cross-contamination
    if current_farm:
        # Fetch the absolute LATEST telemetry point written by your background simulation script
        latest_record = FieldRecord.query.filter_by(farm_id=current_farm.id)\
                                        .order_by(FieldRecord.timestamp.desc())\
                                        .first()
        
        if latest_record:
            # Use the exact state written by the background pipeline (prevents the graph dive)
            sensors = {
                "soil_moisture": latest_record.moisture,
                "nitrogen": 15.0,        # Baselines for UI rendering
                "phosphorus": 11.2,
                "potassium": 21.3
            }
            ndvi = latest_record.ndvi
            ai_advice = latest_record.advice if latest_record.advice else "System synchronized. Monitoring optimal growth conditions."
        else:
            # Fallback initialization state if a new farm has zero simulator rows yet
            sensors = engine.generate_sensor_data()
            ndvi = engine.get_satellite_mock()
            ai_advice = "Initializing sensor links. Waiting for incoming telemetry..."
            
        # Fetch sequential history for chart rendering
        history = FieldRecord.query.filter_by(farm_id=current_farm.id)\
                                   .order_by(FieldRecord.timestamp.desc())\
                                   .limit(15).all()
    else:
        # Free Explorer Mode: Generates synthetic dynamic calculations on-the-fly
        sensors = engine.generate_sensor_data()
        ndvi = engine.get_satellite_mock()
        history = []
        
        # Only query live LLM evaluations for unsaved coordinates to conserve API credits
        try:
            prompt = f"Act as an Indian Agronomist. Analyze structural state of {crop} at {lat}, {lon}. Moisture: {sensors['soil_moisture']}%, NDVI: {ndvi}. Provide a 2-sentence analytical summary."
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            ai_advice = completion.choices[0].message.content
        except:
            ai_advice = "Real-time AI consultant offline. Check regional agronomy updates manually."


    current_moisture = sensors['soil_moisture']
    projected_yield = engine.predict_yield(current_moisture, ndvi)

    # NEW: Run the Sustainability algorithm
    sustainability = engine.get_sustainability_metrics(projected_yield)

    
    # Process sequential graph arrays cleanly
    history_data = [r.moisture for r in reversed(history)]
    history_labels = [r.timestamp.strftime("%H:%M:%S") for r in reversed(history)]

    # 4. Render clean unified state
    return render_template('dashboard.html', 
                           sensors=sensors, 
                           ndvi=ndvi, 
                           lat=lat, 
                           lon=lon, 
                           current_crop=crop,
                           ai_advice=ai_advice,
                           history_data=history_data,
                           history_labels=history_labels,
                           farm=current_farm,
                           all_farms=all_farms,
                           weather_alert=weather_alert,
                           projected_yield=projected_yield,
                           sustainability=sustainability,
                           stage_name=stage_name, 
                           stage_num=stage_num)

@app.route('/disease-check', methods=['GET', 'POST'])
def disease_check():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            try:
                disease, conf = predict_wheat_disease(filepath)
                result_text = f"Identified: {disease.replace('_', ' ').title()} ({conf:.2f}% Confidence)"
            except Exception as e:
                result_text = f"Error running model: {e}"
            
            # Use only the filename for the template to keep paths clean
            return render_template('disease_check.html', result=result_text, image_path=file.filename)
            
    return render_template('disease_check.html', result=None)

@app.route('/grain-check', methods=['GET', 'POST'])
def grain_check():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                detections = predict_wheat_quality(filepath)
                
                healthy_count = sum(1 for d in detections if d['name'] == 'healthy seed')
                bad_count = sum(1 for d in detections if d['name'] == 'bad seed')
                impurity_count = sum(1 for d in detections if d['name'] == 'impurity')
                
                result_text = f"Analysis Complete: {healthy_count} Healthy, {bad_count} Bad, {impurity_count} Impurities."
        
                # POINT TO ANNOTATED IMAGE
                display_image = 'annotated_result.jpg'
                return render_template('grain_check.html', result=result_text, image_path=display_image)

            except Exception as e:
                result_text = f"Error running model: {e}"
                return render_template('grain_check.html', result=result_text, image_path=filename)
            
    return render_template('grain_check.html', result=None)


# --- IoT SENSOR LISTENING ENDPOINT ---
@app.route('/api/sensor/upload', methods=['POST'])
def receive_sensor_data():
    data = request.get_json()
    
    # Extract the simulated data
    farm_id = data.get('farm_id')
    moisture = data.get('moisture')
    ndvi = data.get('ndvi')
    
    # Find the farm to make sure it exists
    farm = Farm.query.get(farm_id)
    if not farm:
        return jsonify({"error": "Farm not found"}), 404

    # Save the incoming telemetry to the database
    new_record = FieldRecord(
        farm_id=farm_id,
        moisture=moisture,
        ndvi=ndvi,
        advice="Live IoT Telemetry reading. Refresh dashboard for AI analysis."
    )
    db.session.add(new_record)
    db.session.commit()
    
    return jsonify({"status": "success", "message": f"Recorded data for {farm.name}"}), 200


# --- REAL-TIME POLLING ENDPOINT ---
@app.route('/api/farm/<int:farm_id>/latest')
def get_latest_sensor_data(farm_id):
    # THE FIX: Added .filter_by(farm_id=farm_id) so it only pulls data for the active dashboard
    latest_record = FieldRecord.query.filter_by(farm_id=farm_id).order_by(FieldRecord.timestamp.desc()).first()
    
    if latest_record:
        return jsonify({
            "moisture": latest_record.moisture,
            "time": latest_record.timestamp.strftime("%H:%M:%S")
        })
    return jsonify({"error": "No data"})


# --- SIMULATOR SYNC ENDPOINT ---
# This lets the Python simulator ask the database what farms exist
@app.route('/api/farms/active')
def get_active_farms():
    farms = Farm.query.all()
    # Returns a simple list of IDs: [1, 2, 3]
    return jsonify([f.id for f in farms])

# --- TRUE FUSION AI ENDPOINT (RAG ENABLED) ---
@app.route('/api/farm/<int:farm_id>/fusion-scan', methods=['POST'])
def fusion_scan(farm_id):
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename != '':
        # 1. Save the leaf image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # 2. Run Vision AI (ResNet-50)
        try:
            disease, conf = predict_wheat_disease(filepath)
        except Exception as e:
            disease = "Unknown (Vision Model Error)"
            
        # 3. Pull the LIVE IoT Data for this farm
        farm = Farm.query.get(farm_id)
        latest_record = FieldRecord.query.filter_by(farm_id=farm_id).order_by(FieldRecord.timestamp.desc()).first()
        moisture = latest_record.moisture if latest_record else "Unknown"
        ndvi = latest_record.ndvi if latest_record else "Unknown"
        
        # 4. RAG-Enabled AI Advice
        try:
            # We now use the RAG engine instead of raw LLM prompting
            ai_advice = get_rag_advice(
                farm_name=farm.name, 
                crop_type=farm.crop_type, 
                moisture=moisture, 
                ndvi=ndvi
            )
        except Exception as e:
            print(f"RAG Error: {e}")
            ai_advice = "The RAG Engine is currently recalibrating."
            
        # 5. Save this critical event to the Farm's Database History
        new_record = FieldRecord(
            moisture=moisture, 
            ndvi=ndvi, 
            advice=f"DISEASE DETECTED: {disease}. {ai_advice}", 
            farm_id=farm_id
        )
        db.session.add(new_record)
        db.session.commit()

        # NEW: Save to Risk Registry
        new_scan = ScanHistory(farm_id=farm_id, disease=disease , advice=ai_advice)
        db.session.add(new_scan)
        db.session.commit()
        
        return jsonify({
            "disease": disease.replace('_', ' ').title(),
            "advice": ai_advice
        })
        
    return jsonify({"error": "Invalid file"}), 400

@app.route('/farm/<int:farm_id>/registry')
def risk_registry(farm_id):
    farm = Farm.query.get_or_404(farm_id)
    # Fetch all past scans for this farm, newest first
    scans = ScanHistory.query.filter_by(farm_id=farm_id).order_by(ScanHistory.timestamp.desc()).all()
    return render_template('risk_registry.html', farm=farm, scans=scans)


if __name__ == '__main__':
    app.run(debug=True)
 