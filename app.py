"""
Retail AI Web Platform
Admin: Tug / Tugdual07
Features: Camera connection, Zone calibration, AI Training (admin only)
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from functools import wraps
import cv2
import numpy as np
import json
import os
import threading
import time
from datetime import datetime
import base64

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'retail-ai-secret-key-2024')
CORS(app)

# Admin credentials (hardcoded as requested)
ADMIN_USER = "Tug"
ADMIN_PASS = "Tugdual07"

# Global camera manager
class CameraManager:
    def __init__(self):
        self.camera = None
        self.camera_id = None
        self.is_streaming = False
        self.frame = None
        self.lock = threading.Lock()
    
    def connect(self, camera_id=0):
        """Connect to camera"""
        try:
            if self.camera is not None:
                self.camera.release()
            
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                return False, "Cannot open camera"
            
            # Test read
            ret, frame = self.camera.read()
            if not ret:
                return False, "Cannot read from camera"
            
            self.camera_id = camera_id
            self.is_streaming = True
            
            # Start capture thread
            threading.Thread(target=self._capture_loop, daemon=True).start()
            
            return True, "Camera connected"
        except Exception as e:
            return False, str(e)
    
    def connect_ip(self, url):
        """Connect to IP/RTSP camera"""
        try:
            if self.camera is not None:
                self.camera.release()
            
            self.camera = cv2.VideoCapture(url)
            if not self.camera.isOpened():
                return False, "Cannot connect to IP camera"
            
            ret, frame = self.camera.read()
            if not ret:
                return False, "Cannot read from IP camera"
            
            self.camera_id = url
            self.is_streaming = True
            
            threading.Thread(target=self._capture_loop, daemon=True).start()
            
            return True, "IP Camera connected"
        except Exception as e:
            return False, str(e)
    
    def _capture_loop(self):
        """Continuously capture frames"""
        while self.is_streaming and self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.033)  # ~30 FPS
    
    def get_frame_base64(self):
        """Get current frame as base64"""
        with self.lock:
            if self.frame is None:
                return None
            ret, jpeg = cv2.imencode('.jpg', self.frame)
            if ret:
                return base64.b64encode(jpeg).decode('utf-8')
        return None
    
    def disconnect(self):
        """Disconnect camera"""
        self.is_streaming = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.camera_id = None
        self.frame = None

camera_mgr = CameraManager()

# Zone configuration storage
zone_config = {
    'zones': [],
    'image': None
}

# Training status
training_status = {
    'is_training': False,
    'progress': 0,
    'epoch': 0,
    'total_epochs': 0,
    'loss': 0,
    'accuracy': 0,
    'message': 'Not started'
}

# Decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or session.get('user') != ADMIN_USER:
            flash('Admin access required')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USER and password == ADMIN_PASS:
            session['user'] = username
            session['is_admin'] = True
            flash('Welcome, Admin!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('is_admin', None)
    flash('Logged out')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', 
                          is_admin=session.get('is_admin', False),
                          camera_connected=camera_mgr.camera is not None)

@app.route('/training')
@login_required
@admin_required
def training_page():
    return render_template('training.html')

# API Routes
@app.route('/api/camera/connect', methods=['POST'])
@login_required
def camera_connect():
    data = request.get_json()
    cam_type = data.get('type', 'usb')
    cam_id = data.get('camera_id', 0)
    
    if cam_type == 'usb':
        success, message = camera_mgr.connect(int(cam_id))
    else:
        url = data.get('url', '')
        success, message = camera_mgr.connect_ip(url)
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/camera/disconnect', methods=['POST'])
@login_required
def camera_disconnect():
    camera_mgr.disconnect()
    return jsonify({'success': True, 'message': 'Camera disconnected'})

@app.route('/api/camera/frame')
@login_required
def camera_frame():
    frame = camera_mgr.get_frame_base64()
    if frame:
        return jsonify({'success': True, 'frame': frame})
    return jsonify({'success': False, 'message': 'No frame available'})

@app.route('/api/camera/status')
@login_required
def camera_status():
    return jsonify({
        'connected': camera_mgr.camera is not None,
        'camera_id': camera_mgr.camera_id
    })

@app.route('/api/zones/save', methods=['POST'])
@login_required
def save_zones():
    global zone_config
    data = request.get_json()
    zone_config['zones'] = data.get('zones', [])
    
    # Save to file
    with open('zones_config.json', 'w') as f:
        json.dump(zone_config, f)
    
    return jsonify({'success': True, 'message': 'Zones saved'})

@app.route('/api/zones/load')
@login_required
def load_zones():
    global zone_config
    try:
        if os.path.exists('zones_config.json'):
            with open('zones_config.json', 'r') as f:
                zone_config = json.load(f)
    except:
        pass
    
    return jsonify(zone_config)

@app.route('/api/training/start', methods=['POST'])
@login_required
@admin_required
def start_training():
    global training_status
    
    if training_status['is_training']:
        return jsonify({'success': False, 'message': 'Training already in progress'})
    
    data = request.get_json()
    training_type = data.get('type', 'detection')
    epochs = int(data.get('epochs', 10))
    
    # Reset status
    training_status = {
        'is_training': True,
        'progress': 0,
        'epoch': 0,
        'total_epochs': epochs,
        'loss': 0,
        'accuracy': 0,
        'message': 'Starting training...'
    }
    
    # Start training in background thread
    threading.Thread(target=training_worker, args=(training_type, epochs), daemon=True).start()
    
    return jsonify({'success': True, 'message': 'Training started'})

@app.route('/api/training/status')
@login_required
@admin_required
def get_training_status():
    return jsonify(training_status)

@app.route('/api/training/stop', methods=['POST'])
@login_required
@admin_required
def stop_training():
    global training_status
    training_status['is_training'] = False
    training_status['message'] = 'Training stopped by user'
    return jsonify({'success': True, 'message': 'Training stopped'})

def training_worker(training_type, epochs):
    """Background training worker"""
    global training_status
    
    try:
        for epoch in range(epochs):
            if not training_status['is_training']:
                break
            
            # Simulate epoch
            time.sleep(2)
            
            training_status['epoch'] = epoch + 1
            training_status['progress'] = int((epoch + 1) / epochs * 100)
            training_status['loss'] = 2.5 - (epoch * 0.2) + np.random.random() * 0.1
            training_status['accuracy'] = 50 + (epoch * 4) + np.random.random() * 2
            training_status['message'] = f'Epoch {epoch+1}/{epochs} - Loss: {training_status["loss"]:.4f}'
        
        if training_status['is_training']:
            training_status['message'] = 'Training completed!'
            
    except Exception as e:
        training_status['message'] = f'Error: {str(e)}'
    
    finally:
        training_status['is_training'] = False

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
