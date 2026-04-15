import numpy as np
from sklearn.linear_model import LinearRegression
from stable_baselines3 import DQN
from traffic_env import TrafficLightEnv
import time
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import sys
import threading
import subprocess
import os

# No global video variable needed

# GUI Setup for 4-Way Intersection using PyQt5
class TrafficLightGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Traffic Light - 4-Way Intersection")
        self.setGeometry(100, 100, 600, 200)

        # Directions: North, South, East, West
        self.directions = ['North', 'South', 'East', 'West']
        self.light_labels = {}
        self.timing_labels = {}

        main_layout = QHBoxLayout()

        for direction in self.directions:
            direction_layout = QVBoxLayout()

            direction_label = QLabel(direction)
            direction_label.setFont(QFont("Arial", 16, QFont.Bold))
            direction_layout.addWidget(direction_label)

            light_label = QLabel("🔴")
            light_label.setFont(QFont("Arial", 50))
            self.light_labels[direction] = light_label
            direction_layout.addWidget(light_label)

            timing_label = QLabel("0s")
            timing_label.setFont(QFont("Arial", 20))
            self.timing_labels[direction] = timing_label
            direction_layout.addWidget(timing_label)

            main_layout.addLayout(direction_layout)

        self.setLayout(main_layout)

    def update_lights(self, lights):
        for dir, color in lights.items():
            self.light_labels[dir].setText(color)

    def update_timings(self, timings):
        for dir, timing in timings.items():
            self.timing_labels[dir].setText(timing)

# Worker thread for traffic system logic
class TrafficWorker(QThread):
    update_lights_signal = pyqtSignal(dict)
    update_timings_signal = pyqtSignal(dict)

    def __init__(self, lr_model, rl_model, use_rl):
        super().__init__()
        self.lr_model = lr_model
        self.rl_model = rl_model
        self.use_rl = use_rl
        self.emergency_count = 0
        self.MAX_EMERGENCIES = 2

    def run(self):
        try:
            # Start video playback in background (Windows)
            video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traffic.mp4')
            video_process = subprocess.Popen(['start', '', video_path], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            last_update_time = time.time()
            cycle_index = 0  # 0: North-South, 1: East-West
            cycle_counter = 0  # Counter for ambulance arrival (every 3rd cycle)

            while True:
                current_time = time.time()

                # Update every 5 seconds
                if current_time - last_update_time >= 5:
                    # Simulate vehicle counts for all directions (since no video detection)
                    vehicle_count_north = np.random.randint(0, 50)
                    vehicle_count_south = np.random.randint(0, 50)
                    vehicle_count_east = np.random.randint(0, 50)
                    vehicle_count_west = np.random.randint(0, 50)

                    # Emergency simulation - ambulance arrives every 2nd cycle
                    ambulance_cycle = (cycle_counter % 2 == 1)  # Every 2nd cycle has ambulance
                    
                    emergency_detected_north = False
                    emergency_detected_south = False
                    emergency_detected_east = False
                    emergency_detected_west = False
                    
                    if self.emergency_count < self.MAX_EMERGENCIES and ambulance_cycle:
                        # Only generate emergency if under limit
                        emergency_dir = np.random.choice(['North', 'South', 'East', 'West'])
                        if emergency_dir == 'North':
                            emergency_detected_north = True
                        elif emergency_dir == 'South':
                            emergency_detected_south = True
                        elif emergency_dir == 'East':
                            emergency_detected_east = True
                        else:
                            emergency_detected_west = True
                        self.emergency_count += 1
                        print(f"Emergency #{self.emergency_count} triggered in pre-cycle: {emergency_dir}")

                    vehicle_counts_dict = {
                        'North': vehicle_count_north,
                        'South': vehicle_count_south,
                        'East': vehicle_count_east,
                        'West': vehicle_count_west
                    }

                    emergency_dict = {
                        'North': emergency_detected_north,
                        'South': emergency_detected_south,
                        'East': emergency_detected_east,
                        'West': emergency_detected_west
                    }

                    # Determine active directions based on cycle
                    if cycle_index == 0:
                        active_directions = ['North', 'South']
                        inactive_directions = ['East', 'West']
                    else:
                        active_directions = ['East', 'West']
                        inactive_directions = ['North', 'South']

                    # Check for ambulance/emergency - only one signal opens for ambulance
                    ambulance_direction = None
                    for dir in ['North', 'South', 'East', 'West']:
                        if emergency_dict[dir]:
                            ambulance_direction = dir
                            break
                    
                    # If ambulance detected, only that direction gets green
                    if ambulance_direction:
                        active_directions = [ambulance_direction]
                        inactive_directions = [d for d in ['North', 'South', 'East', 'West'] if d != ambulance_direction]
                        print(f"AMBULANCE DETECTED - Only {ambulance_direction} signal will be GREEN")

                    # Predict green time for active directions using Linear Regression
                    start_time = time.time()
                    total_vehicles_active = sum(vehicle_counts_dict[dir] for dir in active_directions)
                    predicted_green = self.lr_model.predict([[total_vehicles_active]])[0]
                    # Cap the green time at maximum 35 seconds
                    predicted_green = min(predicted_green, 35)
                    prediction_time = time.time() - start_time
                    print(f"LR Predicted: {predicted_green:.1f}s for {total_vehicles_active} vehicles (Prediction time: {prediction_time:.4f}s)")
                    emergency_active = any(emergency_dict[dir] for dir in active_directions)
                    if emergency_active:
                        predicted_green = 35  # Override for emergency (changed from 30 to 35)

                    # Console output
                    print(f"Cycle: {'North-South' if cycle_index == 0 else 'East-West'}")
                    for dir in ['North', 'South', 'East', 'West']:
                        print(f"{dir} - Vehicles: {vehicle_counts_dict[dir]}, Emergency: {emergency_dict[dir]}")
                    print(f"Predicted green time: {predicted_green:.1f}s")

                    # Simulate traffic light cycle
                    remaining_time = predicted_green
                    
                    # Monitor for emergency during the cycle
                    while remaining_time > 0:
                        # Check for emergency at each second - if detected, immediately respond
                        current_emergency = False
                        emergency_dir = None
                        
                        # Random chance to detect emergency during cycle
                        if self.emergency_count < self.MAX_EMERGENCIES and np.random.random() < 0.05:  # 5% chance per second
                            # Check which direction has emergency
                            for dir in ['North', 'South', 'East', 'West']:
                                if emergency_dict.get(dir, False):
                                    current_emergency = True
                                    emergency_dir = dir
                                    break
                            # If no pre-existing emergency, randomly generate one
                            if not current_emergency:
                                emergency_dir = np.random.choice(['North', 'South', 'East', 'West'])
                                current_emergency = True
                                emergency_dict[emergency_dir] = True
                        
                        # IMMEDIATE EMERGENCY RESPONSE - interrupt cycle immediately
                        if current_emergency and emergency_dir:
                            print(f"\n🚨 EMERGENCY DETECTED! {emergency_dir} direction - IMMEDIATE RESPONSE 🚨")
                            print(f"Interrupting cycle - all signals RED except {emergency_dir}")
                            
                            # Immediately show green for emergency direction, red for all others
                            lights = {}
                            timings = {}
                            for dir in ['North', 'South', 'East', 'West']:
                                if dir == emergency_dir:
                                    lights[dir] = "🟢"
                                    timings[dir] = "35s (EMERGENCY)"
                                else:
                                    lights[dir] = "🔴"
                                    timings[dir] = "0s"
                            self.update_lights_signal.emit(lights)
                            self.update_timings_signal.emit(timings)
                            
                            # Hold emergency green for 35 seconds
                            for sec in range(35, 0, -1):
                                lights = {}
                                timings = {}
                                for dir in ['North', 'South', 'East', 'West']:
                                    if dir == emergency_dir:
                                        lights[dir] = "🟢"
                                        timings[dir] = f"{sec}s"
                                    else:
                                        lights[dir] = "🔴"
                                        timings[dir] = "0s"
                                self.update_lights_signal.emit(lights)
                                self.update_timings_signal.emit(timings)
                                time.sleep(1)
                            
                            # Orange phase for emergency direction - 5 seconds
                            print("Emergency green complete - Orange phase for 5s")
                            for sec in range(5, 0, -1):
                                lights = {}
                                timings = {}
                                for dir in ['North', 'South', 'East', 'West']:
                                    if dir == emergency_dir:
                                        lights[dir] = "🟡"
                                        timings[dir] = f"{sec}s"
                                    else:
                                        lights[dir] = "🔴"
                                        timings[dir] = "0s"
                                self.update_lights_signal.emit(lights)
                                self.update_timings_signal.emit(timings)
                                time.sleep(1)
                            
                            print(f"Emergency vehicle passed - returning to NORMAL operation")
                            if self.emergency_count >= self.MAX_EMERGENCIES:
                                print("Max emergencies (2) reached. No more emergencies will occur.")
                            
                            # Brief all red for safety then next cycle
                            lights = {}
                            timings = {}
                            for dir in ['North', 'South', 'East', 'West']:
                                lights[dir] = "🔴"
                                timings[dir] = "0s"
                            self.update_lights_signal.emit(lights)
                            self.update_timings_signal.emit(timings)
                            time.sleep(1)
                            
                            # Emergency cycle complete, move to next cycle
                            break

                        # Normal cycle - show green for active directions
                        lights = {}
                        timings = {}
                        for dir in ['North', 'South', 'East', 'West']:
                            if dir in active_directions:
                                lights[dir] = "🟢"
                                timings[dir] = f"{int(remaining_time)}s"
                            else:
                                lights[dir] = "🔴"
                                timings[dir] = "0s"
                        self.update_lights_signal.emit(lights)
                        self.update_timings_signal.emit(timings)
                        time.sleep(1)
                        remaining_time -= 1
                    
                    # Skip remaining cycle logic if emergency broke out of loop
                    if current_emergency and emergency_dir:
                        last_update_time = current_time
                        cycle_index = (cycle_index + 1) % 2
                        cycle_counter += 1
                        continue

                    # Orange/Yellow phase - 5 seconds
                    lights = {}
                    timings = {}
                    for dir in ['North', 'South', 'East', 'West']:
                        if dir in active_directions:
                            lights[dir] = "🟡"
                            timings[dir] = "5s"
                        else:
                            lights[dir] = "🔴"
                            timings[dir] = "0s"
                    self.update_lights_signal.emit(lights)
                    self.update_timings_signal.emit(timings)
                    time.sleep(5)

                    # Red phase for all
                    lights = {}
                    timings = {}
                    for dir in ['North', 'South', 'East', 'West']:
                        lights[dir] = "🔴"
                        timings[dir] = "0s"
                    self.update_lights_signal.emit(lights)
                    self.update_timings_signal.emit(timings)
                    time.sleep(5)

                    last_update_time = current_time
                    cycle_index = (cycle_index + 1) % 2  # Toggle cycle
                    cycle_counter += 1  # Increment cycle counter for ambulance timing

        except Exception as e:
            print(f"Error in traffic system: {e}")
        finally:
            # Cleanup
            try:
                video_process.terminate()
            except:
                pass

# Generate synthetic training data for ML model
# Vehicle count from 0 to 50, green time = count * 2 + 10 seconds
vehicle_data = np.arange(0, 51).reshape(-1, 1)
green_times = vehicle_data.flatten() * 2 + 10

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(vehicle_data, green_times)
print(f"LR Model coef: {lr_model.coef_[0]:.2f}, intercept: {lr_model.intercept_:.2f}")

# Evaluate model performance
from sklearn.metrics import mean_squared_error, r2_score
predictions = lr_model.predict(vehicle_data)
mse = mean_squared_error(green_times, predictions)
r2 = r2_score(green_times, predictions)
print(f"Model Performance - MSE: {mse:.2f}, R²: {r2:.2f}")

# Load RL model if available
try:
    rl_model = DQN.load("traffic_light_rl")
    use_rl = True
    print("Using RL model for traffic light timing")
except:
    rl_model = None
    use_rl = False
    print("Using Linear Regression for traffic light timing")

app = QApplication(sys.argv)
gui = TrafficLightGUI()
gui.show()

# Create and start the worker thread
worker = TrafficWorker(lr_model, rl_model, use_rl)
worker.update_lights_signal.connect(gui.update_lights)
worker.update_timings_signal.connect(gui.update_timings)
worker.start()

# Start the GUI main loop
sys.exit(app.exec_())
