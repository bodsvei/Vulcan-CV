"""
Vulcan Unified AI System
Combines face identification, expression recognition, pose estimation, 
hand tracking, voice interaction, and Arduino control

Features:
- Real-time face detection and identification with database
- Facial expression recognition
- Pose estimation with posture analysis
- Hand skeleton tracking
- Voice conversation with speech-to-text and text-to-speech
- RAG-based conversational AI
- Arduino servo control for eye movement and mouth animation
- Visualization grids for pose and hand tracking

Authors: Aaditya Kushawaha, Anirudh Singh Air
Project: Vulcan ERC Humanoid
"""

import cv2
import numpy as np
import pickle
import os
import json
import time
import struct
from datetime import datetime
from scipy.spatial.distance import cosine, euclidean
from pathlib import Path
from collections import deque
import concurrent.futures
import multiprocessing
import multiprocessing.shared_memory
from threading import Event
import threading

# Voice and AI imports
import speech_recognition as sr
import pyttsx3
import serial

# Custom modules (ensure these are available)
try:
    import m_face as face_de
    import m_expression as exp_re
    CUSTOM_FACE_MODULES = True
except ImportError:
    print("⚠️ Custom face modules not found. Using OpenCV only.")
    CUSTOM_FACE_MODULES = False

try:
    from RAG_LLM.app_lang2 import response_rag
    RAG_AVAILABLE = True
except ImportError:
    print("⚠️ RAG module not available")
    RAG_AVAILABLE = False

# MediaPipe for pose and hand tracking
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not installed. Install with: pip install mediapipe")


class VulcanUnifiedSystem:
    """Unified system combining all Vulcan functionalities"""
    
    def __init__(self,
                 video_source="http://192.168.154.230:81/stream",
                 arduino_port='COM8',
                 arduino_baudrate=115200,
                 db_path='face_database.pkl',
                 similarity_threshold=0.55,
                 microphone_index=18,
                 enable_pose=True,
                 enable_hands=True,
                 enable_arduino=True):
        """
        Initialize the Vulcan Unified System
        
        Args:
            video_source: IP camera URL or device index (0 for webcam)
            arduino_port: Serial port for Arduino communication
            arduino_baudrate: Baud rate for Arduino
            db_path: Path to face database
            similarity_threshold: Face matching threshold
            microphone_index: Microphone device index
            enable_pose: Enable pose estimation
            enable_hands: Enable hand tracking
            enable_arduino: Enable Arduino communication
        """
        # Video capture
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            print(f"⚠️ Warning: Could not open video source {video_source}")
        
        # Arduino setup
        self.enable_arduino = enable_arduino
        self.arduino = None
        if enable_arduino:
            try:
                self.arduino = serial.Serial(port=arduino_port, baudrate=arduino_baudrate, timeout=1)
                print(f"✓ Arduino connected on {arduino_port}")
            except Exception as e:
                print(f"⚠️ Arduino connection failed: {e}")
                self.enable_arduino = False
        
        # Face database
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.face_db = {}
        self.next_id = 1
        self.detection_history = {}
        self.history_size = 5
        self.load_database()
        
        # Face detection and recognition
        self.detector = self._load_face_detector()
        self.recognizer = self._load_face_recognizer()
        
        # Expression recognition
        self.glob_expression = "neutral"
        self.glob_name = ""
        
        # Conversation state
        self.conversation_started = False
        self.terminate_event = Event()
        
        # Voice recognition
        self.recognizer = sr.Recognizer()
        self.microphone_index = microphone_index
        
        # Shared memory for phrase detection
        try:
            self.shm = multiprocessing.shared_memory.SharedMemory(create=True, size=250)
        except:
            print("⚠️ Shared memory creation failed")
            self.shm = None
        
        # MediaPipe setup
        self.enable_pose = enable_pose and MEDIAPIPE_AVAILABLE
        self.enable_hands = enable_hands and MEDIAPIPE_AVAILABLE
        self.show_grids = True
        self.grid_width = 400
        self.grid_height = 300
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            if self.enable_pose:
                self.pose = self.mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=1
                )
                print("✓ Pose estimation initialized")
            
            if self.enable_hands:
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("✓ Hand tracking initialized")
        
        # Eye tracking parameters
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        self.tolerance_x = self.frame_width * 0.08
        self.tolerance_y = self.frame_height * 0.08
        
        # Timing for periodic Arduino commands
        self.last_command_time = time.time()
        self.last_command_time1 = time.time()
        
        print(f"✓ System initialized with {len(self.face_db)} known faces")
    
    def _load_face_detector(self):
        """Load OpenCV DNN face detector"""
        try:
            proto_path = "deploy.prototxt"
            model_path = "res10_300x300_ssd_iter_140000.caffemodel"
            
            if os.path.exists(proto_path) and os.path.exists(model_path):
                detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                print("✓ Loaded Caffe face detector")
            else:
                detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                print("✓ Using Haar Cascade detector")
            return detector
        except Exception as e:
            print(f"⚠️ Error loading detector: {e}")
            return cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def _load_face_recognizer(self):
        """Load face recognition model"""
        try:
            model_path = "openface_nn4.small2.v1.t7"
            if os.path.exists(model_path):
                recognizer = cv2.dnn.readNetFromTorch(model_path)
                print("✓ Loaded OpenFace recognition model")
                return recognizer
            else:
                print("⚠️ Recognition model not found. Using simple feature extraction.")
                return None
        except Exception as e:
            print(f"⚠️ Error loading recognizer: {e}")
            return None
    
    def send_command_to_arduino(self, command):
        """Send command to Arduino"""
        if self.enable_arduino and self.arduino:
            try:
                self.arduino.write(f"{command}\n".encode('utf-8'))
                print(f"Arduino command sent: {command}")
            except Exception as e:
                print(f"⚠️ Failed to send Arduino command: {e}")
    
    def text_to_speech(self, text):
        """Convert text to speech using Microsoft Ravi voice"""
        try:
            engine = pyttsx3.init()
            for voice in engine.getProperty('voices'):
                if "RAVI" in voice.id.upper():
                    engine.setProperty('voice', voice.id)
                    break
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"⚠️ TTS error: {e}")
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        if isinstance(self.detector, cv2.CascadeClassifier):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return [(x, y, w, h, 1.0) for (x, y, w, h) in faces]
        else:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0)
            )
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
            return faces
    
    def get_embedding(self, face_img):
        """Extract face embedding"""
        if self.recognizer is not None:
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0/255, (96, 96), (0, 0, 0),
                swapRB=True, crop=False
            )
            self.recognizer.setInput(blob)
            embedding = self.recognizer.forward()
            return embedding.flatten()
        else:
            face_resized = cv2.resize(face_img, (128, 128))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            embedding = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
            return embedding.flatten() / np.linalg.norm(embedding.flatten())
    
    def get_expression(self, gray):
        """Get facial expression"""
        if CUSTOM_FACE_MODULES:
            try:
                return exp_re.get_expression(gray)
            except:
                return "neutral"
        return "neutral"
    
    def match_or_create_id(self, embedding):
        """Match embedding or create new ID"""
        if not self.face_db:
            person_id = f"P{self.next_id:03d}"
            self.next_id += 1
            return person_id, True, 0.0
        
        best_match_id = None
        best_distance = float('inf')
        
        for person_id, data in self.face_db.items():
            distance = cosine(embedding, data['embedding'])
            if distance < best_distance:
                best_distance = distance
                best_match_id = person_id
        
        confidence = 1 - min(best_distance, 1.0)
        
        if best_distance < self.similarity_threshold:
            return best_match_id, False, confidence
        else:
            person_id = f"P{self.next_id:03d}"
            self.next_id += 1
            return person_id, True, 0.0
    
    def update_database(self, person_id, embedding, is_new=False):
        """Update face database"""
        current_time = datetime.now().isoformat()
        
        if person_id not in self.detection_history:
            self.detection_history[person_id] = deque(maxlen=self.history_size)
        
        self.detection_history[person_id].append(embedding)
        avg_embedding = np.mean(list(self.detection_history[person_id]), axis=0)
        
        if is_new:
            self.face_db[person_id] = {
                'embedding': avg_embedding,
                'first_seen': current_time,
                'last_seen': current_time,
                'appearances': 1,
                'name': None,
                'is_erc_member': False
            }
        else:
            self.face_db[person_id]['last_seen'] = current_time
            self.face_db[person_id]['appearances'] += 1
            old_emb = self.face_db[person_id]['embedding']
            self.face_db[person_id]['embedding'] = 0.85 * old_emb + 0.15 * avg_embedding
    
    def create_pose_grid(self, pose_results):
        """Create visualization grid for pose"""
        grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        cv2.rectangle(grid, (0, 0), (self.grid_width, self.grid_height), (30, 30, 30), -1)
        
        # Draw grid lines
        for i in range(0, self.grid_width, 50):
            cv2.line(grid, (i, 0), (i, self.grid_height), (50, 50, 50), 1)
        for i in range(0, self.grid_height, 50):
            cv2.line(grid, (0, i), (self.grid_width, i), (50, 50, 50), 1)
        
        cv2.putText(grid, "Posture", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if not pose_results or not pose_results.pose_landmarks:
            cv2.putText(grid, "No pose detected", (self.grid_width//2 - 80, self.grid_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            return grid
        
        landmarks = pose_results.pose_landmarks.landmark
        points = []
        for lm in landmarks:
            x = int(lm.x * self.grid_width)
            y = int(lm.y * self.grid_height)
            points.append((x, y, lm.visibility))
        
        # Draw connections
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            if (start_idx < len(points) and end_idx < len(points) and
                points[start_idx][2] > 0.5 and points[end_idx][2] > 0.5):
                cv2.line(grid, points[start_idx][:2], points[end_idx][:2],
                        (0, 255, 0), 2)
        
        # Draw landmarks
        for idx, (x, y, vis) in enumerate(points):
            if vis > 0.5:
                if idx in [11, 12, 23, 24]:
                    color = (0, 0, 255)
                elif idx in [13, 14, 15, 16]:
                    color = (255, 0, 0)
                elif idx in [25, 26, 27, 28]:
                    color = (0, 255, 255)
                else:
                    color = (255, 255, 255)
                cv2.circle(grid, (x, y), 4, color, -1)
        
        return grid
    
    def create_hand_grid(self, hand_results):
        """Create visualization grid for hands"""
        grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        cv2.rectangle(grid, (0, 0), (self.grid_width, self.grid_height), (30, 30, 30), -1)
        
        for i in range(0, self.grid_width, 50):
            cv2.line(grid, (i, 0), (i, self.grid_height), (50, 50, 50), 1)
        for i in range(0, self.grid_height, 50):
            cv2.line(grid, (0, i), (self.grid_width, i), (50, 50, 50), 1)
        
        cv2.putText(grid, "Hand Tracking", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if not hand_results or not hand_results.multi_hand_landmarks:
            cv2.putText(grid, "No hands detected", (self.grid_width//2 - 90, self.grid_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            return grid
        
        num_hands = len(hand_results.multi_hand_landmarks)
        section_width = self.grid_width // max(num_hands, 1)
        
        for idx, (hand_landmarks, handedness) in enumerate(
            zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
            
            hand_label = handedness.classification[0].label
            x_offset = idx * section_width
            
            points = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * section_width) + x_offset
                y = int(lm.y * (self.grid_height - 80)) + 60
                points.append((x, y))
            
            connections = self.mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                cv2.line(grid, points[start_idx], points[end_idx],
                        (0, 255, 255), 2)
            
            finger_tips = [4, 8, 12, 16, 20]
            for lm_idx, (x, y) in enumerate(points):
                if lm_idx in finger_tips:
                    color = (0, 0, 255)
                    radius = 6
                elif lm_idx == 0:
                    color = (255, 0, 255)
                    radius = 8
                else:
                    color = (255, 255, 0)
                    radius = 4
                cv2.circle(grid, (x, y), radius, color, -1)
            
            label_x = x_offset + section_width // 2 - 50
            cv2.putText(grid, f"{hand_label} Hand", (label_x, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return grid
    
    def conversation_thread(self):
        """Handle voice conversation in separate thread"""
        while not self.terminate_event.is_set():
            try:
                with sr.Microphone(device_index=self.microphone_index) as source:
                    print("Listening...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5)
                
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                
                if "hello" in text.lower() and not self.conversation_started:
                    self.conversation_started = True
                    self.send_command_to_arduino(8)
                    self.text_to_speech("Hello! How can I assist you?")
                    self.send_command_to_arduino(9)
                    continue
                
                if "bye" in text.lower():
                    print("Conversation ended")
                    self.send_command_to_arduino(8)
                    self.text_to_speech("Goodbye! See You Soon!")
                    self.send_command_to_arduino(9)
                    self.conversation_started = False
                    continue
                
                if "shut down" in text.lower():
                    self.send_command_to_arduino(8)
                    self.text_to_speech("Abort sequence initiated. Shutting down!")
                    self.send_command_to_arduino(9)
                    self.terminate_event.set()
                    break
                
                if self.conversation_started:
                    if RAG_AVAILABLE:
                        response = response_rag(question=text, user_expression=self.glob_expression)
                    else:
                        response = f"I heard you say: {text}. My response system is currently limited."
                    
                    self.send_command_to_arduino(8)
                    self.text_to_speech(response)
                    self.send_command_to_arduino(9)
                else:
                    self.send_command_to_arduino(8)
                    self.text_to_speech("Greet me with HELLO VULCAN to start chatting!")
                    self.send_command_to_arduino(9)
                    
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                if self.conversation_started:
                    self.send_command_to_arduino(8)
                    self.text_to_speech("Sorry, I could not understand.")
                    self.send_command_to_arduino(9)
            except Exception as e:
                print(f"Conversation error: {e}")
                time.sleep(1)
    
    def vision_thread(self):
        """Handle vision processing in main thread"""
        last_expression = ""
        last_update_time = time.time()
        
        while not self.terminate_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Rotate frame if needed
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Process frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pose_results = None
            hand_results = None
            
            if self.enable_pose:
                pose_results = self.pose.process(frame_rgb)
            
            if self.enable_hands:
                hand_results = self.hands.process(frame_rgb)
            
            # Face detection and tracking
            if CUSTOM_FACE_MODULES:
                bbox, gray = face_de.get_face_harr(frame=frame)
            else:
                faces = self.detect_faces(frame)
                bbox = [(x, y, x+w, y+h) for (x, y, w, h, _) in faces]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            current_time = time.time()
            
            # Periodic Arduino commands
            if current_time - self.last_command_time >= 50:
                self.send_command_to_arduino(6)
                self.last_command_time = current_time
            
            if current_time - self.last_command_time1 >= 30:
                self.send_command_to_arduino(5)
                self.last_command_time1 = current_time
            
            # Process detected faces
            if bbox and self.conversation_started:
                b = bbox[0] if isinstance(bbox[0], (list, tuple)) else bbox[0]
                
                # Calculate face center
                if len(b) == 4:
                    face_center_x = int((b[0] + b[2]) / 2)
                    face_center_y = int((b[1] + b[3]) / 2)
                    
                    # Eye movement commands
                    if abs(face_center_x - self.frame_center_x) > self.tolerance_x:
                        if face_center_x < self.frame_center_x:
                            self.send_command_to_arduino(2)
                        else:
                            self.send_command_to_arduino(1)
                    
                    if abs(face_center_y - self.frame_center_y) > self.tolerance_y:
                        if face_center_y < self.frame_center_y:
                            self.send_command_to_arduino(3)
                        else:
                            self.send_command_to_arduino(4)
                    
                    # Expression detection
                    face_img = gray[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                    if face_img.size > 0:
                        expression_result = self.get_expression(face_img)
                        
                        if expression_result != last_expression or (current_time - last_update_time) >= 1:
                            last_expression = expression_result
                            self.glob_expression = last_expression
                            last_update_time = current_time
                        
                        # Face identification
                        embedding = self.get_embedding(cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR))
                        person_id, is_new, confidence = self.match_or_create_id(embedding)
                        self.update_database(person_id, embedding, is_new)
                        
                        person_data = self.face_db[person_id]
                        name = person_data.get('name')
                        
                        # Draw face info
                        color = (255, 100, 0) if person_data.get('is_erc_member') else (0, 255, 0)
                        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
                        
                        label = name if name else person_id
                        label += f" | {self.glob_expression}"
                        cv2.putText(frame, label, (int(b[0]), int(b[1]) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display statistics
            cv2.putText(frame, f"Faces: {len(self.face_db)} | Conv: {'ON' if self.conversation_started else 'OFF'}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Create grids
            pose_grid = self.create_pose_grid(pose_results) if self.show_grids else None
            hand_grid = self.create_hand_grid(hand_results) if self.show_grids else None
            
            # Combine display
            if self.show_grids and pose_grid is not None and hand_grid is not None:
                side_panel = np.vstack((pose_grid, hand_grid))
                main_h, main_w = frame.shape[:2]
                side_h, side_w = side_panel.shape[:2]
                
                combined_h = max(main_h, side_h)
                scale_main = combined_h / main_h
                new_main_w = int(main_w * scale_main)
                resized_main = cv2.resize(frame, (new_main_w, combined_h))
                
                scale_side = combined_h / side_h
                new_side_w = int(side_w * scale_side)
                resized_side = cv2.resize(side_panel, (new_side_w, combined_h))
                
                combined_frame = np.hstack((resized_main, resized_side))
                cv2.imshow('Vulcan Unified System', combined_frame)
            else:
                cv2.imshow('Vulcan Unified System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.terminate_event.set()
                break
            elif key == ord('g'):
                self.show_grids = not self.show_grids
                print(f"Grids: {'ON' if self.show_grids else 'OFF'}")
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def save_database(self):
        """Save face database"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump({
                    'face_db': self.face_db,
                    'next_id': self.next_id
                }, f)
            print(f"✓ Database saved")
        except Exception as e:
            print(f"⚠️ Error saving database: {e}")
    
    def load_database(self):
        """Load face database"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.face_db = data['face_db']
                    self.next_id = data['next_id']
                print(f"✓ Database loaded")
            except Exception as e:
                print(f"⚠️ Error loading database: {e}")
    
    def run(self):
        """Run the unified system"""
        print("\n" + "="*70)
        print("VULCAN UNIFIED AI SYSTEM - STARTING")
        print("="*70)
        print("Controls:")
        print("  q - Quit")
        print("  g - Toggle visualization grids")
        print("Voice commands:")
        print("  'Hello' - Start conversation")
        print("  'Bye' - End conversation")
        print("  'Shut down' - Terminate system")
        print("="*70 + "\n")
        
        # Start conversation thread
        conv_thread = threading.Thread(target=self.conversation_thread, daemon=True)
        conv_thread.start()
        
        # Run vision in main thread
        try:
            self.vision_thread()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🔄 Cleaning up...")
        self.terminate_event.set()
        
        if self.cap:
            self.cap.release()
        
        if self.arduino and self.enable_arduino:
            try:
                self.arduino.close()
                print("✓ Arduino connection closed")
            except:
                pass
        
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except:
                pass
        
        cv2.destroyAllWindows()
        self.save_database()
        print("✓ System shutdown complete")


def main():
    """Main entry point"""
    print("="*70)
    print("VULCAN UNIFIED AI SYSTEM")
    print("Integrating: Face ID + Expression + Pose + Hands + Voice + Arduino")
    print("="*70)
    
    # Configuration
    print("\n📋 Configuration:")
    
    # Video source
    use_ip_camera = input("Use IP camera? (y/n, default=n): ").strip().lower() == 'y'
    if use_ip_camera:
        video_source = input("Enter IP camera URL (default=http://192.168.154.230:81/stream): ").strip()
        if not video_source:
            video_source = "http://192.168.154.230:81/stream"
    else:
        video_source = int(input("Enter camera index (default=0): ").strip() or "0")
    
    # Arduino
    enable_arduino = input("Enable Arduino? (y/n, default=y): ").strip().lower() != 'n'
    arduino_port = 'COM8'
    if enable_arduino:
        custom_port = input("Arduino port (default=COM8): ").strip()
        if custom_port:
            arduino_port = custom_port
    
    # Microphone
    microphone_index = input("Microphone device index (default=18): ").strip()
    microphone_index = int(microphone_index) if microphone_index else 18
    
    # Features
    enable_pose = True
    enable_hands = True
    
    if MEDIAPIPE_AVAILABLE:
        features = input("\nEnable features:\n1. All (Pose + Hands)\n2. Custom\nChoice (default=1): ").strip()
        if features == '2':
            enable_pose = input("Enable pose estimation? (y/n): ").lower() != 'n'
            enable_hands = input("Enable hand tracking? (y/n): ").lower() != 'n'
    
    print("\n🚀 Initializing system...")
    
    try:
        system = VulcanUnifiedSystem(
            video_source=video_source,
            arduino_port=arduino_port,
            arduino_baudrate=115200,
            db_path='face_database.pkl',
            similarity_threshold=0.55,
            microphone_index=microphone_index,
            enable_pose=enable_pose,
            enable_hands=enable_hands,
            enable_arduino=enable_arduino
        )
        
        # Pre-run options
        while True:
            print("\n📝 PRE-RUN OPTIONS:")
            print("1. Start system")
            print("2. List all persons in database")
            print("3. Assign name to person ID")
            print("4. Export database to JSON")
            
            choice = input("Choice (default=1): ").strip() or "1"
            
            if choice == '1':
                break
            elif choice == '2':
                list_all_persons(system.face_db)
            elif choice == '3':
                person_id = input("Enter Person ID (e.g., P001): ").strip().upper()
                name = input("Enter Name: ").strip()
                if person_id and name:
                    if person_id in system.face_db:
                        system.face_db[person_id]['name'] = name
                        system.face_db[person_id]['is_erc_member'] = True
                        system.save_database()
                        print(f"✓ Assigned '{name}' to {person_id}")
                    else:
                        print(f"✗ Person ID {person_id} not found")
            elif choice == '4':
                export_database(system.face_db)
        
        # Run the system
        system.run()
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def list_all_persons(face_db):
    """List all persons in database"""
    print("\n" + "="*70)
    print("DATABASE CONTENTS")
    print("="*70)
    
    if not face_db:
        print("Database is empty")
        return
    
    named = []
    unnamed = []
    
    for person_id, data in sorted(face_db.items()):
        entry = {
            'id': person_id,
            'name': data.get('name', 'UNNAMED'),
            'appearances': data['appearances'],
            'first_seen': data['first_seen'][:19] if 'first_seen' in data else 'N/A',
            'last_seen': data['last_seen'][:19] if 'last_seen' in data else 'N/A',
            'erc_member': data.get('is_erc_member', False)
        }
        
        if data.get('name'):
            named.append(entry)
        else:
            unnamed.append(entry)
    
    if named:
        print(f"\n👥 ERC MEMBERS ({len(named)}):")
        print("-"*70)
        for e in named:
            print(f"{e['id']}: {e['name']} | Appearances: {e['appearances']} | "
                  f"Last seen: {e['last_seen']}")
    
    if unnamed:
        print(f"\n👤 STRANGERS ({len(unnamed)}):")
        print("-"*70)
        for e in unnamed:
            print(f"{e['id']}: [UNNAMED] | Appearances: {e['appearances']} | "
                  f"Last seen: {e['last_seen']}")
    
    print("="*70 + "\n")


def export_database(face_db, json_path='face_database.json'):
    """Export database to JSON"""
    export_data = {}
    for person_id, data in face_db.items():
        export_data[person_id] = {
            'name': data.get('name'),
            'appearances': data['appearances'],
            'first_seen': data.get('first_seen', 'N/A'),
            'last_seen': data.get('last_seen', 'N/A'),
            'is_erc_member': data.get('is_erc_member', False)
        }
    
    try:
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✓ Database exported to {json_path}")
    except Exception as e:
        print(f"✗ Export failed: {e}")


if __name__ == "__main__":
    main()