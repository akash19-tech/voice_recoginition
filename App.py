import streamlit as st
import sqlite3
import numpy as np
import librosa
import os
import torch
from datetime import datetime
from streamlit_audio_recorder import audio_recorder
import tensorflow as tf
from tensorflow.keras import layers, models


os.environ["WHISPER_DISABLE_NUMBA"] = "1"
import whisper


conn = sqlite3.connect('attendance.db')
c = conn.cursor()


c.execute('''CREATE TABLE IF NOT EXISTS students
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT UNIQUE,
              voice_embedding BLOB)''')

c.execute('''CREATE TABLE IF NOT EXISTS attendance
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              date TEXT,
              student_id INTEGER,
              FOREIGN KEY(student_id) REFERENCES students(id))''')
conn.commit()

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model_whisper = load_whisper_model()

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape((input_shape[0], input_shape[1], 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfcc.T

st.title("Voice Recognition Attendance System")

page = st.radio("Navigation", ["Register", "Mark Attendance", "View Attendance"])

if page == "Register":
    st.header("Student Registration")
    name = st.text_input("Enter your name")
    
    audio_data = audio_recorder("Record your voice:", pause_threshold=2.0)
    
    if audio_data and name:
        audio_path = f"audio/{name}.wav"
        os.makedirs("audio", exist_ok=True)
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        
        features = extract_features(audio_path)
        
        c.execute("INSERT INTO students (name, voice_embedding) VALUES (?, ?)",
                 (name, features.tobytes()))
        conn.commit()
        st.success("Registration Successful!")

elif page == "Mark Attendance":
    st.header("Mark Attendance")
    
    audio_data = audio_recorder("Record attendance voice:", pause_threshold=2.0)
    
    if audio_data:
        temp_path = "temp.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_data)
        
        result = model_whisper.transcribe(temp_path)
        spoken_text = result['text'].strip().lower()
        st.write("Detected Speech:", spoken_text)
        
        c.execute("SELECT * FROM students")
        students = c.fetchall()
        
        matched_student = None
        for student in students:
            if student[1].lower() in spoken_text:
                matched_student = student
                break
        
        if matched_student:
            features = extract_features(temp_path)
            student_features = np.frombuffer(matched_student[2], dtype=np.float32).reshape(-1, 40)
            
            X = []
            y = []
            for s in students:
                features = np.frombuffer(s[2], dtype=np.float32).reshape(-1, 40)
                X.append(features)
                y.append(s[0])
            
            model = create_cnn_model((X[0].shape[0], X[0].shape[1]), len(students))
            model.fit(np.array(X), np.array(y), epochs=10)
            
            prediction = model.predict(np.array([features]))
            predicted_id = np.argmax(prediction)
            
            if predicted_id == matched_student[0]:
                today = datetime.now().strftime("%Y-%m-%d")
                c.execute("INSERT INTO attendance (date, student_id) VALUES (?, ?)",
                         (today, matched_student[0]))
                conn.commit()
                st.success(f"Attendance marked for {matched_student[1]}")
            else:
                st.error("Voice verification failed!")
        else:
            st.error("No matching student found!")

elif page == "View Attendance":
    st.header("Attendance Records")
    c.execute('''SELECT students.name, attendance.date 
                 FROM attendance 
                 INNER JOIN students ON attendance.student_id = students.id''')
    records = c.fetchall()
    
    if records:
        for name, date in records:
            st.write(f"{name} - {date}")
    else:
        st.write("No attendance records found")

conn.close()