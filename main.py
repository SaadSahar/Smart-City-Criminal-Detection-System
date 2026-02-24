# -*- coding: utf-8 -*-
"""
Smart City Criminal Detection System
نظام مراقبة ذكي للمدينة للتعرف على المطلوبين

Author: Saad Sahar
"""
import os
import glob
import cv2
import numpy as np
import face_recognition
import sqlite3
from datetime import datetime, timedelta
import urllib.request
import pandas as pd
import threading

# استيراد الإعدادات
from config import (
    SOURCE_IS_IP, IP_URL, WEBCAM_INDEX,
    WANTED_FACES_DIR, UNKNOWN_FACES_DIR,
    DATABASE_PATH, EXCEL_EXPORT_PATH,
    TOLERANCE, FRAME_RESIZING, COOLDOWN_SECONDS,
    SOUND_FILE, ENABLE_SOUND, SAVE_UNKNOWN
)

# ===================== تهيئة الصوت =====================
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[تحذير] pygame غير متوفر - لن يعمل الصوت")

def play_sound_bg(path: str):
    """تشغيل الصوت في الخلفية"""
    if not ENABLE_SOUND or not PYGAME_AVAILABLE:
        return
    if not os.path.isfile(path):
        print(f"[صوت] الملف غير موجود: {path}")
        return
    
    def _run():
        try:
            snd = pygame.mixer.Sound(path)
            snd.play()
            while pygame.mixer.get_busy():
                pygame.time.wait(50)
        except Exception as e:
            print(f"[صوت] خطأ: {e}")
    
    threading.Thread(target=_run, daemon=True).start()


# ===================== كلاس التعرف على الوجوه =====================
class FaceRecognizer:
    """كلاس للتعرف على الوجوه ومقارنتها مع قاعدة البيانات"""
    
    def __init__(self, frame_resizing=0.25, tolerance=0.45):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = frame_resizing
        self.tolerance = tolerance
    
    def _is_image_file(self, path):
        """التحقق من أن الملف صورة"""
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        return os.path.splitext(path)[1].lower() in exts
    
    def load_wanted_faces(self, base_dir):
        """
        تحميل صور المطلوبين من المجلدات
        
        الهيكل المتوقع:
        base_dir/
        ├── شخص1/
        │   ├── photo1.jpg
        │   └── photo2.jpg
        └── شخص2/
            └── photo1.jpg
        """
        if not os.path.isdir(base_dir):
            raise ValueError(f"المسار غير صحيح: {base_dir}")
        
        persons = [d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d))]
        
        if not persons:
            raise ValueError("لا توجد مجلدات أشخاص داخل المسار المحدد.")
        
        total_imgs = 0
        total_encs = 0
        
        for person in persons:
            person_dir = os.path.join(base_dir, person)
            img_paths = [p for p in glob.glob(os.path.join(person_dir, "*")) 
                        if self._is_image_file(p)]
            
            if not img_paths:
                print(f"تحذير: لا توجد صور في مجلد {person}")
                continue
            
            print(f"[{person}] {len(img_paths)} صورة.")
            total_imgs += len(img_paths)
            
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"تخطي ملف غير صالح: {img_path}")
                    continue
                
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb)
                
                if not encs:
                    print(f"لا يوجد وجه في: {os.path.basename(img_path)}")
                    continue
                
                self.known_face_encodings.append(encs[0])
                self.known_face_names.append(person)
                total_encs += 1
        
        if total_encs == 0:
            raise ValueError("لم يتم توليد أي ترميزات. تحقق من الصور.")
        
        unique_persons = len(set(self.known_face_names))
        print(f"✓ تم التحميل: {total_encs} ترميز من {total_imgs} صورة لـ {unique_persons} مطلوب.")
        
        return unique_persons
    
    def detect_known_faces(self, frame):
        """اكتشاف وتعريف الوجوه في الإطار"""
        small = cv2.resize(frame, (0, 0), 
                          fx=self.frame_resizing, 
                          fy=self.frame_resizing)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        
        names = []
        for enc in face_encodings:
            if not self.known_face_encodings:
                names.append("Unknown")
                continue
            
            dists = face_recognition.face_distance(self.known_face_encodings, enc)
            idx = int(np.argmin(dists))
            name = self.known_face_names[idx] if dists[idx] <= self.tolerance else "Unknown"
            names.append(name)
        
        face_locations = np.array(face_locations)
        if len(face_locations) > 0:
            face_locations = (face_locations / self.frame_resizing).astype(int)
        
        return face_locations, names


# ===================== قاعدة البيانات =====================
class RecognitionLogger:
    """كلاس لإدارة سجلات التعرف"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_table()
        self.last_logged_at = {}
    
    def _create_table(self):
        """إنشاء جدول السجلات"""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS recognition_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp DATETIME,
            camera_id INTEGER,
            confidence REAL
        )
        """)
        self.conn.commit()
    
    def should_log(self, name: str, now: datetime, cooldown: int) -> bool:
        """التحقق مما إذا كان يجب تسجيل هذا التعرف"""
        if name == "Unknown":
            return False
        prev = self.last_logged_at.get(name)
        return (prev is None) or ((now - prev) >= timedelta(seconds=cooldown))
    
    def log_recognition(self, name: str, camera_id: int, confidence: float = 0.0):
        """تسجيل عملية التعرف"""
        now = datetime.now()
        ts = now.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            self.cursor.execute(
                "INSERT INTO recognition_log (name, timestamp, camera_id, confidence) VALUES (?, ?, ?, ?)",
                (name, ts, camera_id, confidence)
            )
            self.conn.commit()
            self.last_logged_at[name] = now
            print(f"[LOG] {name} @ {ts} (Cam-{camera_id})")
            return True
        except Exception as e:
            print(f"[DB] خطأ: {e}")
            return False
    
    def export_to_excel(self, output_path: str):
        """تصدير السجلات إلى Excel"""
        try:
            df = pd.read_sql_query("SELECT * FROM recognition_log", self.conn)
            df.to_excel(output_path, index=False)
            print(f"✓ تم التصدير إلى: {output_path}")
            return True
        except Exception as e:
            print(f"خطأ في التصدير: {e}")
            return False
    
    def clear_log(self):
        """مسح السجلات"""
        try:
            self.cursor.execute("DELETE FROM recognition_log")
            self.conn.commit()
            self.last_logged_at.clear()
            print("✓ تم مسح السجل.")
        except Exception as e:
            print(f"خطأ في مسح السجل: {e}")
    
    def close(self):
        """إغلاق الاتصال"""
        self.conn.close()


# ===================== معالجة الإطارات =====================
def draw_and_handle(frame, face_locations, face_names, camera_id, 
                    logger, cooldown, save_unknown=True, unknown_dir=None):
    """رسم النتائج ومعالجة التعرفات"""
    now = datetime.now()
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # تحديد اللون
        if name == "Unknown":
            color = (0, 0, 200)  # أحمر للمجهو
