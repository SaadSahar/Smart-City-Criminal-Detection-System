# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np
import face_recognition
import sqlite3
from datetime import datetime, timedelta
import urllib.request
import pandas as pd
from playsound import playsound
import threading
import time

# ===================== الإعدادات =====================
BASE_DIR = r"D:\\data_base_img"

SOURCE_IS_IP = True  # True: كاميرا IP, False: كاميرا ويب
IP_URL = "http://192.168.136.32:8080/shot.jpg"
WEBCAM_INDEX = 0

SOUND_FILE = r"D:\\مشاريع تخرج\\تطبيقية\\تسجيل حضور\\audio.mp3"

EXCEL_EXPORT_PATH = r"D:\\مشاريع تخرج\\تطبيقية\\تسجيل حضور\\f.xlsx"

SAVE_UNKNOWN = True
UNKNOWN_DIR = r"D:\\مشاريع تخرج\\تطبيقية\\تسجيل حضور"
# =====================================================

if SAVE_UNKNOWN:
    os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ========= صوت باستخدام playsound =========
class SoundPlayer:
    def __init__(self):
        self.is_playing = False
        self.stop_requested = False
        self.thread = None
    
    def play_sound_loop(self, path: str):
        """تشغيل الصوت في حلقة مستمرة فقط للشخص المعروف"""
        if not os.path.isfile(path):
            print(f"[صوت] الملف غير موجود: {path}")
            return
        
        if not self.is_playing:
            self.stop_requested = False
            self.thread = threading.Thread(target=self._play_loop, args=(path,), daemon=True)
            self.thread.start()
            print("[صوت] تشغيل مستمر...")
    
    def _play_loop(self, path):
        """الدالة التي تعمل في الخلفية لتشغيل الصوت بشكل متكرر"""
        self.is_playing = True
        try:
            while not self.stop_requested:
                try:
                    playsound(path, block=True)
                except Exception as e:
                    print(f"[صوت] خطأ أثناء التشغيل: {e}")
                    break
                # تأخير بسيط بين التشغيلات لمنع التحميل الزائد
                time.sleep(0.1)
        finally:
            self.is_playing = False
    
    def stop_sound(self):
        """إيقاف الصوت"""
        if self.is_playing:
            self.stop_requested = True
            print("[صوت] توقيف")

# إنشاء كائن تشغيل الصوت
sound_player = SoundPlayer()

# ========= كلاس التعرف =========
class SimpleFacerec:
    def __init__(self, frame_resizing=0.25, tolerance=0.45):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = frame_resizing
        self.tolerance = tolerance

    def _is_image_file(self, path):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        return os.path.splitext(path)[1].lower() in exts

    def load_encoding_images(self, base_dir):
        if not os.path.isdir(base_dir):
            raise ValueError(f"المسار غير صحيح: {base_dir}")

        persons = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not persons:
            raise ValueError("لا توجد مجلدات أشخاص داخل المسار المحدد.")

        total_imgs = 0
        total_encs = 0
        for person in persons:
            person_dir = os.path.join(base_dir, person)
            img_paths = [p for p in glob.glob(os.path.join(person_dir, "*")) if self._is_image_file(p)]
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

        print(f"تم التحميل: {total_encs} ترميز من {total_imgs} صورة لعدد {len(set(self.known_face_names))} أشخاص.")

    def detect_known_faces(self, frame):
        small = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
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

# ========= قاعدة البيانات =========
sfr = SimpleFacerec(frame_resizing=0.25, tolerance=0.45)
sfr.load_encoding_images(BASE_DIR)

conn = sqlite3.connect('face_recognition_log.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS recognition_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    timestamp DATETIME,
    camera_id INTEGER
)
""")
conn.commit()

def draw_and_handle(frame, face_locations, face_names, camera_id=1):
    now = datetime.now()
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 250, 0) if name != "Unknown" else (0, 0, 200)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        cv2.putText(frame, f"{name} (ID:{camera_id})", (left, max(25, top - 10)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

        # حفظ المجهول (اختياري)
        if SAVE_UNKNOWN and name == "Unknown":
            ts = now.strftime("%Y%m%d_%H%M%S_%f")
            crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
            if crop.size > 0:
                cv2.imwrite(os.path.join(UNKNOWN_DIR, f"unknown_{camera_id}_{ts}.jpg"), crop)

        # تسجيل الشخص المعروف فقط مرة واحدة
        if name != "Unknown":
            try:
                cursor.execute("INSERT OR IGNORE INTO recognition_log (name, timestamp, camera_id) VALUES (?, ?, ?)",
                               (name, now.strftime('%Y-%m-%d %H:%M:%S'), camera_id))
                conn.commit()
                print(f"[LOG] {name} @ {now} (cam {camera_id})")
            except Exception as e:
                print(f"[DB] خطأ أثناء التسجيل: {e}")

    return frame

# ========= فتح المصدر =========
if SOURCE_IS_IP:
    cap = None
else:
    cap = cv2.VideoCapture(WEBCAM_INDEX)

# ========= الحلقة الرئيسية =========
while True:
    if SOURCE_IS_IP:
        try:
            im_arr = np.array(bytearray(urllib.request.urlopen(IP_URL, timeout=3).read()), dtype=np.uint8)
            frame = cv2.imdecode(im_arr, -1)
            if frame is None:
                raise ValueError("تعذّر فك ترميز الإطار من كاميرا IP.")
        except Exception as e:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"IP cam error: {e}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        ret, frame = cap.read()
        if not ret:
            print("خطأ في قراءة كاميرا الويب.")
            break

    # التعرف
    face_locations, face_names = sfr.detect_known_faces(frame)
    frame = draw_and_handle(frame, face_locations, face_names, camera_id=1)

    # الصوت: يشتغل فقط إذا فيه شخص معروف
    if any(name != "Unknown" for name in face_names):
        sound_player.play_sound_loop(SOUND_FILE)
    else:
        sound_player.stop_sound()

    # عرض
    frame=cv2.resize(frame,(640,480))
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        try:
            df = pd.read_sql_query("SELECT * FROM recognition_log", conn)
            df.to_excel(EXCEL_EXPORT_PATH, index=False)
            print(f"تم تصدير السجل إلى: {EXCEL_EXPORT_PATH}")
        except Exception as e:
            print(f"تعذّر التصدير إلى Excel: {e}")
        break

    elif key == ord('w'):
        try:
            cursor.execute("DELETE FROM recognition_log")
            conn.commit()
            print("تم مسح السجل.")
        except Exception as e:
            print(f"تعذّر مسح السجل: {e}")

# تنظيف
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
conn.close()