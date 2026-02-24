# -*- coding: utf-8 -*-
"""
إعدادات نظام مراقبة المدينة
Configuration for Smart City Criminal Detection System
"""
import os
from pathlib import Path

# المسار الأساسي للمشروع
BASE_DIR = Path(__file__).parent.absolute()

# ===================== إعدادات الكاميرا =====================
# مصدر الكاميرا: True = كاميرا IP, False = كاميرا ويب
SOURCE_IS_IP = False

# رابط كاميرا IP (إذا كان SOURCE_IS_IP = True)
IP_URL = "http://192.168.1.100:8080/shot.jpg"

# فهرس كاميرا الويب (إذا كان SOURCE_IS_IP = False)
WEBCAM_INDEX = 0

# ===================== إعدادات قاعدة البيانات =====================
# مجلد صور المطلوبين (مجلد لكل شخص بداخله عدة صور)
WANTED_FACES_DIR = str(BASE_DIR / "data" / "wanted_faces")

# مجلد الوجوه المجهولة
UNKNOWN_FACES_DIR = str(BASE_DIR / "data" / "unknown_faces")

# قاعدة بيانات السجلات
DATABASE_PATH = str(BASE_DIR / "outputs" / "recognition_log.db")

# ملف تصدير Excel
EXCEL_EXPORT_PATH = str(BASE_DIR / "outputs" / "recognition_export.xlsx")

# ===================== إعدادات التعرف =====================
# عتبة التعرف (0.4-0.6، الأقل = أكثر دقة)
TOLERANCE = 0.45

# حجم تصغير الإطار للسرعة
FRAME_RESIZING = 0.25

# مهلة التهدئة بالثواني (لتفادي التكرار)
COOLDOWN_SECONDS = 30

# ===================== إعدادات الصوت =====================
# ملف الصوت للتنبيه
SOUND_FILE = str(BASE_DIR / "assets" / "alert.mp3")

# تفعيل الصوت
ENABLE_SOUND = True

# ===================== إعدادات الحفظ =====================
# حفظ الوجوه المجهولة
SAVE_UNKNOWN = True

# إنشاء المجلدات تلقائياً
for dir_path in [UNKNOWN_FACES_DIR, str(BASE_DIR / "outputs")]:
    os.makedirs(dir_path, exist_ok=True)
