import os
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    print("--- 1. بدء تحميل البيانات من Roboflow مباشرة إلى Azure ---")
    # هذا المفتاح والمسار تم استخراجهم من ملفات الكود الخاصة بك
    rf = Roboflow(api_key="bXgZ0bX66gXlZ8XgXgXg") # يرجى التأكد من كتابة المفتاح كاملاً كما في ملف الـ Notebook الخاص بك
    project = rf.workspace("accident-detection-vby8i").project("traffic-accidents-severity")
    
    # تحميل البيانات بصيغة متوافقة مع موديلات YOLO الحديثة
    dataset = project.version(1).download("yolov11")
    
    # تحديد مسار ملف data.yaml التلقائي الذي ينزل داخل السيرفر
    yaml_path = os.path.join(dataset.location, "data.yaml")
    print(f"تم تحميل البيانات بنجاح في المسار السحابي: {dataset.location}")

    print("--- 2. تهيئة موديل YOLO ---")
    # سنقوم بتحميل الموديل الأساسي (يمكنك كتابة yolo11x.pt أو yolo11s.pt حسب رغبتك)
    model = YOLO("yolo11s.pt") 

    print("--- 3. بدء تدريب الموديل على الـ GPU في Azure ---")
    # قمنا بضبط الإعدادات هنا لتعمل على كروت شاشة قوية ولتوفير التقدم تلقائياً
    results = model.train(
        data=yaml_path,
        epochs=100,             # عدد الدورات للتدريب
        imgsz=640,              # حجم الصور المتفق عليه
        batch=64,               # حجم الـ Batch (قم برفعه إلى 128 إذا كان الكرت قوي جداً)
        device=0,               # استخدام كرت الـ GPU المتاح في السيرفر
        workers=8,              # تسريع معالجة الصور من خلال النوايا المتعددة للمقاطع المعالجة
        project="azure_accident_detection",
        name="yolo_azure_run",
        save=True               # حفظ الأوزان تلقائياً عند كل Epoch لحمايتها من الضياع
    )
    print("--- مبروك! اكتمل التدريب بنجاح وتم حفظ الأوزان النهائية ---")

if __name__ == "__main__":
    main()