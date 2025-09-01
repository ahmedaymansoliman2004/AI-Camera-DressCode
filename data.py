import os
from tkinter import Tk, filedialog
from PIL import Image

# فتح نافذة لاختيار الصور
Tk().withdraw()  # إخفاء نافذة tkinter الرئيسية
file_paths = filedialog.askopenfilenames(
    title="اختر الصور",
    filetypes=[("Images", "*.jpg;*.jpeg;*.png")]
)

# معالجة الصور المختارة
for img_path in file_paths:
    filename = os.path.basename(img_path)
    folder = os.path.dirname(img_path)

    # افتح الصورة
    img = Image.open(img_path)
    width, height = img.size

    # لو الأبعاد مش 640x640 غيرها
    if (width, height) != (640, 640):
        img = img.resize((640, 640))
        img.save(img_path)
        print(f"✅ Resized: {filename}")
    else:
        print(f"✔ Already 640x640: {filename}")

    # اعمل ملف txt بنفس الاسم
    txt_name = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(folder, txt_name)

    open(txt_path, 'w').close()
    print(f"📝 Created: {txt_name}")
