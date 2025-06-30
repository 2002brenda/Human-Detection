import os
import shutil

# Use raw string to prevent unicode escape issues
original_dataset_path = r"D:\Cigarette-Detection-YOLOv8n-main\people 2.v1i.yolov8"
output_dataset_path = r"D:\Cigarette-Detection-YOLOv8n-main\people 2.v1i.yolov8"

splits = ["train", "test"]
for split in splits:
    img_dst = os.path.join(output_dataset_path, "images", "train" if split == "train" else "val")
    lbl_dst = os.path.join(output_dataset_path, "labels", "train" if split == "train" else "val")
    
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)
    
    split_path = os.path.join(original_dataset_path, split)
    
    for file in os.listdir(split_path):
        src_file = os.path.join(split_path, file)
        if file.endswith(".jpg"):
            shutil.copy(src_file, os.path.join(img_dst, file))
        elif file.endswith(".txt"):
            shutil.copy(src_file, os.path.join(lbl_dst, file))

print("✅ Dataset reorganized successfully.")
