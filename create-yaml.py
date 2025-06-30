import os

output_dataset_path = r"D:\Cigarette-Detection-YOLOv8n-main\people 2.v1i.yolov8"

data_yaml = """\
path: {output_path}
train: images/train
val: images/val

nc: 1
names: ['cigarette']
""".format(output_path=output_dataset_path)

with open(os.path.join(output_dataset_path, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("✅ data.yaml created.")
