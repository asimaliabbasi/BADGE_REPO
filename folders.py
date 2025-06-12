import os

# List of class names
class_names = [
    "Yes", "No", "Thankyou", "Surrender",
    "Think", "Defence", "Stop", "Salute"
]

# Root directory
root_dir = "data"

# Create the root folder if it doesn't exist
os.makedirs(root_dir, exist_ok=True)

# Create subfolders for each class
for class_name in class_names:
    class_path = os.path.join(root_dir, class_name)
    os.makedirs(class_path, exist_ok=True)

print("Folders created successfully.")
