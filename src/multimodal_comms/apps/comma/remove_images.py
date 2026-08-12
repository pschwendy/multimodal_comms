import os
import sys

# Define common image extensions (case-insensitive)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.svg'}

def remove_images(directory):
    total_deleted = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    total_deleted += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    print(f"\nTotal image files deleted: {total_deleted}")

if __name__ == "__main__":
    start_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(start_dir):
        print(f"Error: '{start_dir}' is not a valid directory.")
        sys.exit(1)
    
    print(f"Searching for images in: {os.path.abspath(start_dir)}\n")
    remove_images(start_dir)
