import os
import shutil
import random
from PIL import Image, ImageDraw, ImageFont

def save_uploaded_files(uploaded_files, destination_dir, append=False):
    """
    Saves uploaded Streamlit files to the specified directory.
    If append=False, clears the directory first.
    If append=True, adds files to the existing directory.
    """
    # Clear existing files if not appending
    if not append and os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)
    
    os.makedirs(destination_dir, exist_ok=True)

    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(destination_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    
    return saved_paths

def draw_bounding_boxes(image_path, detections):
    """
    Draws bounding boxes on an image based on detection results.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # Load a default font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        for det in detections:
            # Dummy logic for drawing: in a real scenario, use actual bbox coordinates
            # bbox = det['bbox']
            # draw.rectangle(bbox, outline="red", width=3)
            
            # For this demo, we just print text since we generated random bboxes
            label = f"{det['class']} ({det['confidence']:.2f})"
            
            # Draw a random box for visual demo
            w, h = image.size
            x1 = random.randint(0, int(w*0.8))
            y1 = random.randint(0, int(h*0.8))
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(50, 150)
            
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=4)
            draw.text((x1, y1-10), label, fill="#00FF00", font=font)
            
        return image
    except Exception as e:
        print(f"Error drawing bounding boxes: {e}")
        return None
