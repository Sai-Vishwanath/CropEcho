import torch
import os
from PIL import Image  # <--- Add this line!

def predict_wheat_quality(image_path, model_path='models/grain_yolov5_best.pt'):
    # Load the custom YOLOv5 model
    # Note: This requires the yolov5 repo to be cached or accessible.
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    except Exception as e:
        return f"Error loading YOLOv5 model: {e}"

    # Run the prediction
    # ... inside predict_wheat_quality function ...
    results = model(image_path)
    
    # Force YOLO to save the annotated image to a specific filename
    # We'll save it as 'annotated_result.jpg' in the uploads folder
    annotated_path = os.path.join('static', 'uploads', 'annotated_result.jpg')
    results.render()  # This draws the boxes onto the image pixels in memory
    rendered_img = Image.fromarray(results.ims[0]) # Get the image with boxes
    rendered_img.save(annotated_path) # Save it physically
    
    predictions_data = results.pandas().xyxy[0].to_dict(orient="records")
    return predictions_data

# --- Test the script locally ---
if __name__ == "__main__":
    import glob
    
    # Search for any file starting with 'test_grain' in the uploads folder
    possible_images = glob.glob("static/uploads/test_grain.*")
    
    if possible_images:
        test_image = possible_images[0]
        print(f"Found test image: {test_image}")
        
        results = predict_wheat_quality(test_image)
        
        # Check if the model returned an error string instead of data
        if isinstance(results, str):
            print(f"\n❌ MODEL ERROR:\n{results}")
        elif len(results) == 0:
            print("\n⚠️ No wheat grains detected in the image.")
        else:
            print("\n✅ Detections found:")
            for det in results:
                print(f"- {det['name']} (Confidence: {det['confidence']:.2f})")
    else:
         print("Please place a test image named 'test_grain.jpg' or 'test_grain.png' in static/uploads/")