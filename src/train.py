from ultralytics import YOLO
import os

def train_model():
    """Trains the YOLOv8 model on the Skin Lesion dataset."""
    print("Initializing YOLOv8 model...")
    
    # Check if data.yaml exists
    if not os.path.exists("data.yaml"):
        print("❌ Error: data.yaml not found. Please run src/data_setup.py first.")
        return

    # Load the base nano model
    model = YOLO("yolov8n.pt")

    print("Starting training process...")
    
    # Train the model
    # We save the results into the standard `models/` directory instead of the root
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # Uses GPU 0. Change to 'cpu' if you don't have a GPU setup.
        workers=2,
        project="models/skin_lesion_yolo", 
        name="yolov8_detection"
    )
    
    print("\n✅ Training complete! The weights and metrics are saved in 'models/skin_lesion_yolo/yolov8_detection'.")

if __name__ == "__main__":
    train_model()
