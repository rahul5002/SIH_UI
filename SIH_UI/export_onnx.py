"""
Script to export the PyTorch MobileNetV3 face embedding model to ONNX format.
This enables cost-effective, high-speed CPU inference using ONNX Runtime.
"""
import os
import torch
import torchvision.models as models

def export_model_to_onnx(output_path="mobilenet_v3.onnx"):
    print("Loading PyTorch MobileNetV3 model...")
    # 1. Initialize the same model architecture used in face_matcher.py
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[0].in_features, 512),
        torch.nn.LayerNorm(512)
    )
    model.eval()

    # 2. Create a dummy input tensor matching the expected input shape
    # Shape: (batch_size, channels, height, width) = (1, 3, 160, 160)
    dummy_input = torch.randn(1, 3, 160, 160)

    print(f"Exporting model to {output_path}...")
    
    # 3. Export to ONNX
    torch.onnx.export(
        model,                      # model being run
        dummy_input,                # model input (or a tuple for multiple inputs)
        output_path,                # where to save the model
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=14,           # the ONNX version to export the model to
        do_constant_folding=True,   # whether to execute constant folding for optimization
        input_names=['input'],      # the model's input names
        output_names=['output'],    # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX export successful. Model saved at: {output_path}")

if __name__ == "__main__":
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "face_engine", "models")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "face_embedder.onnx")
    export_model_to_onnx(target_path)
