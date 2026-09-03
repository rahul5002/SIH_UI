import os
import torch
import torchvision.models as models

def export_model_to_onnx(output_path="mobilenet_v3.onnx"):
    print("Loading PyTorch MobileNetV3 model...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[0].in_features, 512),
        torch.nn.LayerNorm(512)
    )
    model.eval()

    dummy_input = torch.randn(1, 3, 160, 160)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"ONNX export successful. Model saved at: {output_path}")

if __name__ == "__main__":
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "face_engine", "models")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "face_embedder.onnx")
    export_model_to_onnx(target_path)
