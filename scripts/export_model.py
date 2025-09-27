# scripts/export_model.py

import torch
import sys
from kaissa_net.model import KaissaNet
from kaissa_encoding.board_encoder import BOARD_ROWS, BOARD_COLS, encode_board
import numpy as np

def export_to_onnx(model_path, out_path="model.onnx", device="cpu"):
    model = KaissaNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    dummy_input = torch.zeros((1, 3, BOARD_ROWS, BOARD_COLS), dtype=torch.float32).to(device)

    torch.onnx.export(
        model, dummy_input,
        out_path,
        export_params=True,
        input_names=['input'],
        output_names=['policy', 'value'],
        opset_version=12
    )
    print(f"Exported ONNX to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_model.py <model_path> [onnx_path]")
        sys.exit(1)
    mp = sys.argv[1]
    outp = "model.onnx"
    if len(sys.argv) > 2:
        outp = sys.argv[2]
    export_to_onnx(mp, outp)
