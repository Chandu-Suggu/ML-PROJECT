"""
Inference Script for Testing on Custom / User Images.
Takes any raw user photo or driving scene, performs semantic segmentation
(color-based or deep learning), extracts a user-specified edit region
(e.g., sky, road, car, vegetation), and applies region-aware, structure-preserving
translation while strictly preserving all other regions and boundaries.
"""

import argparse
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch

from data.cityscapes_classes import (
    CITYSCAPES_CLASSES,
    NAME_TO_CLASS,
    ID_TO_COLOR,
    color_to_train_id_map,
    train_id_to_color_map,
    create_mask_from_class_names,
    create_mask_from_color,
)
from models.generator import RegionAwareGenerator
from segmentation.color_segmenter import ColorSegmenter
from segmentation.deep_segmenter import DeepSemanticSegmenter
from utils.visualization import save_comparison_grid
from utils.metrics import compute_outside_mask_l1, compute_boundary_fscore, compute_psnr


def load_image(image_path: str, size: Tuple[int, int] = (256, 512)) -> Image.Image:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    img = Image.open(image_path).convert("RGB")
    # Resize to (width, height) = (size[1], size[0])
    img = img.resize((size[1], size[0]), Image.BILINEAR)
    return img


def run_inference_on_image(
    image_path: str,
    checkpoint_path: str = "checkpoints/best_model.pth",
    edit_class: str = "sky",
    edit_color: Optional[str] = None,
    custom_mask_path: Optional[str] = None,
    custom_semantic_path: Optional[str] = None,
    segmentation_method: str = "deep",
    style: str = "sunset",
    output_dir: str = "outputs/user_test",
    image_size: Tuple[int, int] = (256, 512),
    device: Optional[str] = None,
):
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    print(f"\n=======================================================")
    print(f"   Testing on Own Image: {image_path}")
    print(f"=======================================================")
    print(f"Device: {device_obj} | Target Class: '{edit_class}' | Style: '{style}'")

    os.makedirs(output_dir, exist_ok=True)
    pil_img = load_image(image_path, size=image_size)
    np_img = np.array(pil_img)

    # 1. Obtain Semantic Segmentation Map
    if custom_semantic_path and os.path.isfile(custom_semantic_path):
        print(f"Loading custom semantic segmentation map: {custom_semantic_path}")
        sem_pil = load_image(custom_semantic_path, size=image_size)
        sem_color_map = np.array(sem_pil)
        id_map = color_to_train_id_map(sem_color_map)
    else:
        print(f"Segmenting image using method: '{segmentation_method}'...")
        if segmentation_method.lower() == "color":
            segmenter = ColorSegmenter()
            id_map, sem_color_map = segmenter.segment_by_nearest_color(np_img)
        else:
            deep_segmenter = DeepSemanticSegmenter(model_name="lraspp", device=str(device_obj))
            id_map, sem_color_map = deep_segmenter.segment_image(pil_img)

    # 2. Extract Edit Mask
    if custom_mask_path and os.path.isfile(custom_mask_path):
        print(f"Loading custom edit mask: {custom_mask_path}")
        mask_pil = Image.open(custom_mask_path).convert("L").resize((image_size[1], image_size[0]), Image.NEAREST)
        edit_mask = (np.array(mask_pil) > 128).astype(np.float32)
    elif edit_color:
        rgb_vals = tuple(map(int, edit_color.split(",")))
        print(f"Extracting edit mask matching color RGB {rgb_vals}...")
        edit_mask = create_mask_from_color(sem_color_map, rgb_vals, tolerance=40)
    else:
        print(f"Extracting edit mask for semantic class: '{edit_class}'...")
        edit_mask = create_mask_from_class_names(id_map, [edit_class])

    # Check mask coverage
    coverage_pct = (edit_mask.sum() / edit_mask.size) * 100.0
    print(f"Edit Mask generated: {coverage_pct:.2f}% of pixels selected for translation.")
    if coverage_pct < 0.1:
        print(f"[!] Warning: Class '{edit_class}' covers almost 0% of the image. Defaulting upper 35% as edit region.")
        edit_mask[: int(image_size[0] * 0.35), :] = 1.0

    # 3. Load Trained Region-Aware Generator
    # Auto-detect ngf from checkpoint if present
    ngf = 32
    state = None
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device_obj)
        if "net_G" in state:
            state = state["net_G"]
        if "enc1.block.0.weight" in state:
            ngf = state["enc1.block.0.weight"].shape[0]
            print(f"Auto-detected generator ngf={ngf} from checkpoint.")

    net_G = RegionAwareGenerator(in_channels=6, out_channels=3, ngf=ngf).to(device_obj)
    if state is not None:
        net_G.load_state_dict(state)
        print(f"Loaded trained generator from: {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint '{checkpoint_path}' not found! Using initialized weights.")

    net_G.eval()

    # 4. Prepare Tensors
    img_tensor = torch.from_numpy(np_img.transpose(2, 0, 1)).float().unsqueeze(0) / 127.5 - 1.0
    sem_tensor = torch.from_numpy(sem_color_map.transpose(2, 0, 1)).float().unsqueeze(0) / 127.5 - 1.0
    mask_tensor = torch.from_numpy(edit_mask).unsqueeze(0).unsqueeze(0).float()

    img_tensor = img_tensor.to(device_obj)
    sem_tensor = sem_tensor.to(device_obj)
    mask_tensor = mask_tensor.to(device_obj)

    # 5. Perform Translation
    with torch.no_grad():
        translated_tensor = net_G(img_tensor, sem_tensor, mask_tensor)

    # Convert back to uint8 image
    trans_np = ((translated_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    # 6. Compute Preservation Metrics
    out_l1 = compute_outside_mask_l1(trans_np, np_img, edit_mask)
    b_fscore = compute_boundary_fscore(trans_np, np_img, edit_mask)
    psnr_val = compute_psnr(trans_np, np_img, edit_mask)

    print("\n---------------- Preserved Region Quality ----------------")
    print(f"Outside-Mask L1 Error : {out_l1:.4f} (Goal: Minimal / Near 0)")
    print(f"Boundary F-Score      : {b_fscore:.4f} (Goal: High edge preservation)")
    print(f"PSNR in Preserved Zone: {psnr_val:.2f} dB (Goal: High structural fidelity)")
    print("----------------------------------------------------------\n")

    # 7. Save Individual and Composite Outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_trans_path = os.path.join(output_dir, f"{base_name}_translated_{edit_class}.png")
    out_sem_path = os.path.join(output_dir, f"{base_name}_semantic_map.png")
    out_mask_path = os.path.join(output_dir, f"{base_name}_mask_{edit_class}.png")
    out_grid_path = os.path.join(output_dir, f"{base_name}_comparison_grid.png")

    Image.fromarray(trans_np).save(out_trans_path)
    Image.fromarray(sem_color_map).save(out_sem_path)
    Image.fromarray((edit_mask * 255).astype(np.uint8)).save(out_mask_path)

    save_comparison_grid(
        input_img=np_img,
        semantic_map=sem_color_map,
        edit_mask=(edit_mask * 255).astype(np.uint8),
        translated=trans_np,
        save_path=out_grid_path,
        title=f"Semantic-Controlled Translation: Editing '{edit_class}' Region"
    )

    print(f"Outputs successfully saved to:")
    print(f"  - Translated Image:   {out_trans_path}")
    print(f"  - Semantic Color Map: {out_sem_path}")
    print(f"  - Edit Mask:          {out_mask_path}")
    print(f"  - Multi-Panel Grid:   {out_grid_path}")
    print("=======================================================\n")

    return {
        "translated_path": out_trans_path,
        "semantic_path": out_sem_path,
        "mask_path": out_mask_path,
        "grid_path": out_grid_path,
        "outside_l1": out_l1,
        "boundary_fscore": b_fscore,
        "psnr": psnr_val,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Semantic-Controlled Translation on Own Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--edit_class", type=str, default="sky", help="Target class to edit (e.g., sky, road, car, vegetation)")
    parser.add_argument("--edit_color", type=str, default=None, help="Target RGB color to edit (e.g. '70,130,180')")
    parser.add_argument("--mask", type=str, default=None, help="Optional pre-existing binary mask file")
    parser.add_argument("--semantic_map", type=str, default=None, help="Optional pre-existing semantic color map")
    parser.add_argument("--method", type=str, default="deep", choices=["deep", "color"], help="Segmentation method")
    parser.add_argument("--style", type=str, default="sunset", help="Target translation style")
    parser.add_argument("--output_dir", type=str, default="outputs/user_test", help="Directory for output images")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    args = parser.parse_args()
    device_arg = "cpu" if args.cpu else None

    run_inference_on_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        edit_class=args.edit_class,
        edit_color=args.edit_color,
        custom_mask_path=args.mask,
        custom_semantic_path=args.semantic_map,
        segmentation_method=args.method,
        style=args.style,
        output_dir=args.output_dir,
        image_size=(args.height, args.width),
        device=device_arg,
    )
