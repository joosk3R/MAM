from pathlib import Path
import argparse

from import_py import *
from function_py import *


def parse_exclude_list(exclude_text):
    """Parse comma-separated FDI tooth numbers, e.g. '18,28'."""
    if not exclude_text:
        return []
    return [int(x.strip()) for x in exclude_text.split(',') if x.strip()]


def load_state_dict_safely(checkpoint_path, map_location):
    """Load either a raw state_dict or a checkpoint dict containing a state_dict."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Inference for MAM: Metal Artifact-Aware Tooth Segmentation"
    )
    parser.add_argument(
        "--choice",
        type=int,
        choices=[1, 2],
        required=True,
        help="1: save individual tooth masks, 2: save full-tooth segmentation masks",
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing DICOM files for one CBCT case",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save segmentation outputs",
    )
    parser.add_argument(
        "--exclude_list",
        type=str,
        default="",
        help="Comma-separated FDI tooth numbers to exclude, e.g. '18,28'",
    )
    parser.add_argument(
        "--pt_file",
        type=str,
        default="pt_file",
        help="Root directory containing checkpoint experiments",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default="all_model_pt",
        help="Experiment/checkpoint folder name under --pt_file",
    )
    return parser


def main():
    args = build_argparser().parse_args()

    dicom_dir = Path(args.directory)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not dicom_dir.is_dir():
        raise NotADirectoryError(f"DICOM directory not found: {dicom_dir}")

    exclude_list = parse_exclude_list(args.exclude_list)

    pt_file = Path(args.pt_file)
    experiment_id = args.experiment_id
    output_dir = pt_file / experiment_id / "output"
    json_dir = output_dir / "json"
    checkpoint_dir = pt_file / experiment_id / "checkpoint"

    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path_voi_crop = checkpoint_dir / "model_voi_crop.pth"
    checkpoint_path_bbox_detection = checkpoint_dir / "model_bbox.pth"
    checkpoint_path_segmentation = checkpoint_dir / "model_seg.pt"

    print(f"Using device: {device}")
    print(f"DICOM directory: {dicom_dir}")
    print(f"Output directory: {save_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Step 1. VOI localization
    model_voi_crop = VGG16_depth4_fc1_concat(base_dim=8, num_classes=6).to(device)
    load_pretrain(model_voi_crop, str(checkpoint_path_voi_crop))
    model_voi_crop.eval()

    mip_images, dims, resizes, org_dcm, dcm_metainfo = load_dcm_to_mip(str(dicom_dir))
    val_dataset = SNHUMipDataset(
        images=mip_images,
        dims=dims,
        org=org_dcm,
        affine=dcm_metainfo,
        augmentation=False,
        degree=(-10, 10),
        translate=(0.2, 0.2),
        p=1,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    all_results = []
    with torch.no_grad():
        for image, dim, org, initial_position, voxel_spacing in val_loader:
            image = image.type(torch.float32).to(device)
            output = model_voi_crop(image)
            for j in range(org.size(0)):
                pred = output[j]
                upper_meta = {
                    "angle": pred[2].detach().cpu().item(),
                    "p": [pred[0].detach().cpu().item(), pred[1].detach().cpu().item()],
                }
                lower_meta = {
                    "angle": pred[5].detach().cpu().item(),
                    "p": [pred[3].detach().cpu().item(), pred[4].detach().cpu().item()],
                }
                json_object = {
                    "Dimension": dim,
                    "Upper": upper_meta,
                    "Lower": lower_meta,
                }
                all_results.append(
                    {
                        "image": org.detach().cpu().numpy(),
                        "dim": dim,
                        "output": json_object,
                        "initial_position": initial_position,
                        "spacing": voxel_spacing,
                    }
                )

    if not all_results:
        raise RuntimeError("VOI localization produced no result.")

    # Step 2. Crop upper/lower CT volumes
    detection_data = []
    print("STEP 2 - upper")
    upper_crop_img, upper_mat, initial_position, voxel_spacing, dims = detect_CT(
        all_results[0]["image"],
        all_results[0]["output"],
        all_results[0]["initial_position"],
        all_results[0]["spacing"],
        all_results[0]["dim"],
        None,
        None,
        upper_lower="upper",
    )
    detection_data.append((upper_crop_img, upper_mat, "upper", initial_position, voxel_spacing, dims))

    print("STEP 2 - lower")
    lower_crop_img, lower_mat, initial_position, voxel_spacing, dims = detect_CT(
        all_results[0]["image"],
        all_results[0]["output"],
        all_results[0]["initial_position"],
        all_results[0]["spacing"],
        all_results[0]["dim"],
        None,
        None,
        upper_lower="lower",
    )
    detection_data.append((lower_crop_img, lower_mat, "lower", initial_position, voxel_spacing, dims))

    test_infer = Compose(
        [
            AddChanneld(keys=["resize_image"]),
            NormalizeIntensityd(keys=["resize_image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["resize_image", "cls"], device=device),
        ]
    )

    # Step 3. Bbox detection and segmentation models
    model_box = HourGlass3D(
        nStack=2,
        nBlockCount=4,
        nResidualEachBlock=1,
        nMidChannels=128,
        nChannels=128,
        nJointCount=128,
        bUseBn=True,
    ).to(device)
    model_box.load_state_dict(load_state_dict_safely(checkpoint_path_bbox_detection, device))
    model_box.eval()

    model_seg = UNet3D(n_channels=1, n_classes=1).to(device)
    model_seg.load_state_dict(load_state_dict_safely(checkpoint_path_segmentation, device))
    model_seg.eval()

    test_dataloader = get_detection_dataloader(detection_data, test_infer)

    print("Models and data are ready.")
    if args.choice == 1:
        print("Running individual tooth segmentation...")
        main_1(model_box, model_seg, test_dataloader, exclude_list, str(save_dir))
    elif args.choice == 2:
        print("Running full-tooth segmentation...")
        main_2(model_box, model_seg, test_dataloader, exclude_list, str(save_dir))

    print("Done.")


if __name__ == "__main__":
    main()
