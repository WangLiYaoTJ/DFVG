import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np
import cv2
from scipy.ndimage import convolve1d


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def compute_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def apply_advanced_filter(brightness_values, kernel):
    return convolve1d(brightness_values, kernel, mode='reflect')


def process_brightness_sequence(folder_path, kernel):
    image_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path)
         if f.lower().endswith(('.jpg', '.png', '.jpeg'))],
        key=natural_sort_key
    )

    raw_brightness = []
    for file in image_files:
        image = cv2.imread(file)
        if image is not None:
            raw_brightness.append(compute_brightness(image))
        else:
            raw_brightness.append(0)

    smoothed = raw_brightness

    return raw_brightness, smoothed


def calculate_A_values(smoothed_brightness):
    A_values = []
    for brightness in smoothed_brightness:
        adjusted = brightness + 100
        clamped = min(adjusted, 255)
        normalized = clamped / 255.0
        A_values.append(normalized)
    return A_values


def split_into_tiles(image_tensor):
    h, w = image_tensor.shape[:2]
    h_split = [h // 2, h - h // 2]
    w_split = [w // 2, w - w // 2]
    tiles = []
    positions = []
    for i in range(2):
        for j in range(2):
            h_start = i * h_split[0]
            h_end = h_start + h_split[i]
            w_start = j * w_split[0]
            w_end = w_start + w_split[j]
            tile = image_tensor[h_start:h_end, w_start:w_end]
            tiles.append(tile)
            positions.append((h_start, h_end, w_start, w_end))
    return tiles, positions


def merge_tiles(tiles, positions, target_shape, device='cpu'):
    merged = torch.zeros(target_shape, device=device)
    for (h_start, h_end, w_start, w_end), tile in zip(positions, tiles):
        merged[h_start:h_end, w_start:w_end] = tile.to(device)
    return merged


def rgb_to_depth(rgb_image, cmap="Spectral", min_depth=0, max_depth=1, device='cpu'):
    assert rgb_image.ndim == 3 and rgb_image.shape[2] == 3, "The input must be an RGB image"

    cm = plt.colormaps[cmap]
    depth_values = torch.linspace(0, 1, 256, device=device)
    lut = torch.tensor(cm(depth_values.cpu())[:, :3], dtype=torch.float32, device=device)

    tiles, positions = split_into_tiles(rgb_image)
    processed_tiles = []

    for tile in tiles:
        rgb_norm = (tile.float() / 255.0).to(device)
        rgb_expanded = rgb_norm.unsqueeze(2)
        lut_expanded = lut.unsqueeze(0).unsqueeze(0)

        diff = torch.norm(rgb_expanded - lut_expanded, dim=3)
        indices = torch.argmin(diff, dim=2)
        depth_tile = depth_values[indices]
        processed_tiles.append(depth_tile)

    depth_map = merge_tiles(processed_tiles, positions, rgb_image.shape[:2], device)
    return depth_map * (max_depth - min_depth) + min_depth


def generate_haze_image(orig_img, depth_img, A, beta, device='cpu'):
    orig_array = np.array(orig_img)
    h, w = orig_array.shape[:2]

    depth_min = depth_img.min().item()
    depth_max = depth_img.max().item()
    depth_normalized = (depth_img.cpu() - depth_min) / (depth_max - depth_min + 1e-8)

    tiles, positions = split_into_tiles(torch.from_numpy(orig_array))
    processed_tiles = []

    for idx, tile in enumerate(tiles):
        orig_tile = tile.numpy()
        h_start, h_end, w_start, w_end = positions[idx]
        depth_tile = depth_normalized[h_start:h_end, w_start:w_end].to(device)

        orig_tensor = torch.tensor(orig_tile, device=device, dtype=torch.float32) / 255.0
        t = torch.exp(-beta * depth_tile)
        I_haze_tile = orig_tensor * t.unsqueeze(2) + A * (1 - t.unsqueeze(2))
        processed_tiles.append(I_haze_tile.cpu())

    haze_array = torch.zeros((h, w, 3))
    for (h_start, h_end, w_start, w_end), tile in zip(positions, processed_tiles):
        haze_array[h_start:h_end, w_start:w_end] = tile
    return Image.fromarray((haze_array.numpy() * 255).astype(np.uint8))


def process_video_sequence(input_folder, depth_folder, output_folder, kernel,
                           cmap="Spectral", min_depth=0, max_depth=1, beta=0.0, device='cpu'):
    _, smoothed_brightness = process_brightness_sequence(input_folder, kernel)
    A_values = calculate_A_values(smoothed_brightness)

    # 获取排序后的文件名
    filenames = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))],
        key=natural_sort_key
    )

    for idx, filename in enumerate(filenames):
        orig_path = os.path.join(input_folder, filename)
        depth_path = os.path.join(depth_folder, filename.replace(".jpg", ".png").replace(".jpeg", ".png"))
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path):
            print(f"Skip existing file: {filename}")
            continue

        if not os.path.exists(depth_path):
            print(f"Missing depth map: {depth_path}")
            continue

        try:
            image = cv2.imread(orig_path)
            if image is None:
                raise ValueError("Unable to read image files")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_img = Image.fromarray(image_rgb)

            with Image.open(depth_path).convert('RGB') as depth_img:
                depth_tensor = torch.tensor(np.array(depth_img), device=device, dtype=torch.float32)
                depth_map = rgb_to_depth(depth_tensor, cmap, min_depth, max_depth, device)

                current_A = A_values[idx]
                haze_img = generate_haze_image(orig_img, depth_map, current_A, beta, device)
                haze_img.save(output_path)
                print(f"Successfully process: {filename} (A={current_A:.5f})")

        except Exception as e:
            print(f"Fail to process {filename}: {str(e)}")


def set_gpu_device(gpu_id):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Use GPU {gpu_id}")
    else:
        device = torch.device('cpu')
        print("Use CPU")
    return device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fog Generator')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--dataset', type=str, default="ship", help='dataset to test')
    parser.add_argument('--phase', type=str, default="test", help='train or test')
    parser.add_argument('--data_dir', type=str, default='data', help='data dir')
    parser.add_argument('--list_dir', type=str, default='lists', help='list dir')
    parser.add_argument('--kernel', type=float, nargs='+', default=[0.1, 0.2, 0.4, 0.2, 0.1],
                        help='LightAdaptiveKernel')

    args = parser.parse_args()

    device = set_gpu_device(args.gpu_id)

    list_filename = os.path.join(args.list_dir, f"{args.dataset}_{args.phase}.txt")
    with open(list_filename) as f:
        video_list = [line.strip() for line in f]

    for video in video_list:
        print(f"\nStart processing: {video}")

        beta = 2.0
        input_folder = os.path.join(args.data_dir, args.phase, "input", video)
        depth_folder = os.path.join(args.data_dir, args.phase, "output/final_depth", video)
        output_folder = os.path.join(args.data_dir, args.phase, f"output_haze_dynamicA_beta{beta}", video)

        os.makedirs(output_folder, exist_ok=True)
        print(f"Output dir has been created: {output_folder}")

        process_video_sequence(
            input_folder=input_folder,
            depth_folder=depth_folder,
            output_folder=output_folder,
            kernel=args.kernel,
            cmap="Spectral",
            min_depth=0,
            max_depth=10,
            beta=beta,
            device=device
        )
