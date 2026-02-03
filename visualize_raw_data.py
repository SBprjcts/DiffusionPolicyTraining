"""
Visualize raw frames and actions from the custom PushT dataset.

This script loads your locally collected custom_pusht_data and renders:
  1. A grid of sample frames with action arrows → outputs/visualizations/frame_grid_ep<N>.png
  2. A full episode video with action overlays  → outputs/visualizations/episode_<N>.mp4

Usage:
  python visualize_raw_data.py
  python visualize_raw_data.py --episode 1 --num_grid_frames 16
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio


def load_dataset(dataset_path: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(dataset_path)
    return dataset


def get_episode_indices(dataset, episode_index: int) -> list[int]:
    """Return global indices for all frames in the given episode."""
    indices = []
    for i in range(len(dataset)):
        item = dataset.hf_dataset[i]
        if item["episode_index"] == episode_index:
            indices.append(i)
    return indices


def frame_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW float [0,1] tensor to HWC uint8 numpy array."""
    if image_tensor.dim() == 3 and image_tensor.shape[0] in (1, 3):
        image_tensor = image_tensor.permute(1, 2, 0)
    img = (image_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
    return img


def draw_action_arrow(ax, state, action, img_h, img_w, color="red"):
    """Draw an arrow from current state (agent pos) to action target.

    state and action are in PushT coordinate space (~0-512).
    We scale them to image pixel coordinates.
    """
    scale_x = img_w / 512.0
    scale_y = img_h / 512.0

    sx, sy = state[0] * scale_x, state[1] * scale_y
    ax_val, ay_val = action[0] * scale_x, action[1] * scale_y

    ax.annotate(
        "",
        xy=(ax_val, ay_val),
        xytext=(sx, sy),
        arrowprops=dict(arrowstyle="->", color=color, lw=2),
    )


def make_frame_grid(dataset, episode_indices: list[int], num_frames: int, output_path: Path):
    """Create a grid of sample frames with action arrows overlaid."""
    total = len(episode_indices)
    step = max(1, total // num_frames)
    selected = episode_indices[::step][:num_frames]

    cols = min(4, len(selected))
    rows = (len(selected) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, global_idx in enumerate(selected):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        item = dataset[global_idx]
        img = frame_to_numpy(item["observation.image"])
        state = item["observation.state"].numpy()
        action = item["action"].numpy()

        ax.imshow(img)
        draw_action_arrow(ax, state, action, img.shape[0], img.shape[1])
        ax.set_title(f"Frame {item['frame_index'].item()}", fontsize=9)
        ax.axis("off")

    for idx in range(len(selected), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle("Raw PushT Frames with Action Arrows (red = state -> action)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved frame grid to {output_path}")


def make_episode_video(dataset, episode_indices: list[int], output_path: Path, fps: int = 10):
    """Render a full episode as an MP4 video with action overlays."""
    frames = []

    for global_idx in episode_indices:
        item = dataset[global_idx]
        img = frame_to_numpy(item["observation.image"])
        state = item["observation.state"].numpy()
        action = item["action"].numpy()

        dpi = 100
        h, w = img.shape[:2]
        fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
        ax.imshow(img)
        draw_action_arrow(ax, state, action, h, w)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame_rgb = buf[:, :, :3].copy()
        plt.close(fig)

        frames.append(frame_rgb)

    imageio.mimsave(str(output_path), frames, fps=fps)
    print(f"Saved episode video to {output_path} ({len(frames)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Visualize raw PushT dataset frames and actions")
    parser.add_argument(
        "--dataset",
        type=str,
        default="custom_pusht_data/custom_pusht",
        help="Path to local dataset directory",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--num_grid_frames", type=int, default=16, help="Number of frames in the grid")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"Dataset loaded: {len(dataset)} total frames")

    episode_indices = get_episode_indices(dataset, args.episode)
    if not episode_indices:
        print(f"No frames found for episode {args.episode}.")
        return

    print(f"Episode {args.episode}: {len(episode_indices)} frames")

    # 1. Frame grid (PNG)
    grid_path = output_dir / f"frame_grid_ep{args.episode}.png"
    make_frame_grid(dataset, episode_indices, args.num_grid_frames, grid_path)

    # 2. Full episode video (MP4)
    video_path = output_dir / f"episode_{args.episode}.mp4"
    make_episode_video(dataset, episode_indices, video_path, fps=dataset.fps)


if __name__ == "__main__":
    main()
