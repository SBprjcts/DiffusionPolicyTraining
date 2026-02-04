"""
Visualize raw frames and actions from the custom PushT dataset.

Reads directly from the raw_episodes/ folder (PNG frames + JSON data),
no LeRobot or parquet dependency needed.

Outputs:
  1. A grid of sample frames with action arrows → outputs/visualizations/frame_grid_ep<N>.png
  2. A full episode video with action overlays  → outputs/visualizations/episode_<N>.mp4

Usage:
  python visualize_raw_data.py
  python visualize_raw_data.py --episode 5 --num_grid_frames 16
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio


def load_raw_episode(raw_dir: Path, episode_index: int):
    """Load PNG frames and JSON data for one episode from raw_episodes/."""
    ep_dir = raw_dir / f"episode_{episode_index:04d}"
    if not ep_dir.exists():
        return None, None

    json_path = ep_dir / "episode_data.json"
    with open(json_path) as f:
        episode_data = json.load(f)

    frames = []
    for entry in episode_data:
        idx = entry["frame_index"]
        img_path = ep_dir / f"frame_{idx:04d}.png"
        img = np.array(Image.open(img_path))
        frames.append(img)

    return frames, episode_data


def list_available_episodes(raw_dir: Path) -> list[int]:
    """Return sorted list of episode indices found in raw_episodes/."""
    episodes = []
    for d in sorted(raw_dir.iterdir()):
        if d.is_dir() and d.name.startswith("episode_"):
            try:
                episodes.append(int(d.name.split("_")[1]))
            except ValueError:
                pass
    return episodes


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


def make_frame_grid(frames, episode_data, num_frames: int, output_path: Path, episode_index: int):
    """Create a grid of sample frames with action arrows overlaid."""
    total = len(frames)
    step = max(1, total // num_frames)
    indices = list(range(0, total, step))[:num_frames]

    cols = min(4, len(indices))
    rows = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for plot_idx, frame_idx in enumerate(indices):
        r, c = divmod(plot_idx, cols)
        ax = axes[r, c]

        img = frames[frame_idx]
        entry = episode_data[frame_idx]
        state = entry["observation.state"]
        action = entry["action"]

        ax.imshow(img)
        draw_action_arrow(ax, state, action, img.shape[0], img.shape[1])
        ax.set_title(f"Frame {frame_idx}", fontsize=9)
        ax.axis("off")

    for plot_idx in range(len(indices), rows * cols):
        r, c = divmod(plot_idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Episode {episode_index} — Raw Frames with Action Arrows (red = state -> action)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved frame grid to {output_path}")


def make_episode_video(frames, episode_data, output_path: Path, fps: int = 10):
    """Render a full episode as an MP4 video with action overlays."""
    rendered = []

    for i, img in enumerate(frames):
        entry = episode_data[i]
        state = entry["observation.state"]
        action = entry["action"]

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

        rendered.append(frame_rgb)

    imageio.mimsave(str(output_path), rendered, fps=fps)
    print(f"Saved episode video to {output_path} ({len(rendered)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Visualize raw PushT dataset frames and actions")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="custom_pusht_data/raw_episodes",
        help="Path to raw_episodes/ directory",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--num_grid_frames", type=int, default=16, help="Number of frames in the grid")
    parser.add_argument("--fps", type=int, default=10, help="FPS for output video")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations", help="Output directory")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Raw data directory not found: {raw_dir}")
        print("Run collect_data.py first to collect episodes.")
        return

    available = list_available_episodes(raw_dir)
    print(f"Found {len(available)} episodes in {raw_dir}")
    if not available:
        return

    if args.episode not in available:
        print(f"Episode {args.episode} not found. Available: {available[0]}–{available[-1]}")
        return

    print(f"Loading episode {args.episode}...")
    frames, episode_data = load_raw_episode(raw_dir, args.episode)
    print(f"Episode {args.episode}: {len(frames)} frames")

    # 1. Frame grid (PNG)
    grid_path = output_dir / f"frame_grid_ep{args.episode}.png"
    make_frame_grid(frames, episode_data, args.num_grid_frames, grid_path, args.episode)

    # 2. Full episode video (MP4)
    video_path = output_dir / f"episode_{args.episode}.mp4"
    make_episode_video(frames, episode_data, video_path, fps=args.fps)


if __name__ == "__main__":
    main()
