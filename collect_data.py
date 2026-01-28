"""
PushT Data Collection Script — run from terminal:
    python collect_data.py

Controls:
  - Hover mouse near the blue circle to grab the agent
  - Push the T-block into the green target area
  - Space: hold to pause
  - R: retry current episode
  - Q: quit collection
"""

import shutil
from pathlib import Path

import cv2
import gymnasium as gym
import gym_pusht  # registers PushT-v0 with gymnasium
import numpy as np
import pygame
import pymunk.pygame_util
from PIL import Image
from pymunk.vec2d import Vec2d

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ── Configuration ──────────────────────────────────────────────
NUM_EPISODES = 200
MAX_STEPS_PER_EPISODE = 300  # 300 steps = 30s at 10 fps
FPS = 10
OBS_SIZE = 384  # observation image size
REPO_ID = "custom_pusht"
DATASET_ROOT = Path("./custom_pusht_data") / REPO_ID

# Clean previous data if it exists
if DATASET_ROOT.exists():
    shutil.rmtree(DATASET_ROOT)

# ── Create LeRobot dataset on disk ────────────────────────────
features = {
    "observation.image": {
        "dtype": "video",
        "shape": (OBS_SIZE, OBS_SIZE, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["x", "y"],
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["x", "y"],
    },
}

dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    fps=FPS,
    features=features,
    root=DATASET_ROOT,
    robot_type="pusht_sim",
    use_videos=True,
    image_writer_processes=0,
    image_writer_threads=0,
)

# ── Open PushT environment ────────────────────────────────────
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    render_mode="human",
    observation_width=OBS_SIZE,
    observation_height=OBS_SIZE,
    max_episode_steps=MAX_STEPS_PER_EPISODE,
)

clock = pygame.time.Clock()
episode_count = 0
teleop_active = False

print(f"Collecting up to {NUM_EPISODES} episodes.")
print("Hover near the blue circle to grab. Push T onto target.")
print("Space=pause, R=retry, Q=quit\n")


def get_mouse_action(unwrapped_env):
    """Replicate teleop_agent logic using env.window instead of env.screen."""
    global teleop_active
    mouse_position = pymunk.pygame_util.from_pygame(
        Vec2d(*pygame.mouse.get_pos()), unwrapped_env.window
    )
    if teleop_active or (mouse_position - unwrapped_env.agent.position).length < 30:
        teleop_active = True
        return np.array(mouse_position, dtype=np.float32)
    return None


def capture_image(unwrapped_env):
    """Capture an observation image from the env's internal draw surface."""
    screen = unwrapped_env._draw()
    img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    img = cv2.resize(img, (OBS_SIZE, OBS_SIZE))
    return img


while episode_count < NUM_EPISODES:
    obs, info = env.reset()
    teleop_active = False

    episode_frames = []
    retry = False
    pause = False
    done = False

    pygame.display.set_caption(f"Episode {episode_count + 1}/{NUM_EPISODES}")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                dataset.finalize()
                print(f"\nQuit early. Saved {episode_count} episodes to: {DATASET_ROOT}")
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    env.close()
                    dataset.finalize()
                    print(f"\nQuit. Saved {episode_count} episodes to: {DATASET_ROOT}")
                    raise SystemExit
                if event.key == pygame.K_r:
                    retry = True
                if event.key == pygame.K_SPACE:
                    pause = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    pause = False

        if retry:
            break
        if pause:
            clock.tick(FPS)
            continue

        action = get_mouse_action(env.unwrapped)
        if action is None:
            clock.tick(FPS)
            continue

        # Capture image BEFORE stepping (from current state)
        img = capture_image(env.unwrapped)
        agent_pos = np.array(env.unwrapped.agent.position, dtype=np.float32)

        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_frames.append({
            "observation.image": Image.fromarray(img),
            "observation.state": agent_pos,
            "action": action,
            "task": "Push the T-block onto the target.",
        })

        obs = next_obs
        done = terminated or truncated
        clock.tick(FPS)

    if retry:
        print(f"  Retrying episode {episode_count + 1}")
        continue

    for frame in episode_frames:
        dataset.add_frame(frame)
    dataset.save_episode()
    episode_count += 1
    print(f"Episode {episode_count}/{NUM_EPISODES} saved ({len(episode_frames)} steps)")

env.close()
dataset.finalize()
print(f"\nDone! Dataset saved to: {DATASET_ROOT}")
