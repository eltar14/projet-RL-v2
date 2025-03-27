import pygame
import time
import torch
import numpy as np
import os
from datetime import datetime
import imageio
from SnakeEnv import SnakeEnv
from Agent import DQNAgent
from utils import load_model_state

CELL_SIZE = 32
MARGIN = 1
colors = {
    0: (30, 30, 30),  # background
    1: (0, 200, 0),  # snake body
    2: (255, 0, 0),  # apple
    3: (100, 100, 100),  # wall
    4: (0, 255, 0)  # head
}

def draw_grid(screen, grid):
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            val = grid[y, x]
            color = colors.get(val, (255, 255, 255))
            rect = pygame.Rect(x * (CELL_SIZE + MARGIN), y * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

def draw_text(screen, font, text, y, grid_height):
    surf = font.render(text, True, (255, 255, 255))
    screen.blit(surf, (5, grid_height * (CELL_SIZE + MARGIN) + y))

def run_pygame_visual(model_path, fps=5, width=10, height=10, record_gif=False):
    pygame.init()
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    env = SnakeEnv(width=width, height=height)
    state = env.reset()
    stacked_shape = (4, *state.shape)
    agent = DQNAgent(state_shape=stacked_shape, action_size=4)
    load_model_state(agent.model, model_path, device=agent.device, eval_mode=True)
    agent.epsilon = 0.0

    grid_height, grid_width = state.shape
    window_size = ((CELL_SIZE + MARGIN) * grid_width, (CELL_SIZE + MARGIN) * grid_height + 40)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Snake RL Viewer")

    os.makedirs("outputs", exist_ok=True)

    while True:
        state = env.reset()
        agent.init_stack(state)
        stacked_state = agent.get_stacked_state()

        done = False
        steps = 0
        apples = 0
        frames = []

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.get_action(stacked_state)
            next_state, reward, done, info = env.step(action)
            agent.update_stack(next_state)
            stacked_state = agent.get_stacked_state()

            screen.fill((0, 0, 0))
            draw_grid(screen, env.get_state())
            draw_text(screen, font, f"steps: {steps}  apples: {apples}  death: {info.get('death_reason', 'none')}", 5, grid_height)
            pygame.display.flip()

            if record_gif:
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))  # convert to HWC
                frames.append(frame)

            if info.get("ate_apple", False):
                apples += 1
            steps += 1
            clock.tick(fps)

        if record_gif:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            score_str = str(apples).zfill(3)  # fixed length score
            gif_path = f"outputs/{score_str}_snake_{timestamp}.gif"
            imageio.mimsave(gif_path, frames, fps=fps)
            print(f"saved gif: {gif_path}")

        time.sleep(1.0)
