import torch
import os
from collections import deque

def load_model_state(model, path, device="cpu", eval_mode=True):
    """
    loads a .pth file into a pytorch model
    automatically handles pure state_dict or full checkpoint
    set eval_mode=False if the model should stay in train mode
    returns extra fields (optimizer_state, epsilon, memory, episode) if present and eval_mode=False
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")  # check file existence

    checkpoint = torch.load(path, map_location=device, weights_only=False)  # load checkpoint safely

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])  # full checkpoint case
        print(f"model loaded from checkpoint: {path}")
        if not eval_mode:
            optimizer_state = checkpoint.get('optimizer_state_dict', None)
            epsilon = checkpoint.get('epsilon', None)
            memory = deque(checkpoint.get('replay_buffer', []))
            episode = checkpoint.get('episode', 0)
            return model, optimizer_state, epsilon, memory, episode
    else:
        model.load_state_dict(checkpoint)  # raw state_dict only
        print(f"model loaded from pure state_dict: {path}")

    if eval_mode:
        model.eval()  # set to eval mode if requested

    return model, None, None, None, None


def save_checkpoint(path, model, optimizer=None, epsilon=None, memory=None, episode=None):
    """
    saves full training checkpoint to the given path
    includes model, optimizer, epsilon, memory, and episode if provided
    """
    checkpoint = {
        'model_state_dict': model.state_dict()
    }

    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epsilon is not None:
        checkpoint['epsilon'] = epsilon
    if memory is not None:
        checkpoint['replay_buffer'] = list(memory)
    if episode is not None:
        checkpoint['episode'] = episode

    torch.save(checkpoint, path)
    print(f"checkpoint saved to: {path}")


def run_simple_visual_test(model_path, fps=3):
    """
    loads a model and runs it visually in loop with step control
    clears terminal for each frame and displays basic stats
    """
    import time
    import numpy as np
    from SnakeEnv import SnakeEnv
    from Agent import DQNAgent

    render_delay = 1.0 / fps  # convert fps to time delay

    env = SnakeEnv(width=10, height=10)
    state = env.reset()
    stacked_state_shape = (4, *state.shape)  # prepare shape for agent init

    agent = DQNAgent(state_shape=stacked_state_shape, action_size=4)
    load_model_state(agent.model, model_path, device=agent.device, eval_mode=True)  # use unified loader
    agent.epsilon = 0.0  # pure exploitation

    while True:
        state = env.reset()
        agent.init_stack(state)
        stacked_state = agent.get_stacked_state()

        done = False
        steps = 0
        apples = 0

        while not done:
            os.system('cls' if os.name == 'nt' else 'clear')  # clear console for next frame

            action = agent.get_action(stacked_state)
            next_state, reward, done, info = env.step(action)

            agent.update_stack(next_state)
            stacked_state = agent.get_stacked_state()

            env.render()  # display the grid
            print(f"apples: {apples} | steps: {steps} | death: {info.get('death_reason', 'none')}")

            if info.get("ate_apple", False):
                apples += 1  # increment if apple eaten

            steps += 1
            time.sleep(render_delay)  # control frame rate

        print("\nrun ended. restarting in 1 second...\n")
        time.sleep(1)  # short pause before restarting
