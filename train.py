from SnakeEnv import SnakeEnv
from Agent import DQNAgent
from logger import TrainingLogger
import time
import numpy as np
import os
os.makedirs("models", exist_ok=True)
import torch

env = SnakeEnv(width=10, height=10)
state = env.reset()
dummy_stack = np.stack([state for _ in range(4)], axis=0)  # simulate la stack
agent = DQNAgent(state_shape=dummy_stack.shape, action_size=4)

n_episodes = 30000

logger = TrainingLogger()

for ep in range(n_episodes):
    state = env.reset()
    agent.init_stack(state)
    stacked_state = agent.get_stacked_state()

    done = False
    total_reward = 0
    steps = 0
    apples = 0

    losses = []
    q_values = []

    start_time = time.time()
    death_reason = "none"

    while not done:
        action = agent.get_action(stacked_state)
        next_state, reward, done, info = env.step(action)

        agent.update_stack(next_state)
        next_stacked_state = agent.get_stacked_state()

        agent.store_transition(stacked_state, action, reward, next_stacked_state, done)
        agent.train()

        stacked_state = next_stacked_state
        #print(stacked_state.shape)
        total_reward += reward
        steps += 1
        #if reward == 1:
        #    apples += 1
        if info.get("ate_apple", False):
            apples += 1

        if done:
            death_reason = info.get("death_reason", "unknown")

        loss, q = agent.train()
        if loss is not None:
            losses.append(loss)
            q_values.append(q)

    avg_loss = np.mean(losses) if losses else 0
    avg_q = np.mean(q_values) if q_values else 0

    duration = (time.time() - start_time) * 1000

    logger.log(
        episode=ep + 1,
        reward=total_reward,
        epsilon=agent.epsilon,
        length=len(env.snake),
        steps=steps,
        apples=apples,
        death_reason=death_reason,
        time_ms=duration,
        avg_loss=avg_loss,
        avg_q=avg_q
    )

    print(f"[EP {ep + 1}] Reward={total_reward:.2f}, Steps={steps}, Apples={apples}, Death={death_reason}, Epsilon={agent.epsilon:.3f}, avg_loss={avg_loss:.2f}, avg_q={avg_q:.2f}")

    if (ep + 1) % 4000 == 0:
        torch.save({
            'episode': ep,
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'replay_buffer': list(agent.memory),  # attention à la taille
        }, "models/checkpoint.pth")
