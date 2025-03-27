import SnakeEnv
from utils import load_model_state, run_simple_visual_test
#snake = SnakeEnv.SnakeEnv()

MOV_DICT = {
    "UP":0,
    "RIGHT":1,
    "DOWN":2,
    "LEFT":3,
}
def demo_console():
    model_path = "models/checkpoint - Copie (5).pth"
    run_simple_visual_test(model_path)

def demo_pygame():
    from pygame_display import run_pygame_visual

    model_path = "models/checkpoint - Copie (5).pth"  # update this path if needed
    run_pygame_visual(model_path, fps=5, width=10, height=10, record_gif=True)


if __name__ == "__main__":
    demo_pygame()

