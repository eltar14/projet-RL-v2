import SnakeEnv
from utils import load_model_state, run_simple_visual_test
from pygame_display import run_pygame_visual
def demo_console():
    model_path = "models/demo_model_1.pth"
    run_simple_visual_test(model_path)

def demo_pygame():
    model_path = "models/demo_model_1.pth"
    run_pygame_visual(model_path, fps=5, width=10, height=10, record_gif=True)


if __name__ == "__main__":
    demo_pygame()

