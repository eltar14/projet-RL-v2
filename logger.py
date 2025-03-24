import csv
import os
from datetime import datetime

class TrainingLogger:
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/snake_dqn_log_{timestamp}.csv"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename
        self._init_file()

    def _init_file(self):
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "reward", "epsilon", "length",
                "steps", "apples", "death_reason", "time_ms",
                "avg_loss", "avg_q"
            ])

    def log(self, episode, reward, epsilon, length, steps, apples, death_reason, time_ms, avg_loss, avg_q):
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, reward, round(epsilon, 5), length,
                steps, apples, death_reason, int(time_ms),
                round(avg_loss, 6), round(avg_q, 6)
            ])
