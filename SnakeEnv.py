import numpy as np
import random


class SnakeEnv:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset()
        self.apple_reward = 100
        self.death_reward = -3
        self.step_reward = 0.02
        self.heading_reward = 0.2

    def reset(self):
        """Réinitialise l'environnement et retourne l'état initial."""
        self.snake = [(self.height // 2, self.width // 2)]  # Serpent au centre
        self.direction = (0, 1)  # Direction initiale (droite)
        self.place_food()
        self.done = False
        return self.get_state()

    def place_food(self):
        """Place une pomme aléatoirement sur la grille."""
        empty_cells = [(i, j) for i in range(self.height) for j in range(self.width) if (i, j) not in self.snake]
        self.food = random.choice(empty_cells)

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}
        old_dist = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1]) # distance to apple from position before movement

        directions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.direction = directions[action]
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])


        death_reason = None
        ate_apple = False

        # Collision avec mur
        if new_head[0] < 0 or new_head[0] >= self.height or new_head[1] < 0 or new_head[1] >= self.width:
            self.done = True
            death_reason = "wall"
            return self.get_state(), self.death_reward, True, {"death_reason": death_reason, "ate_apple": False}

        # Collision avec soi-même
        if new_head in self.snake:
            self.done = True
            death_reason = "self"
            return self.get_state(), self.death_reward, True, {"death_reason": death_reason, "ate_apple": False}

        self.snake.insert(0, new_head) # placing the new head !! after checking this new position is possible !!

        new_dist = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1]) # distance to apple from position after movement

        # === REWARDS ===
        if new_head == self.food:
            ate_apple = True
            self.place_food()
            reward = self.apple_reward
        else:
            self.snake.pop()
            #reward = self.step_reward
            delta = old_dist - new_dist # bonus if is going towards apple
            reward = self.heading_reward if delta > 0 else -self.heading_reward

        return self.get_state(), reward, self.done, {
            "death_reason": death_reason,
            "ate_apple": ate_apple
        }

    def get_state(self):
        """
        Retourne la grille avec bordures, sous forme de matrice 2D.
        0 = fond, 1 = serpent, 2 = pomme, 3 = mur
        """
        grid = np.zeros((self.height + 2, self.width + 2), dtype=int)

        # murs tout autour
        grid[0, :] = 3  # haut
        grid[-1, :] = 3  # bas
        grid[:, 0] = 3  # gauche
        grid[:, -1] = 3  # droite

        ## placer le serpent (+1 à chaque coordonnée à cause des murs)
        #for y, x in self.snake:
        #    grid[y + 1, x + 1] = 1

        # placer la tête (valeur spéciale)
        head_y, head_x = self.snake[0]
        grid[head_y + 1, head_x + 1] = 4

        # placer le reste du corps
        for y, x in self.snake[1:]:
            grid[y + 1, x + 1] = 1

        # placer la pomme
        grid[self.food[0] + 1, self.food[1] + 1] = 2

        return grid

    def render(self):
        """Affiche la grille dans le terminal."""
        grid = self.get_state()
        symbols = {0: '.', 1: 'S', 2: 'F', 3: 'X'}
        for row in grid:
            print(' '.join(symbols[cell] for cell in row))
        print()
