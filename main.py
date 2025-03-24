import SnakeEnv
snake = SnakeEnv.SnakeEnv()

MOV_DICT = {
    "UP":0,
    "RIGHT":1,
    "DOWN":2,
    "LEFT":3,
}

while True:
    snake.render()
    print(snake.step(int(input("0=haut, 1=droite, 2=bas, 3=gauche"))))