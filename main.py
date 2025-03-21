import SnakeEnv
snake = SnakeEnv.SnakeEnv()



while True:
    snake.render()
    print(snake.step(int(input("0=haut, 1=droite, 2=bas, 3=gauche"))))