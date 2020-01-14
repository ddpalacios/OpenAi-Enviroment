segments = []
import random
import turtle
from time import sleep
class Collide:
    def __init__(self, player, food):
        self.player = player
        self.food = food

    def hit_wall(self, head, is_done):
        hitWall = False
        # Check for a collision with the border

        if head.xcor() > 290 or head.xcor() < -290 or head.ycor() > 290 or head.ycor() < -290:
            print("HIT WALL\n\n\n")
            head.reset()
            hitWall = True
            #    sleep(2)
            head.goto(0, 0)
            self.food.goto(0, 100)
            head.direction = "stop"
            is_done = True
            print(is_done)

            # Hide the segments
            for segment in segments:
                segment.goto(1000, 1000)

            # Clear the segments list
            segments.clear()
        return is_done, hitWall

    def hit_food(self, food, head, reward):
        # food.reset()
        ate = False
        if head.distance(food) < 20:
            print("FOOD OBTAINED")
            sleep(.5)
            ate = True

            # Move the food to a random spot
            x = random.randint(-290, 290)
            y = random.randint(-290, 290)
            food.goto(x, y)
            # Add a segment
            new_segment = turtle.Turtle()
            new_segment.speed(0)
            new_segment.shape("square")
            new_segment.color("grey")
            new_segment.penup()
            segments.append(new_segment)

            # Move the end segments first in reverse order
        for index in range(len(segments) - 1, 0, -1):
            x = segments[index - 1].xcor()
            y = segments[index - 1].ycor()
            segments[index].goto(x, y)

        # Move segment 0 to where the head is
        if len(segments) > 0:
            x = head.xcor()
            y = head.ycor()
            segments[0].goto(x, y)

        return ate