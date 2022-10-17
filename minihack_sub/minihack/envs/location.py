
import random
from tracemalloc import start



def location(distance = None, size = None):

    distance = distance
    size = size
    xy = []
    
    for x in range(distance):

        y = distance - x

        if x == 0:
            xy.append([x,  y])
            xy.append([x, -y])
            xy.append([-y, x])
            xy.append([y,  x])
    
        elif x == y:
            xy.append([x,  y])
            xy.append([x, -y])
            xy.append([-x, y])
            xy.append([-x,-y])

        else:
            xy.append([x,  y])
            xy.append([x, -y])
            xy.append([-x, y])
            xy.append([-x,-y])

    start_point = [random.randint(0, size - 1), random.randint(0, size - 1)]

    rand_idx = random.randint(0, len(xy)-1)

    x = xy[rand_idx][0]
    y = xy[rand_idx][1]

    e_x = start_point[0] + x
    e_y = start_point[1] + y

    goal = 0 <= e_x < size and 0 <= e_y < size 

    while not goal:
        rand_idx = random.randint(0, len(xy)-1)
        x = xy[rand_idx][0]
        y = xy[rand_idx][1]
    

        e_x = start_point[0] + x
        e_y = start_point[1] + y


        goal = 0 <= e_x <size and 0 <= e_y < size 

    return (start_point[0], start_point[1]), (e_x, e_y)
       

