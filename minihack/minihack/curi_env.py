
from minihack import LevelGenerator

def des_file_reset(size, start, end, lit = True):

    lvl_gen = LevelGenerator(w=size, h=size, lit=lit)
            
    lvl_gen.add_goal_pos(end)
    lvl_gen.set_start_pos(start)

    return lvl_gen.get_des()
