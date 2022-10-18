import gym
import minihack
env = gym.make("MiniHack-Room-Random_curi-5x5-v0")
stair = "(1,3)"
branch = "(3,1,3,1),(2,0,2,0)"
env.reset(stair, branch) # each reset generates a new environment instance

# state = env.step(2)[0]  # move agent '@' north
# state = env.step(0)[0]  # move agent '@' north

env.render()

# print("hihi")
# print(env.distance)
# env = gym.make("MiniHack-Room-Random_curi-5x5-v0")

stair = "(1,4)"
branch = "(3,1,3,1),(2,0,2,0)"

env.reset(stair, branch) # each reset generates a new environment instance

env.render()


#0 : 위쪽 
#1 : 오른쪽
#2 : 아래쪽
#3 : 왼쪽 


#4 : 위 오른 
#5 : 아래 오른
#6 :
#7 :