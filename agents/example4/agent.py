import random

from ExampleAgent import ExampleAgent

class Agent(ExampleAgent):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

        self.dxy = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.parent = [[None for _ in range(79)] for _ in range(21)]
        self.children = [[None for _ in range(79)] for _ in range(21)]
        self.visited = [[False for _ in range(79)] for _ in range(21)]
        self.goal = (None, None)
        self.dungeon_lv = 1
    
    def new_lv(self):
        self.parent = [[None for _ in range(79)] for _ in range(21)]
        self.children = [[None for _ in range(79)] for _ in range(21)]
        self.visited = [[False for _ in range(79)] for _ in range(21)]
        self.goal = (None, None)

    def get_action(self, env, obs):
        cur_lv = obs['blstats'][12]
        time = obs['blstats'][20]
        if self.dungeon_lv != cur_lv or time == 1:
            self.dungeon_lv = cur_lv
            self.new_lv()

        x, y = obs['blstats'][:2]
        screen = obs['tty_chars']

        if self.is_more(screen):
            action = 0
        elif self.is_yn(screen):
            action = 8
        elif self.is_locked(screen):
            action = 20
        elif (y, x) == self.goal:
            action = 18
        else:
            pre_map = self.preprocess_map(obs)
            self.visited[y][x] = True

            if self.children[y][x] == None:
                py, px = None, None
                if self.parent[y][x] != None:
                    dxy_i = self.parent[y][x] - 1
                    py = y + self.dxy[dxy_i][0]
                    px = x + self.dxy[dxy_i][1]
                
                children_list = []
                for i in range(4):
                    ny = y + self.dxy[i][0]
                    nx = x + self.dxy[i][1]

                    if ny < 0 or ny >= 21 or nx < 0 or nx >= 79:
                        is_child = False
                    else:
                        is_child = pre_map[ny][nx] and (ny, nx) != (py, px) and not self.visited[ny][nx]
                    children_list.append(is_child)
                    if is_child:
                        self.parent[ny][nx] = (i+2)%4 + 1
                
                self.children[y][x] = children_list
            
            for i in range(4):
                ny = y + self.dxy[i][0]
                nx = x + self.dxy[i][1]
                if ny < 0 or ny >= 21 or nx < 0 or nx >= 79:
                    continue
                if self.children[y][x][i] and not self.visited[ny][nx]:
                    action = i+1
                    return action
            action = self.parent[y][x]

        if action == None:
            action = 17

        return action
        
    def preprocess_map(self, obs):
        pre = []

        available = [ord('.'), ord('#')]
        unavailable = [ord(' '), ord('`')]
        door_or_wall = [ord('|'), ord('-')]

        chars = obs['chars']
        colors = obs['colors']
        for y in range(21):
            pre_line = []
            for x in range(79):
                char = chars[y][x]
                color = colors[y][x]
                
                pre_char = True
                if char in unavailable:
                    pre_char = False
                elif char in door_or_wall and color == 7:
                    pre_char = False
                elif char == ord('#') and color == 6:
                    pre_char = False # bar의 경우.
                elif char == ord('>'):
                    self.goal = (y, x)
                pre_line.append(pre_char)
            pre.append(pre_line)
        return pre