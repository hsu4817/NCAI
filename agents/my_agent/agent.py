import random
import gym
import copy
from re import T
from time import sleep
from queue import PriorityQueue

from ExampleAgent import ExampleAgent

INF = 987654321

BLACK = 0
RED	= 1
GREEN = 2
BROWN =	3
BLUE = 4
MAGENTA = 5
CYAN = 6
GRAY = 7
NO_COLOR = 8
ORANGE = 9
BRIGHT_GREEN = 10
YELLOW = 11
BRIGHT_BLUE = 12
BRIGHT_MAGENTA = 13
BRIGHT_CYAN = 14
WHITE = 15
MAX = 16

passive_mobs = set([('b', GREEN), ('j', GREEN), ('F', GREEN), ('R', BROWN), ('b', CYAN), ('e', BLUE), ('j', BLUE), ('F', BROWN), ('F', YELLOW), ('F', RED), ('v', YELLOW), ('E', YELLOW), ('v', BRIGHT_BLUE)])

wall_set = set(['|','-'])
walkable_objects = set(['.','#','<','>','+','@','$','^',')','[','%','?','/','=','!','(','"','*',])

log = open("log.txt", "w")
output = open("output.txt", "w")

def printf(string, end = '\n'):
    output.write(string + end)

def is_valid(y, x):
    return (0 <= y < 21) and (0 <= x < 79) 

def is_monster(letter, color):
    #Deal passive mob as not monster
    if (chr(letter), color) in passive_mobs:
        return False
    if letter >= 65 and letter <= 90:
        return True
    elif letter >= 97 and letter <= 122:
        return True
    elif letter == ord(":"):
        return True
    else:
        return False

def is_enemy(original_map, color_map, pos):
    letter = original_map[pos[1]][pos[0]]
    color = color_map[pos[1]][pos[0]]
    if color_map[pos[1]][pos[0]] == 15:
        return False
    else:
        return is_monster(letter, color)
            
def find_closest_enemy(obs, pos):
    original_map = obs['chars']
    color_map = obs['colors']
    x, y = obs['blstats'][:2]
    enemy_pos_list = []
    for j in range(21):
        for i in range(max(0, x - 10), min(79, x + 11)):
            if manhattan_distance(pos,(i,j)) >= 10:
                continue
            if is_enemy(original_map, color_map, (i, j)):
                enemy_pos_list.append((i, j))
    if not enemy_pos_list:
        return None, None
    enemy_dist = [abs(i - pos[0]) + abs(j - pos[1]) for (i, j) in enemy_pos_list] ## L1 dist
    min_dist = min(enemy_dist)
    return enemy_pos_list[enemy_dist.index(min_dist)], min_dist

def move_dir_to_action(dir):
    if dir[0] == 1:
        if dir[1] == -1:
            return 5
        elif dir[1] == 0:
            return 2
        elif dir[1] == 1:
            return 6
    elif dir[0] == 0:
        if dir[1] == -1:
            return 1
        elif dir[1] == 0:
            return 19
        elif dir[1] == 1:
            return 3
    elif dir[0] == -1:
        if dir[1] == -1:
            return 8
        elif dir[1] == 0:
            return 4
        elif dir[1] == 1:
            return 7
    return 19

def find_in_message(screen, str):
    for i in range(22):
        line = screen[i]
        interp = ''
        for letter in line:
            interp += chr(letter)

            if str in interp:
                return True
    return False


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class Floor():
    def __init__(self):
        self.room = []
        self.cur_room = None
        self.corridor = []
        self.interesting = []
        
        self.current_place = None
        self.parent = [[None for _ in range(79)] for _ in range(21)]
        self.children = [[None for _ in range(79)] for _ in range(21)]
        self.visited = [[False for _ in range(79)] for _ in range(21)]
        self.occ_map = [[1/(21*79) for _ in range(79)] for _ in range(21)]
        self.search_number = [[0 for _ in range(79)] for _ in range(21)]
        self.walls = [[False for _ in range(79)] for _ in range(21)]
        self.stuck_boulders = []
        self.search_completed = False
        self.goal = None
        self.search_pos = None
        self.in_room = True

        self.last_pos = None
        self.frozen_cnt = 0
        #self.dxy = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.dxy = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
        self.opposite = [3, 4, 1, 2, 7, 8, 5, 6]
        
    def search(self, env, obs):
        x, y = obs['blstats'][:2]
        screen = obs['tty_chars']

        action = None
            
        if self.last_pos == (x, y):
            self.frozen_cnt += 1
            if self.frozen_cnt > 10:
                self.frozen_cnt = 0
                action = env.action_space.sample()
                
                return [action]
        else:
            self.frozen_cnt = 0
        
        self.last_pos = (x, y)
        
        pre_map = self.preprocess_map(obs)
        self.calculate_occupancy_map(obs, pre_map, (x,y))

        self.visited[y][x] = True

        if self.children[y][x] == None:
            py, px = None, None
            if self.parent[y][x] != None:
                dxy_i = self.parent[y][x] - 1 #부모 노드가 있다면 부모 노드의 위치 기억
                py = y + self.dxy[dxy_i][0]
                px = x + self.dxy[dxy_i][1]
            
            children_list = []
            for i in range(8):
                ny = y + self.dxy[i][0]
                nx = x + self.dxy[i][1]

                #범위 안이면서, 이동 가능하면서, 부모 노드가 아니고, 방문하지 않았다면 자식 노드
                is_child = (0 <= ny < 21 and 0 <= nx < 79) and pre_map[ny][nx] and (ny, nx) != (py, px) and not self.visited[ny][nx]

                #재탐색할때 지웠던거 다시 채우기
                is_previous_child = ((0 <= ny < 21 and 0 <= nx < 79) and 
                                    pre_map[ny][nx] and    
                                    (ny, nx) != (py, px) and 
                                    self.parent[ny][nx] != None and 
                                    (y == ny + self.dxy[self.parent[ny][nx] - 1][0] and x == nx + self.dxy[self.parent[ny][nx] - 1][1]))

                is_child = is_child or is_previous_child

                children_list.append(is_child)

                #자식 노드에 부모 노드에 대한 정보 추가
                if is_child:
                    self.parent[ny][nx] = self.opposite[i]
            
            #해당 좌표에 자식 리스트 추가
            self.children[y][x] = children_list

        prior = 0
        if self.interesting:
            dist_nearby = 100
            nearby = None
            for dest in self.interesting:
                dist = manhattan_distance(dest, (y,x))
                if dist < dist_nearby:
                    nearby = dest
                    dist_nearby = dist
            dy = 0 if nearby[0] == y else (nearby[0] - y) // abs(nearby[0] - y)
            dx = 0 if nearby[1] == x else (nearby[1] - x) // abs(nearby[1] - x)
            for i in range(8):
                if self.dxy[i] == (dy, dx):
                    prior = i
        elif self.goal is not None:
            self.search_completed = True
            return [19]

        for i in range(8):
            ny = y + self.dxy[(i + prior) % 8][0]
            nx = x + self.dxy[(i + prior) % 8][1]
            if (0 <= ny < 21 and 0 <= nx < 79) and self.children[y][x][(i + prior) % 8] and not self.visited[ny][nx]:
                action = (i + prior) % 8 + 1
                return [action]
            
        # 막다른 길에서는 search로 길 찾기 트라이
        wall_check = 4
        for i in range(4):
            ny = y + self.dxy[i][0]
            nx = x + self.dxy[i][1]
            if (0 <= ny < 21 and 0 <= nx < 79) and pre_map[ny][nx] is True:
                wall_check -= 1
        if wall_check >= 3:
            #대각선에 문이 있으면 찾을 때까지 찾는다
            door_or_wall = [ord('|'), ord('-')]
            for i in range(4, 8):   
                ny = y + self.dxy[i][0]
                nx = x + self.dxy[i][1]
                if (0 <= ny < 21 and 0 <= nx < 79) and obs['chars'][ny][nx] in door_or_wall and obs['colors'][ny][nx] != 7:
                    action = 22
            #아니면 5번만 (확률 53%)
            if self.search_number[y][x] < 5:
                action = 22
                self.search_number[y][x] += 1
            else:
                action = self.parent[y][x]
        else:
            action = self.parent[y][x]

        #더 이상 탐색할 곳이 없다면 계단 올라가기
        if action == None:
            self.search_completed = True
            action = 19

        return [action]
    
    def preprocess_map(self, obs):
        pre = []
        self.interesting = []
        available = [ord('.'), ord('#')]
        unavailable = [ord(' '), ord('@'), ord('^'), ord('"')]
        door_or_wall = [ord('|'), ord('-')]
        
        chars = obs['chars']
        colors = obs['colors']
        cx, cy = obs['blstats'][:2]

        for y in range(21):
            pre_line = []
            for x in range(79):
                char = chars[y][x]
                color = colors[y][x]
                
                pre_char = True
                if char in unavailable:
                    pre_char = False
                elif char in door_or_wall and color == GRAY:
                    pre_char = False
                    self.walls[y][x] = True
                elif char == ord('#') and color == CYAN:
                    pre_char = False
                elif char == ord('>'):
                    if self.goal == None:
                        printf('found goal')
                    self.goal = (x, y)
                    
                elif char == ord('%'):
                    if obs['blstats'][21] > 0:
                        self.interesting.append((y, x))
                elif char == ord('`') and (y, x) in self.stuck_boulders:
                    pre_char = False

                if (char in available) or (char in door_or_wall and color != GRAY):                 # Check unlightened area as interesting
                    for i in (0,2,1,3):
                        ny, nx = y + self.dxy[i][0], x + self.dxy[i][1]
                        if not is_valid(ny, nx): continue
                        if chars[ny][nx] == ord(' '):
                            if not self.visited[y][x]:
                                self.interesting.append((y, x))
                                break
                pre_line.append(pre_char)
            pre.append(pre_line)

        # Check invalid diagonal moves
        for i in range(4,8):
            left = i - 4
            right = (i - 3) % 4
            if not is_valid(cy + self.dxy[i][0], cx + self.dxy[i][1]):
                continue

            if not pre[cy + self.dxy[left][0]][cx + self.dxy[left][1]] and not pre[cy + self.dxy[right][0]][cx + self.dxy[right][1]]:
                pre[cy + self.dxy[i][0]][cx + self.dxy[i][1]] = False            

        for i in range(8):
            ny, nx = cy + self.dxy[i][0], cx + self.dxy[i][1]
            if not is_valid(ny, nx):continue
            if chars[ny][nx] == ord('#'):
                self.in_room = False

        if self.in_room:
            borders = []

            for i in range(4):
                y, x = cy, cx
                while is_valid(y, x):
                    if not pre[y][x]: break
                    ny, nx = y + self.dxy[i][0], x + self.dxy[i][1]
                    if self.walls[ny][nx]:
                        break
                    elif self.walls[ny + self.dxy[(i + 1) % 4][0]][nx + self.dxy[(i + 1) % 4][1]] \
                        and self.walls[ny + self.dxy[(i - 1) % 4][0]][nx + self.dxy[(i - 1) % 4][1]]:
                        break
                    else:
                        y, x = ny, nx
                if pre[y][x]:
                    borders.append((ny, nx))

            if len(borders) == 4:
                c1 = (borders[0][0], borders[3][1])
                c2 = (borders[2][0], borders[1][1])
                room_corners = [c1, c2]
                cur_room = None
                for i in range(len(self.room)):
                    if self.room[i] == room_corners:
                        cur_room = i
                        self.cur_room = cur_room
                if cur_room == None:
                    self.cur_room = len(self.room)
                    self.room.append(room_corners)
                    printf(("Entered to room %d " + str(room_corners)) % self.cur_room)

        return pre
    
    def stuck(self, pos, action, obs):
        y, x = pos[0], pos[1]
        dy, dx = self.dxy[action - 1][0], self.dxy[action - 1][1]
        target = obs['chars'][y + dy][x + dx]
        if target == ord('`'):
            printf("stuck by boulder")
            self.stuck_boulders.append((y + dy, x + dx))

            for i in range(8):
                if (dy, dx) == self.dxy[i]:
                    self.children[y][x][i] = False
                    self.parent[y + dy][x + dx] = None
                    break

    def navigate(self, obs, now_pos, target_pos):
        pre_map = self.preprocess_map(obs)
        # use A*
        # use 0.5 * L1 length for heuristic function
        # return a list of actions to go from now_pos to target_pos
        pq = PriorityQueue()
        visited = [[False for _ in range(79)] for _ in range(21)]
        parent = [[None for _ in range(79)] for _ in range(21)]
        distance = [[INF for _ in range(79)] for _ in range(21)]
        pq.put((0.5 * manhattan_distance(now_pos, target_pos), now_pos))
        distance[now_pos[1]][now_pos[0]] = 0
        while not pq.empty():
            _, (x, y) = pq.get_nowait()
            if visited[y][x]:
                continue
            visited[y][x] = True
            if (x, y) == target_pos:
                rx = x
                ry = y
                action_list = []
                while not (rx == now_pos[0] and ry == now_pos[1]):
                    action_list.append(parent[ry][rx])
                    return_dir = self.opposite[parent[ry][rx] - 1]
                    rx = rx + self.dxy[return_dir - 1][1]
                    ry = ry + self.dxy[return_dir - 1][0]
                return action_list
            for i in range(8):
                ny = y + self.dxy[i][0]
                nx = x + self.dxy[i][1]
                if (0 <= ny < 21 and 0 <= nx < 79) and pre_map[ny][nx] and not visited[ny][nx]:
                    parent[ny][nx] = i + 1
                    distance[ny][nx] = distance[y][x] + 1
                    pq.put((0.5 * manhattan_distance((nx, ny), target_pos) + distance[ny][nx], (nx, ny)))
        #not reachable
        return [19]
    
    def calculate_occupancy_map(self, obs, pre_map, pos):
        diff_factor = 0.5
        prob_culled = 0
        if self.visited[pos[1]][pos[0]] is False:
            prob_culled += self.occ_map[pos[1]][pos[0]]
            self.occ_map[pos[1]][pos[0]] = 0
            self.visited[pos[1]][pos[0]] = True
        if (prob_culled > 0):
            for i in range(21):
                for j in range(79):
                    if self.visited[i][j] is False:
                        self.occ_map[i][j] /= 1 - prob_culled
            cur_occ_map = copy.deepcopy(self.occ_map)
            for i in range(21):
                for j in range(79):
                    dirs = [(-1,0),(0,1),(1,0),(0,-1)]
                    neighbour_probs = 0
                    for dir in dirs:
                        try:
                            neighbour_probs += cur_occ_map[i + dir[0]][j + dir[1]]
                        except IndexError:
                            continue
                    self.occ_map[i][j] = (1- diff_factor) * cur_occ_map[i][j] + diff_factor * neighbour_probs / 4
                    if self.visited[i][j]:
                        self.occ_map[i][j] = 0
    
    def calculate_search_position(self):
        pq = PriorityQueue()
        for i in range(21):
            for j in range(79):
                if self.visited[i][j]:
                    dirs = [(-1,0),(0,1),(1,0),(0,-1)]
                    neighbour_probs = 0
                    for dir in dirs:
                        try:
                            neighbour_probs += self.occ_map[i + dir[0]][j + dir[1]]
                        except IndexError:
                            continue
                    pq.put((-neighbour_probs, (j,i)))
        return pq
                
class CombatAgent():
    def __init__(self) -> None:
        self.battle = False
        self.move_memory = []
        self.dxy = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.battle_start_pos = None
        
    def start_battle(self, env, obs, floor, enemy_pos):
        x, y = obs['blstats'][:2]
        self.battle_start_pos = (x, y)
        self.battle = True
        return self.get_action(env, obs, floor, enemy_pos)
        
    def end_battle(self, obs, pos):
        if self.battle_start_pos is not None and self.battle_start_pos == pos:
            self.battle_start_pos = None
            self.battle = False 
        
    def get_action(self, env, obs, floor, enemy_pos):
        x, y = obs['blstats'][:2]
        pos = (x,y)
        if enemy_pos is not None:
            ##battle!
            #algorithm for now: fight only when enemy is in 8 direction.
            #else wait for coming.
            return [move_dir_to_action((enemy_pos[0] - x, enemy_pos[1] - y))]
        else:  ##get back to original position
            if not self.move_memory:
                self.move_memory += floor.navigate(obs, (x,y), self.battle_start_pos)
            return [self.move_memory.pop()]
 
class Agent(ExampleAgent):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)
        self.floor = {}
        self.combat_agent = CombatAgent()
        self.action_list = []
        self.last_action = None
        self.orange_count = 5
        self.last_time = 0
        self.wait_cnt = 0
        self.moved = False

        self.env = gym.make(
                FLAGS.env,
                savedir=FLAGS.savedir,
                max_episode_steps=FLAGS.max_steps,
                allow_all_yn_questions=True,
                allow_all_modes=True,
            )
        
    def reset(self):
        self.floor = {}
        self.combat_agent = CombatAgent()
        self.action_list = []
        self.last_action = None
        self.orange_count = 5

    # get_action without error
    def get_action(self, env, obs):
        action = None

        action = self.action_select(env, obs)

        self.moved = 0 < action <= 8
        return action

    def action_select(self, env, obs):
        time = obs['blstats'][20]
        if time == 1 and self.last_time != 1:
            self.reset()
        self.last_time = time
        x, y = obs['blstats'][:2]
        screen = obs['tty_chars']
        original_map = obs['chars']
        if obs['blstats'][12] in self.floor:
            current_floor = self.floor[obs['blstats'][12]]
        else:
            current_floor = Floor()
            self.floor[obs['blstats'][12]] = current_floor
            
        if find_in_message(screen, "hidden"):
            current_floor.children[y][x] = None
            current_floor.search_completed = False
            current_floor.search_pos = None
        closest_enemy_pos, min_dist = find_closest_enemy(obs, (x, y))
        if self.combat_agent.battle and closest_enemy_pos is None:
            self.combat_agent.end_battle(obs, (x, y))
        
        if self.wait_cnt > 0:
            sleep(1)
            self.wait_cnt -= 1

        if current_floor.last_pos == (x, y) and self.moved:                       # move failed because of boulder or something
            current_floor.stuck((y, x), self.last_action, obs)

        #actions
        if find_in_message(screen, 'More'):
            self.action_list.append(0)
        elif find_in_message(screen, '[yn]'):
            self.action_list.append(8)
        elif find_in_message(screen, 'locked'):
            self.action_list.append(20)
        elif find_in_message(screen, 'direction?'):
            original_map = obs['chars'] 
            
            north = original_map[y-1][x]
            east = original_map[y][x+1]
            south = original_map[y+1][x]
            west = original_map[y][x-1]
            northeast = original_map[y-1][x+1]
            southeast = original_map[y+1][x+1]
            southwest = original_map[y+1][x-1]
            northwest = original_map[y-1][x-1]

            if north == ord('+'):
                self.action_list.append(1)
            elif east == ord('+'):
                self.action_list.append(2)
            elif south == ord('+'):
                self.action_list.append(3)
            elif west == ord('+'):
                self.action_list.append(4)
            if northeast == ord('+'):
                self.action_list.append(5)
            elif southeast == ord('+'):
                self.action_list.append(6)
            elif southwest == ord('+'):
                self.action_list.append(7)
            elif northwest == ord('+'):
                self.action_list.append(8)
        elif find_in_message(screen, 'diagonally'):
            action_list = []
            original_map = obs['chars']
            
            north = original_map[y-1][x]
            south = original_map[y+1][x]

            rev = False
            if self.last_action == 5:
                action_list = [2,1]
                rev = north in [ord('-'), ord('|'), ord("`")]
            elif self.last_action == 6:
                action_list = [2,3]
                rev = south in [ord('-'), ord('|'), ord("`")]
            elif self.last_action == 7:
                action_list = [4,3]
                rev = south in [ord('-'), ord('|'), ord("`")]
            elif self.last_action == 8:
                action_list = [4,1]
                rev = north in [ord('-'), ord('|'), ord("`")]
            if rev: action_list.reverse()
            self.action_list += action_list
        elif find_in_message(screen, 'What'):
            dialog = ''
            copy = False
            for i in range(22):
                line = screen[i]
                interp = ''
                for letter in line:
                    interp += chr(letter)
                    if letter == ord('['): copy = True
                    if letter == ord(']'): copy = False
                    if copy : dialog += chr(letter)
            
            choices = dialog.split('or')[0]
            choices = choices[1:-1]
            if self.orange_count > 0:
                self.action_list.append(4)
                self.orange_count -= 1
            else:
                self.action_list.append(0)
            
        elif find_in_message(screen, 'Call'):
            self.action_list.append(0)
        elif not self.combat_agent.battle and min_dist is not None and min_dist < 4:# get into battle
            self.action_list += self.combat_agent.start_battle(env, obs, current_floor, closest_enemy_pos)
        elif self.combat_agent.battle:                                              # keep battle
            self.action_list += self.combat_agent.get_action(env, obs, current_floor, closest_enemy_pos)

        elif self.action_list:                                                      # solve continuations
            self.last_action = self.action_list.pop()
            return self.last_action
        elif obs['blstats'][21] > 1 and self.orange_count > 0:                     # eat orange
            self.action_list += [4, 21]
            self.orange_count -= 1
        elif obs['blstats'][21] > 0 and find_in_message(screen, "corpse") and find_in_message(screen, "here"):
            self.action_list += [8, 21]

        elif (current_floor.search_completed == False):                             # basic searching
            search_ans = current_floor.search(env, obs)
            if search_ans is None:
                print("search_ans ERROR!")
            self.action_list += search_ans
        else:
            if current_floor.goal is not None:
                if (x, y) != current_floor.goal:
                    navigate_ans = current_floor.navigate(obs, (x, y), current_floor.goal)
                    if navigate_ans is None:
                        print("navigagte_ans ERROR!")
                    self.action_list += navigate_ans
                elif (x, y) == current_floor.goal:
                    self.action_list.append(18) # go to next floor
            else: #current_floor.goal is None
                if current_floor.search_pos is None:
                    for j in range(21):
                        for i in range(79):
                            log.write("%.3f" % current_floor.occ_map[j][i])
                        log.write("\n")
                    current_floor.search_pos = current_floor.calculate_search_position()
                if (x, y) != current_floor.search_pos.queue[0][1]:
                    self.action_list += current_floor.navigate(obs, (x,y), current_floor.search_pos.queue[0][1])
                else:
                    self.action_list += [22] * 20
                    current_floor.search_pos.get_nowait()
                self.last_action = self.action_list.pop()
                return self.last_action
        self.last_action = self.action_list.pop()
        return self.last_action