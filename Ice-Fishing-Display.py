# Modules
import pygame
from pygame.locals import *
import os
import numpy as np
from random import randint, choice
from time import time, sleep
from collections import Counter
from statistics import mean
import neat
import pickle
import gzip
from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Process, Array, Value

pygame.font.init()
plt.rcParams['toolbar'] = 'None'
f = open('state.txt','w')
f.write('NotDone')
f.close()

def draw_graphs(state, action, best_fitnesses, display_max):
    def draw_neuron_output(i):
        ax1.clear()
        # Draw tanh
        ax1.plot(in_array, tanh_array, c='snow')
        ax1.axhline(y=0, color='black')
        ax1.axvline(x=0, color='black')
        # Draw line
        line_y = action[0]
        ax1.axhline(y=line_y, color=['lightgreen','lightcoral'][(action[0]<0)])

    def draw_fitness(i):
        trim_best_fitnesses = []
        new_best_fitnesses = best_fitnesses[:]
        for fitness in new_best_fitnesses:
            if fitness != 0:
                trim_best_fitnesses.append(fitness)
        ax2.clear()
        in_array = np.arange(len(trim_best_fitnesses))
        ax2.plot(in_array, trim_best_fitnesses, color='cyan')
        ax2.set_xlabel('Generation',c='snow',fontweight='bold')
        ax2.set_ylabel('Best Fitness',c='snow',fontweight='bold')

    def draw_network(i):
        ax3.clear()
        ax3.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), c=bg_col, alpha=0)
        input_node_coords = []
        input_node_alphas = []
        for i in list(reversed(state[:])):
            if 0.1+np.cbrt(abs(i)/1.12) <= 1 :
                input_node_alphas.append(0.1+np.cbrt(abs(i)/1.12))
            else:
                input_node_alphas.append(1)
        output_node_coords = [0.9, 0.55]
        output_node_alpha = abs(action[0])
        for y in range(0, 8):
            input_node_coords.append([0.5, 0.05 + y * 0.15])

        for i, coords in enumerate(input_node_coords):
            ax3.text(0.35, coords[1], node_texts[i], fontsize=12, c='snow', horizontalalignment='center',
                     fontweight='bold')
            ax3.plot([coords[0], output_node_coords[0]], [coords[1], output_node_coords[1]], lw=2, zorder=1,
                     color=node_cols[i], alpha=input_node_alphas[i])
            ax3.scatter(coords[0], coords[1], s=node_size, fc=node_bg, ec=node_edge, lw=node_w, zorder=2)
            ax3.scatter(coords[0], coords[1], s=node_size, fc=node_cols[i], ec=None, lw=node_w, zorder=3,
                        alpha=input_node_alphas[i])
        ax3.scatter(output_node_coords[0], output_node_coords[1], s=node_size, fc=node_bg, ec='snow', lw=node_w,
                    zorder=2)
        ax3.scatter(output_node_coords[0], output_node_coords[1], s=node_size, fc=['lightgreen','red'][(action[0]<0)], ec=None, lw=node_w,
                    zorder=3, alpha=np.cbrt(output_node_alpha))

    # Init
    display_max = display_max[0]
    bg_col = '#404040'
    if display_max:
        # Neuron Output
        fig1 = plt.figure('Neuron Output')
        fig1.patch.set_facecolor(bg_col)
        fig1.subplots_adjust(bottom=0, top=1, left=0, right=1)
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.set_facecolor(bg_col)

        w = 3
        in_array = np.linspace(-w, w, 50)
        tanh_array = np.tanh(in_array)

        ani1 = animation.FuncAnimation(fig1, draw_neuron_output, interval=100)
        # Network
        node_cols = ['white', 'hotpink', 'yellow', 'yellow', 'saddlebrown', 'saddlebrown', 'cyan', 'cyan']
        node_texts = ['Hook-State', 'Hook-y', 'Fish-1', 'Fish-2', 'Obstacle-1', 'Obstacle-2', 'Jellyfish-1',
                      'Jellyfish-2']
        node_texts = list(reversed(node_texts))
        node_cols = list(reversed(node_cols))
        node_bg = '#111111'
        node_edge = '#111111'
        node_size = 1300
        node_w = 5
        fig3 = plt.figure('Network')
        fig3.subplots_adjust(bottom=0, top=1, left=0, right=1)
        ax3 = fig3.add_subplot(1, 1, 1)
        ax3.set_facecolor(bg_col)

        ani3 = animation.FuncAnimation(fig3, draw_network, interval=50)
    else:
        # Average fitness
        fig2 = plt.figure('Best fitness per generation')
        fig2.patch.set_facecolor(bg_col)
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.set_facecolor('#505050')

        ani2 = animation.FuncAnimation(fig2, draw_fitness, interval=50)

    plt.show()

def main_code():
    # Variables
    win_width = 752
    win_height = 460
    new_win_width = int(win_width*1.5)
    new_win_height = int(win_height*1.5)
    gen = 0
    display, load = True, True
    ''''
    if input("Load?: ") == 'y':
        

    else:
        log_dir = 'Logs/{}'.format(int(time()))
        os.mkdir(os.path.abspath(log_dir))
        display, load = False, False
    '''

    # Images
    bg = pygame.image.load(os.path.join("Images", "bg.png"))
    font = pygame.font.SysFont('Calibri', 20, True, False)
    small_font = pygame.font.SysFont('Calibri', 17, True, False)
    hook_imgs = [pygame.image.load(os.path.join("Images", "Hook.png")),pygame.image.load(os.path.join("Images", "Fish.png")),pygame.image.load(os.path.join("Images", "HookShocked.png"))]
    fish_imgs = [pygame.image.load(os.path.join("Images", "Fish.png"))]
    jellyfish_imgs = [pygame.image.load(os.path.join("Images", "Jellyfish.png"))]
    obstacle_imgs = [pygame.image.load(os.path.join("Images", "Boot.png")),pygame.image.load(os.path.join("Images", "Barrel.png"))]

    # Classes
    class hook_class(object):
        sprites = hook_imgs
        def __init__(self, y):
            self.x = 362
            self.y = y
            self.w = 20
            self.h = 26
            self.sprite = self.sprites[0]
            self.hooked = False
            self.obstacle_invinticks = 0
            self.jellyfish_invinticks = 0
            self.score = 0
            self.phase_score = 0
            self.zapped = False
            self.zapped_num = 0
            self.jellyfish_dodge_num = 0

        def draw(self, win):
            pygame.draw.line(win, (0, 0, 0), (self.x, 19), (self.x, self.y), 1)
            pygame.draw.circle(win, (0, 255, 0), (self.x, self.y), 3)
            if self.jellyfish_invinticks > 0 :
                self.sprite = self.sprites[2] # Shocked
                self.w = 48
                self.h = 50
                win.blit(self.sprite, (self.x - self.w / 2, self.y- self.h/2 - 6))
            elif self.hooked:
                self.sprite = self.sprites[1] # Caught
                self.w = 50
                self.h = 134
                win.blit(pygame.transform.rotate(self.sprite, 90), (self.x - self.w/2 - 7, self.y - 3))
            else:
                self.sprite = self.sprites[0] # Default
                self.w = 20
                self.h = 26
                win.blit(self.sprite, (self.x - self.w / 2, self.y))

    class fish_class(object):
        sprites = fish_imgs
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.w = 134
            self.h = 51
            self.caught = False
            self.vel = 5

        def draw(self, win):
            sprite = pygame.image.load(os.path.join("Images", "Fish.png"))
            if not self.caught:
                pygame.draw.circle(win, (255, 0, 0), (self.x, self.y), 3)
            if self.caught:
                if display_max:
                    sprite.fill((255, 255, 255, 0), None, pygame.BLEND_RGBA_MULT)
                else:
                    sprite.fill((255, 255, 255, 64), None, pygame.BLEND_RGBA_MULT)
            win.blit(sprite, (self.x - self.w, self.y - self.h / 2))

    class obstacle_class(object):
        sprites = obstacle_imgs
        def __init__(self,x,y,type):
            self.type = type
            self.x = x
            self.y = y
            if self.type == "boot":
                self.w = 80
                self.h = 97
                self.sprite = self.sprites[0]
            elif self.type == "barrel":
                self.w = 108
                self.h = 130
                self.sprite = self.sprites[1]
            self.vel = 5

        def draw(self, win):
            pygame.draw.circle(win, (0, 0, 255), (self.x, self.y), 3)
            win.blit(self.sprite, (self.x - self.w, self.y - self.h / 2))

        def get_mask(self):
            return pygame.mask.from_surface(self.sprite)

    class jellyfish_class(object):
        sprites = jellyfish_imgs
        def __init__(self,x,y):
            self.x = x
            self.y = y
            self.w = 98
            self.h = 84
            self.sprite = self.sprites[0]
            self.vel = 5

        def draw(self, win):
            pygame.draw.circle(win, (0, 0, 255), (self.x, self.y), 3)
            win.blit(self.sprite, (self.x - self.w, self.y - self.h / 2))

    def redraw_game_window(win, hooks, fishes, obstacles, jellyfishes, scores, fitnesses, num_dead, tick, gen):
        win.blit(bg, (0, 0))
        for hook in hooks:
            hook.draw(win)
        for fish in fishes:
            fish.draw(win)
        for jellyfish in jellyfishes:
            jellyfish.draw(win)
        for obstacle in obstacles:
            obstacle.draw(win)
        if not display_max:
            text_gen = font.render('Generation: {}'.format(gen), 1, (0,0,0))
            text_alive = small_font.render('Alive: {}'.format(20-num_dead),1, (0,0,0))
            text_best_score = small_font.render('Best Score: {:.2f}'.format(scores[0]), 1, (0, 0, 0))
            text_average_score = small_font.render('Average Score: {:.2f}'.format(sum(scores)/len(scores)), 1, (0, 0, 0))
            text_best_fitness = small_font.render('Best Fitness: {:.2f}'.format(fitnesses[0]), 1, (0, 0, 0))
            text_average_fitness = small_font.render('Average Fitness: {:.2f}'.format(sum(fitnesses) / len(fitnesses)), 1, (0, 0, 0))
            text_tick = small_font.render('Tick: {}'.format(tick), 1, (0, 0, 0))

            win.blit(text_gen, (5, 5))
            spacing = 17
            win.blit(text_alive, (5,5+spacing*1))
            win.blit(text_best_score, (5, 5+spacing*2))
            win.blit(text_best_fitness, (5, 5+spacing*3))
            win.blit(text_average_score, (5, 5+spacing*4))
            win.blit(text_average_fitness, (5, 5+spacing*5))
            win.blit(text_tick, (5, 5+spacing*6))
        else:
            text_fittest_individual = font.render('Fittest Individual', 1, (0, 0, 0))
            text_gen = font.render('Generation: {}'.format(33), 1, (0, 0, 0))
            win.blit(text_fittest_individual, (5, 5))
            win.blit(text_gen, (5, 25))

        text_total_score = font.render(str(sum(scores)), 1, (0, 0, 0))
        win.blit(text_total_score, (269, 85))
        pygame.display.update()

    # Main Game
    def play_game(genomes, config):
        # Window
        if display:
            new_win_width = int(win_width*1.5)
            new_win_height = int(win_height*1.5)
            pygame.init()
            resize_win = pygame.display.set_mode((win_width, win_height),HWSURFACE|DOUBLEBUF|RESIZABLE)
            pygame.display.set_caption('Ice Fishing')
            win = resize_win.copy()
            resize_win = pygame.display.set_mode((new_win_width, new_win_height), HWSURFACE | DOUBLEBUF | RESIZABLE)
            # Sounds
            sound_catch = pygame.mixer.Sound(os.path.join("Sounds", "Catch.wav"))
            sound_deliver = pygame.mixer.Sound(os.path.join("Sounds", "Deliver.wav"))
            sound_shock = pygame.mixer.Sound(os.path.join("Sounds", "Shock.wav"))
            sound_obstacle = pygame.mixer.Sound(os.path.join("Sounds", "Obstacle.wav"))

        file_list = ["Fittest-Individual"]

        # Loops
        gen = 0
        for file in file_list:
            # Initialise
            hooks = []
            hook_nets = []
            hook_genomes = []
            fishes = []
            obstacles = []
            jellyfishes = []
            scores = [0]*20
            fitnesses = [0]*20
            num_dead = 0
            tick = 0
            max_ticks = 100*60*5 
            print('Gen: {}'.format(gen))

            # Fitness Stats
            inwater_tick_reward = 0
            outwater_tick_reward = -0.1
            hook_reward = 20
            deliver_reward = 80
            hit_obstacle_reward = -30
            hit_obstacle_hooked_reward = -20
            hit_jellyfish_reward = -100
            hit_jellyfish_hooked_reward = -20
            dodge_jellyfish_reward = 25
            finish_phase_reward = 0
            fail_phase_reward = 0

            # Neat
            if load:
                with gzip.open(file) as f:
                    genomes, config = pickle.load(f)
                if display :
                    max_ticks = max_ticks

            max_fitness = -100000
            for genome_id, genome in genomes:
                if genome.fitness > max_fitness:
                    max_fitness = genome.fitness

            for genome_id, genome in genomes:
                if (load and display_max and genome.fitness == max_fitness) or (load and not display_max) or (not load) :
                    net = neat.nn.FeedForwardNetwork.create(genome, config)
                    genome.fitness = 0
                    hooks.append(hook_class(250))
                    hook_nets.append(net)
                    hook_genomes.append(genome)

            # Main Loop
            run = True
            print("Displaying Fittest Individual")
            while run and ((not display_max and tick < max_ticks) or display_max):
                # Check
                '''
                if display_max:
                    f = open('state.txt','r').read()
                    if f == 'Done':
                        run = False
                '''
                # Tick
                if len(hooks) == 0:
                    break
                tick += 1

                # Game Phase
                if display_max:
                    spawn_fish, spawn_obstacle, spawn_jellyfish = True, True, True
                    fish_rate, obstacle_rate, jellyfish_rate = 120, 360, 360
                    '''
                    if tick == 1 : # Phase 1
                        #print("Phase 1")
                        spawn_fish, spawn_obstacle, spawn_jellyfish = False, False, False
                        fish_rate, obstacle_rate, jellyfish_rate = 0, 0, 0
                    elif tick == int(max_ticks / 5): # Phase 2
                        #print("Phase 2")
                        spawn_fish, spawn_obstacle, spawn_jellyfish = True, False, False
                        fish_rate, obstacle_rate, jellyfish_rate = 80, 480, 0
                    elif tick == 2*int(max_ticks / 5): # Phase 3
                        #print("Phase 3")
                        spawn_fish, spawn_obstacle, spawn_jellyfish = True, True, False
                        fish_rate, obstacle_rate, jellyfish_rate = 100, 360, 0
                    elif tick == 3*int(max_ticks / 5): # Phase 4
                        spawn_fish, spawn_obstacle, spawn_jellyfish = True, True, True
                        fish_rate, obstacle_rate, jellyfish_rate = 120, 360, 480
                    elif tick == 4*int(max_ticks / 5): # Phase 5
                        spawn_fish, spawn_obstacle, spawn_jellyfish = True, True, True
                        fish_rate, obstacle_rate, jellyfish_rate = 120, 360, 480
                    '''
                else:
                    if tick == 1 : # Phase 1
                        #print("Phase 1")
                        spawn_fish, spawn_obstacle, spawn_jellyfish = True, True, True
                        fish_rate, obstacle_rate, jellyfish_rate = 240, 480, 480

                    elif tick == int(max_ticks/5) : # Phase 2
                        for hook_id, hook in enumerate(hooks):
                            if hook.phase_score >= (max_ticks/(5*fish_rate))*0.4:
                                #print("Hook {} past phase 1 with {} score".format(hook_id,hook.phase_score))
                                hook_genomes[hook_id].fitness += finish_phase_reward
                            else:
                                hook_genomes[hook_id].fitness += fail_phase_reward
                            hook.phase_score, hook.zapped_num, hook.jellyfish_dodge_num = 0, 0, 0

                        #print("Phase 2")
                        spawn_fish, spawn_obstacle, spawn_jellyfish = True, True, True
                        fish_rate, obstacle_rate, jellyfish_rate = 120, 360, 360

                    elif tick == 2*int(max_ticks/5) : # Phase 3
                        for hook_id, hook in enumerate(hooks):
                            if hook.phase_score >=3 and hook.jellyfish_dodge_num >= (max_ticks / (5 * jellyfish_rate)) * 0.5 and hook.zapped_num <= 1 :
                                #print("Hook {} past phase 2 with {} score, zapped {}".format(hook_id, hook.phase_score, hook.zapped_num))
                                hook_genomes[hook_id].fitness += finish_phase_reward
                            else:
                                hook_genomes[hook_id].fitness += fail_phase_reward
                            hook.phase_score, hook.zapped_num, hook.jellyfish_dodge_num = 0, 0, 0

                        #print("Phase 3")
                        spawn_fish, spawn_obstacle, spawn_jellyfish = True, True, True
                        fish_rate, obstacle_rate, jellyfish_rate = 120, 360, 360

                    elif tick == 3*int(max_ticks/5) : # Phase 4
                        for hook_id, hook in enumerate(hooks):
                            if hook.phase_score >= (max_ticks/(5*fish_rate))*0.4 :
                                #print("Hook {} past phase 3 with {} score".format(hook_id, hook.phase_score))
                                hook_genomes[hook_id].fitness += finish_phase_reward
                            else:
                                hook_genomes[hook_id].fitness += fail_phase_reward
                            hook.phase_score, hook.zapped_num, hook.jellyfish_dodge_num = 0, 0, 0

                        #print("Phase 4")
                        spawn_fish, spawn_obstacle, spawn_jellyfish = True, True, True
                        fish_rate, obstacle_rate, jellyfish_rate = 120, 240, 240

                    elif tick == max_ticks :
                        for hook_id, hook in enumerate(hooks):
                            if hook.phase_score >= 2*(max_ticks/(5*fish_rate))*0.4 :
                                #print("Hook {} past phase 4 with {} score".format(hook_id, hook.phase_score))
                                hook_genomes[hook_id].fitness += finish_phase_reward
                            else:
                                hook_genomes[hook_id].fitness += fail_phase_reward

                # Fish
                #  Spawn
                if spawn_fish and tick % fish_rate == 0:
                    fishes.append(fish_class(randint(-750,0), 300))
                    fish = fishes[-1]
                    fish.y = randint(round(150 + fish.h/2), round(win_height - fish.h/2))

                for fish in fishes:
                    #  Move
                    fish.x += fish.vel

                    #  Clear
                    if fish.x - fish.w > win_width:
                        fishes.pop(fishes.index(fish))

                # Obstacles
                #  Spawn
                if spawn_obstacle and tick % obstacle_rate == 0:
                    obstacles.append(obstacle_class(randint(-750,0), 300, choice(['boot','barrel'])))
                    obstacle = obstacles[-1]
                    obstacle.y = randint(round(150 + obstacle.h / 2), round(win_height - obstacle.h / 2))

                for obstacle in obstacles:
                    #  Move
                    obstacle.x += obstacle.vel

                    #  Clear
                    if obstacle.x - obstacle.w > win_width:
                        obstacles.pop(obstacles.index(obstacle))

                # Jellyfish
                #  Spawn
                if spawn_jellyfish and tick % jellyfish_rate == 0:
                    jellyfishes.append(jellyfish_class(randint(-750,0), 300))
                    jellyfish = jellyfishes[-1]
                    jellyfish.y = randint(round(150 + jellyfish.h / 2), round(win_height - jellyfish.h / 2))

                for jellyfish in jellyfishes:
                    #  Move
                    jellyfish.x += jellyfish.vel

                    #  Clear
                    if (jellyfish.x - jellyfish.w > win_width) :
                        jellyfishes.pop(jellyfishes.index(jellyfish))

                # Hooks
                for hook_id, hook in enumerate(hooks):
                    #  Catch Fish
                    for fish in fishes:
                        if (fish.x in range(hook.x - 10, hook.x + 10)) and (fish.y in range(hook.y - 12, hook.y + 12)) and not hook.hooked:
                                sound_catch.play()
                                fish.caught = True
                                hook.hooked = True
                                hook.phase_score += 0.1
                                hook_genomes[hook_id].fitness += hook_reward

                    # Hit Obstacle
                    for obstacle in obstacles:
                        if (hook.x in range(int(obstacle.x-obstacle.w), int(obstacle.x))) and (hook.y in range(int(obstacle.y-obstacle.h),int(obstacle.y+obstacle.h))) and (hook.obstacle_invinticks == 0):
                            hook_genomes[hook_id].fitness += hit_obstacle_reward
                            hook.obstacle_invinticks = int(obstacle.w / obstacle.vel) + 1
                            if hook.hooked:
                                sound_obstacle.play()
                                hook_genomes[hook_id].fitness += hit_obstacle_hooked_reward
                                hook.hooked = False

                    #  Deliver
                    if hook.y < 115 and hook.hooked:
                        sound_deliver.play()
                        hook.score += 1
                        hook.hooked = False
                        hook.phase_score += 1
                        hook_genomes[hook_id].fitness += deliver_reward

                    # Fitness
                    if 175 < hook.y < win_height - 26: # Give small amount of fitness if hook in water
                        hook_genomes[hook_id].fitness += inwater_tick_reward
                    else:
                        hook_genomes[hook_id].fitness += outwater_tick_reward


                    # State
                    state = []
                    fishes_coords = []
                    num_fish = 2
                    obstacles_coords = []
                    num_obstacles = 2
                    num_jellyfish = 2
                    jellyfishes_coords =  []


                    # 1 Hook status
                    state.append(int(hook.hooked))

                    # 2 Hook height
                    state.append((hook.y - 21) / (win_height - 21))

                    # 3 Fish distances
                    for fish in fishes:
                        if 0 < fish.x < hook.x + 11 and not fish.caught:
                            fishes_coords.append([fish.x,fish.y])
                    fishes_coords = sorted(fishes_coords, key=lambda i: i[0], reverse=True) # Sort by x
                    for i in range(0,num_fish):
                        try:
                            state.append((fishes_coords[i][1]-175) / (win_height-26-175)) # y value
                        except IndexError:
                            state.append(0) # Return zero if no fish availible

                    # 4 Obstacle distances
                    for obstacle in obstacles:
                        if 0 < obstacle.x < obstacle.x + obstacle.w/2:
                            obstacles_coords.append([obstacle.x,obstacle.y])
                    obstacles_coords = sorted(obstacles_coords, key=lambda i: i[0], reverse=True) # Sort by x
                    for i in range(0,num_obstacles):
                        try:
                            state.append((obstacles_coords[i][1]-175) / (win_height-26-175)) # y value
                        except IndexError:
                            state.append(0) # Return zero if no obstacles availible

                    # 5 Jellyfish distances
                    for jellyfish in jellyfishes:
                        if 0 < jellyfish.x < hook.x + jellyfish.w/2 :
                            jellyfishes_coords.append([jellyfish.x, jellyfish.y])
                    jellyfishes_coords = sorted(jellyfishes_coords, key=lambda i: i[0], reverse=True)  # Sort by x
                    for i in range(0, num_jellyfish):
                        try:
                            state.append(((jellyfishes_coords[i][1]-jellyfish.h/2) - 175 / (win_height - 26 - 175))/2 + 0.5)  # y value of top of jellyfish with weight
                        except IndexError:
                            state.append(0)  # Return zero if no jellyfish availible

                    # Determine Hook Action
                    action = hook_nets[hook_id].activate([state[i] for i in range(len(state))])
                    weight = 15
                    hook.y -= action[0] * weight # Perform Action

                    # Update globals
                    for i in range(0,len(state)):
                        global_state[i] = state[i]
                    global_action[0] = action[0]
                    #print('State: {} | Action: {}'.format(state,action))

                    # Hit Jellyfish
                    for jellyfish in jellyfishes:
                        if (hook.x in range(jellyfish.x-jellyfish.w,jellyfish.x)) and (jellyfish.y - jellyfish.h / 2) < hook.y and (hook.jellyfish_invinticks == 0) : # Hit
                            sound_shock.play()
                            hook.jellyfish_invinticks = int(jellyfish.w / jellyfish.vel) + 1
                            hook_genomes[hook_id].fitness += hit_jellyfish_reward
                            hook.zapped_num += 1
                            if hook.zapped == False:
                                num_dead += 1
                                hook.zapped = True
                            if hook.hooked:
                                hook_genomes[hook_id].fitness += hit_jellyfish_hooked_reward
                                hook.hooked = False
                        if hook.x in range(jellyfish.x-jellyfish.w, jellyfish.x-jellyfish.w+jellyfish.vel) and (jellyfish.y - jellyfish.h / 2) >= hook.y and (hook.jellyfish_invinticks == 0): # Dodge
                            hook_genomes[hook_id].fitness += dodge_jellyfish_reward
                            hook.jellyfish_dodge_num += 1

                    # Invin
                    if hook.obstacle_invinticks > 0 :
                        hook.obstacle_invinticks -= 1
                    if hook.jellyfish_invinticks > 0 :
                        hook.jellyfish_invinticks -= 1

                    # Position
                    if hook.y > win_height or hook.y < 21:  # Keep hook in limits
                        hook.y = win_height * (hook.y > win_height) + 21 * (hook.y < 21)
                    hook.y = round(hook.y)

                    # Texts
                    scores[hook_id] = hook.score
                    fitnesses[hook_id] = (hook_genomes[hook_id].fitness)

                scores.sort(reverse=True)
                fitnesses.sort(reverse=True)
                # Display
                if display:
                    # Tick
                    pygame.time.delay(1)

                    # Window
                    pygame.event.pump()
                    redraw_game_window(win, hooks, fishes, obstacles, jellyfishes, scores, fitnesses, num_dead, tick, gen)  # Draw sprites
                    resize_win.blit(pygame.transform.scale(win, (new_win_width, new_win_height)), (0, 0))  # Resize
                    pygame.display.flip()

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            run = False
                        elif event.type == VIDEORESIZE:
                            new_win_width, new_win_height = (event.dict['size'])
                            resize_win = pygame.display.set_mode((new_win_width, new_win_height), HWSURFACE | DOUBLEBUF | RESIZABLE)  # Resize


            # Post Loop

            # Check
            '''
            f = open('state.txt', 'w')
            if display_max:
                f.write('NotDone')
            else:
                f.write('Done')
            f.close()
            '''
            # Evaluate population
            '''
            fitnesses = []
            init_hook_genomes = genomes
            for genome in hook_genomes:
                fitnesses.append(genome.fitness)
            sorted_fitnesses = fitnesses
            sorted_fitnesses.sort(reverse=True)
            global_best_fitnesses[current_file_num] = sorted_fitnesses[0]
            '''

            # Saves
            if not load:
                # Fittest
                for genome in hook_genomes:
                    if genome.fitness >= config.fitness_threshold :
                        print('Saving Fittest')
                        with gzip.open('Fittest', 'w', compresslevel=5) as f:
                            data = (genomes, config)
                            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Logs
                init_threshold = config.fitness_threshold
                config.fitness_threshold = -1000000
                for genome in hook_genomes:
                    if genome.fitness >= config.fitness_threshold :
                        np.save('{}\\Fitnesses.npy'.format(log_dir), best_fitnesses)
                        with gzip.open('{}\\{}'.format(log_dir,gen), 'w', compresslevel=5) as f:
                            data = (genomes, config)
                            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                config.fitness_threshold = init_threshold
            gen += 1
        return

    # Neat
    def run(config_path):
        num_generations = 500
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        p = neat.Population(config) # Generate population from config file
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.run(play_game, num_generations)

    #if __name__ == "__main__":
    if load:
        play_game(0,0)
    else:
        local_dir = os.path.dirname(__file__) # Gives path to current directory
        config_path = os.path.join(local_dir, "config.txt")
        run(config_path)

if __name__ == '__main__':
    display_max = True # False to display evolution
    global_display_max = Array('i',[int(display_max)])
    global_state = Array('d', [0]*8)
    global_action = Array('d', [0])
    global_best_fitnesses = Array('d',[0]*34)
    graphs = Process(target=draw_graphs, args=(global_state, global_action, global_best_fitnesses,global_display_max,))
    graphs.start()
    main_code()
    graphs.join()
