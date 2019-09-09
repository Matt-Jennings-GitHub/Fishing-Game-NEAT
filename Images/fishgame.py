# Modules
import pygame
import os
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from random import randint, random
from time import time
from collections import Counter
from statistics import mean

# Main Game
def run_game(bot=False, display=False):
    # Window
    win_width = 752
    win_height = 460
    if display:
        pygame.init()
        win = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption('Fish Game')
        bg = pygame.image.load(os.path.join("Images", "bg.png"))
        font = pygame.font.SysFont('Calibri', 20, True, False)

    # Classes
    class hook(object):
        sprite = pygame.image.load(os.path.join("Images", "Hook.png"))

        def __init__(self, y):
            self.x = 362
            self.y = y
            self.w = 20
            self.h = 26

        def draw(self, win):
            pygame.draw.line(win, (0, 0, 0), (self.x, 19), (self.x, self.y), 1)
            pygame.draw.circle(win, (0, 255, 0), (self.x, self.y), 3)
            win.blit(self.sprite, (self.x - self.w / 2, self.y))

    class salmon(object):
        swim = [pygame.image.load(os.path.join("Images", "Fish{}.png".format(i))) for i in range(1, 3)]

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.w = 134
            self.h = 51
            self.frame = randint(0, 1)
            self.caught = False
            self.vel = 5

        def draw(self, win):
            if not self.caught:
                win.blit(self.swim[self.frame], (self.x - self.w, self.y - self.h / 2))
                self.frame = 1 - self.frame
            else:
                win.blit(pygame.transform.rotate(self.swim[self.frame], 90), (hook.x - fish.h / 2 - 7, hook.y - 3))
            pygame.draw.circle(win, (255, 0, 0), (self.x, self.y), 3)

    def redraw_game_window():
        win.blit(bg, (0, 0))
        hook.draw(win)
        for fish in fishes:
            fish.draw(win)
        text_pos = font.render('{},{}'.format(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]), 1, (0, 0, 0))
        text_score = font.render(str(score), 1, (0, 0, 0))
        win.blit(text_pos, (5, 5))
        win.blit(text_score, (269, 85))
        pygame.display.update()

    # Initialise
    score = 0
    game_data = []
    prev_state = []
    prev_y = 0

    hook = hook(200)
    fishes = []
    hooked = False


    tick = 0

    # Main Loop
    run = True
    while run:
        tick += 1
        if display:
            # Tick
            pygame.time.delay(10)

            # Window
            redraw_game_window()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

        # Determine Hook Action
        weight = 20
        if bot and prev_state: # Run predicted inputs
            prev_state = np.array(prev_state)
            X = prev_state.reshape(-1, len(prev_state),1)
            prediction = model.predict(X)
            action = list(prediction[0])
            print(action)
            # Perform Action
            if action.index(max(action)) == 0:
                hook.y += int(round(weight * action[0]))
            elif action.index(max(action)) == 1:
                hook.y -= int(round(weight * action[1]))


        elif not bot: # Read mouse inputs
            y_moved = pygame.mouse.get_pos()[1] - prev_y
            if y_moved > 0 :
                action = [( (weight - y_moved) / weight)*(y_moved < weight) + (y_moved >= weight), 0]
            elif y_moved < 0:
                action = [0, ( (weight - abs(y_moved)) / weight )*( abs(y_moved) < weight) + (abs(y_moved) >= weight)]
            else:
                action = [0, 0]
            hook.y = pygame.mouse.get_pos()[1]
            prev_y = pygame.mouse.get_pos()[1]

        else:
            action = [0, 0]

        if hook.y > win_height or hook.y < 21: # Keep hook in limits
            hook.y = win_height * (hook.y > win_height) + 21 * (hook.y < 21)

        # Fish
        #  Spawn
        if tick % 10 == 0 and randint(0,5) == 0 :
            fishes.append(salmon(0,randint(175,win_height-26)))

        # Events
        for fish in fishes:
            # Move
            if fish.caught:
                fish.vel = 0
                fish.y = hook.y

            fish.x += fish.vel
            # Catch
            if (fish.x in range(hook.x - 10, hook.x + 10)) and (fish.y in range(hook.y - 10, hook.y + 10)) and not hooked:
                    fish.caught = True
                    hooked = True
            # Deliver
            if fish.y < 115:
                score += 1
                hooked = False
                fishes.pop(fishes.index(fish))

            # Clear
            if fish.x - fish.w > win_width:
                fishes.pop(fishes.index(fish))


        # State
        state = []
        input_type = 'simple'
        fishes_coords = []
        num_fish = 4

        if input_type == 'simple':
            for fish in fishes:
                if not fish.caught:
                    fishes_coords.append([fish.x,fish.y])
            fishes_coords = sorted(fishes_coords, key=lambda i: i[0], reverse=True) # Sort by x
            for i in range(0,num_fish):
                try:
                    state.append((fishes_coords[i][1]-175) / (win_height-26-175))
                except IndexError:
                    state.append(0)

        elif input_type == 'channel':
            num_channels = int((win_height-26- 175) / 20) + 1
            for channel in range(0,num_channels) :
                channel_y = int(175 + ((win_height-26- 175) / (num_channels+1)) * (channel + 1))
                maxfish_x = 0
                for fish in fishes:
                    maxfish_x = 0
                    if fish.y in range(channel_y - 10, channel_y + 10):
                        if fish.x < hook.x and fish.x > maxfish_x :
                            maxfish_x = fish.x
                state.append(maxfish_x / hook.x)


        state.append(int(hooked))
        state.append((hook.y-21) / (win_height-21))

        if list(prev_state):
            game_data.append([prev_state, action])
        prev_state = state

    if display:
        pygame.quit()

    return game_data, score

# Produce Training Data
if input('New Training Set?: ') == 'y':
    training_data, score = run_game(bot=False, display=True)
    print('{} Score, {} Training examples'.format(score,len(training_data)))
    training_data_save = np.array(training_data)
    np.save('training_data.npy', training_data_save)

# Pass input
else :
    training_data = np.load('training_data.npy', allow_pickle=True)
    #actions = [data[1] for data in training_data]
    #run_game(actions=actions, display=True)

# Define Network
def define_model(shape):
    #Layers
    model = Sequential()

    model.add(LSTM(units=32, input_shape=(shape[1],1), return_sequences=True, name='layer_1'))
    model.add(Dropout(0.2))

    model.add(LSTM(units=32, input_shape=(shape[1], 1), return_sequences=False, name='layer_2'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax', name='output_layer'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

# Train Network
def train_model(training_data):
    X = np.array([data[0] for data in training_data]).reshape(-1, len(training_data[0][0]),1)  # Transpose + Add dimension
    y = np.asarray([data[1] for data in training_data])

    model = define_model(shape=X.shape)

    model.fit(X, y, epochs=num_epochs, shuffle=False, verbose=2)

    return model

# Train Model
if input('Retrain Network?: ') == 'y':
    num_epochs = 4
    model = train_model(training_data)
    model.save('trained_model.h5')
else:
    model = load_model('trained_model.h5')

# Test model

game_data, score = run_game(bot=True, display=True)


