# MSS is for screen capture
from mss import mss
# Sending commands
import pydirectinput
# Open CV allows for frame processing
import cv2
import numpy as np
# this is OCR to detect end of game
import pytesseract
# Visualize captured frames
from matplotlib import pyplot as plt
# Brings in time for pauses
import time
from gym import Env
from gym.spaces import Box, Discrete

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Erict\AppData\Local\Tesseract-OCR\tesseract.exe'


class NGame(Env):
    # Setup environment action and observation shapes
    def __init__(self):
        super().__init__()
        # setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1, 200, 300), dtype=np.uint8)
        # 5 actions: 0 = jump, 1 = left, 2 = right, 3 = jump+left, 4 = jump+right, 5 = no action
        self.action_space = Discrete(6)
        # Define extraction parameters for the game
        self.cap = mss()
        # Captures the game state
        self.game_location = {'top': 600, 'left': 488, 'width': 520, 'height': 290}

        # Get the death message, 'ouch...'
        self.death_location = {'top': 290, 'left': 878, 'width': 40, 'height': 20}
        # Get the game over message, 'Game Over' i.e. running out of time
        self.game_over_location = {'top': 361, 'left': 709, 'width': 60, 'height': 31}
        # Get the complete level message, 'level'
        self.level_complete_location = {'top': 284, 'left': 851, 'width': 57, 'height': 30}
        # Get the victory message, episode is completed
        self.victory_location = {'top': 395, 'left': 709, 'width': 60, 'height': 1}
        # Get the level timer bar
        self.timerbar_location = {'top': 135 + 45, 'left': 460 + 110, 'width': 780, 'height': 1}
        # Get the level timer number
        self.time_location = {'top': 160, 'left': 500, 'width': 65, 'height': 45}

        self.last_time_left = [39, 39]

    # What is called to perform an action in the game
    def step(self, action):
        # action maps to corresponding keys pressing down and all other keys going up
        match action:
            case 0:
                # jump
                pydirectinput.keyUp('left')
                pydirectinput.keyUp('right')
                pydirectinput.keyDown('z')
            case 1:
                # left
                pydirectinput.keyUp('z')
                pydirectinput.keyUp('right')
                pydirectinput.keyDown('left')
            case 2:
                # right
                pydirectinput.keyUp('z')
                pydirectinput.keyUp('left')
                pydirectinput.keyDown('right')
            case 3:
                # jump left
                pydirectinput.keyUp('right')
                pydirectinput.keyDown('left')
                pydirectinput.keyDown('z')
            case 4:
                # jump right
                pydirectinput.keyUp('left')
                pydirectinput.keyDown('right')
                pydirectinput.keyDown('z')
            case 5:
                # no op
                pydirectinput.keyUp('z')
                pydirectinput.keyUp('left')
                pydirectinput.keyUp('right')

        reset = False
        observation = self.get_observation()
        info = {}
        reward = 0
        self.last_time_left[0] = self.last_time_left[1]
        self.last_time_left[1] = self.get_time_left()

        #cur_time = self.get_time()

        #Only check game state when the timer has not changed
        if self.check_episode_exit():
            time.sleep(1)
            if self.check_victory():
                reward = 500 + self.last_time_left[0]
                reset = True
                print("victory")
            else:
                reward = -20 + self.last_time_left[0] // 10
                reset = True
                print("death")


        # alternate gameover check (force death if timerbar is too low)
        if reward == 0 and self.last_time_left[0] < 3:
            pydirectinput.press('k')
            reset = True
            reward = -100
            print("gameover")

        return observation, reward, reset, info

    # Visualizes the game
    def render(self):
        pass

    # Restarts the game
    def reset(self):
        # reset key presses to avoid unexpected behaviour
        self.last_time_left = [39, 39]
        pydirectinput.press('z')
        pydirectinput.press('z')

        return self.get_observation()

    def get_observation(self):
        # Get the screen capture of the game
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]
        # Greyscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        # Resize
        resized = cv2.resize(gray, (300, 200))

        # Add channels first
        channel = np.reshape(resized, (1, 200,300))
        return channel

    def close(self):
        pass

    def check_victory(self):
        # Check if the episode has been beaten, victory screen
        pixels_under_VICTORY = np.array(self.cap.grab(self.victory_location))[:, :, :3].tolist()

        for pixel in pixels_under_VICTORY[0]:
            #print(pixel)
            if pixel != [208, 202, 202] and pixel != [136, 121, 121]:
                return True
        return False

    def get_time(self):
        capture = pytesseract.image_to_string(np.array(self.cap.grab(self.time_location))[:, :, :3])
        return capture

    def check_status(self, prev):
        if prev == self.get_time():
            return True
        else:
            return False

    def check_episode_exit(self):
        # alternate check for "Victory" and "Game Over"
        # check if timer is still shown by extracting 1 row of pixels and checking for non-bg colour (gray)
        timer = np.array(self.cap.grab({'top': 179, 'left': 510, 'width': 44, 'height': 1}))[:, :, :3].tolist()

        for i in range(44):
            if timer[0][i] != [136, 121, 121]:
                return False

        # all pixels were gray, timer is no longer shown and game will exit episode
        return True

    def get_time_left(self):
        # grab timer screenshot
        timerbar = np.array(self.cap.grab(self.timerbar_location))[:, :, :3].tolist()

        PURPLE = [136, 34, 34]
        GRAY = [136, 121, 121]

        # iterate twice through pixels 60 at a time, (13 positions/blocks)
        # first iter, stop if middle of block is gray
        # second iter, stop if dark purple

        # check for timer <= 180
        for i in range(78):
            if timerbar[0][i * 10 + 5] == GRAY:
                return i

        # check for timer > 180
        for i in range(78):
            # if pixel i*60+30 is purple, return i+7
            if timerbar[0][i * 10 + 5] == PURPLE:
                return i + 78

        # default negative reward if timer messes up
        return 0

"""
env = NGame()
img = env.get_observation()

plt.imshow(cv2.cvtColor(env.get_observation()[0],cv2.COLOR_BGR2RGB))
plt.show()

env = NGame()
img = env.get_observation()

t = env.get_time()
print(env.check_status(t))

# Status Checking
env.check_death()
env.check_gameover()
env.check_level_complete()
env.check_victory()
env.get_time()

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
# input testing
# game states: "RUN", "DEAD", "TIME", "COMPLETE", "VICTORY"
game_state = "RUN"
for episode in range(3):
    obs = env.reset()
    game_state = "RUN"
    total_reward = 0
    while game_state == "RUN":
        obs, reward, game_state, info = env.step(env.action_space.sample())
        total_reward += reward
    print("Total reward for episode {} is {}".format(episode, total_reward))

# reset keystrokes to avoid holding last action
pydirectinput.press('z')
pydirectinput.press('left')
pydirectinput.press('right')

"""



