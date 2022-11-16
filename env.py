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


class NGame(Env):
    # Setup environment action and observation shapes
    def __init__(self):
        super().__init__()
        # setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        # 5 actions: 0 = jump, 1 = left, 2 = right, 3 = jump+left, 4 = jump+right, 5 = no action
        self.action_space = Discrete(6)
        # Define extraction parameters for the game
        self.cap = mss()
        # Change these
        self.game_location = {'top': 135, 'left': 460, 'width': 983, 'height': 800}
        self.finish_location = {'top': 405, 'left': 630, 'width': 660, 'height': 70}

    # What is called to perform an action in the game
    def step(self, action):

        match action:
            case 0:
                # keyDown jump, keyUp left and right
                pydirectinput.keyUp('left')
                pydirectinput.keyUp('right')
                pydirectinput.keyDown('z')
            case 1:
                # keyDown left, keyUp jump and right
                pydirectinput.keyUp('z')
                pydirectinput.keyUp('right')
                pydirectinput.keyDown('left')
            case 2:
                # keyDown right, keyUp jump and left
                pydirectinput.keyUp('z')
                pydirectinput.keyUp('left')
                pydirectinput.keyDown('right')
            case 3:
                # keyDown left and jump, keyUp right
                pydirectinput.keyUp('right')
                pydirectinput.keyDown('left')
                pydirectinput.keyDown('z')
            case 4:
                # keyDown right and jump, keyUp left
                pydirectinput.keyUp('left')
                pydirectinput.keyDown('right')
                pydirectinput.keyDown('z')
            case 5:
                # keyUp jump, left, and right
                pydirectinput.keyUp('z')
                pydirectinput.keyUp('left')
                pydirectinput.keyUp('right')

        """
        done, done_cap = self.get_done()
        observation = self.get_observation()
        reward = some_reward
        info = {}
        return observation, reward, done, info
        """

    # Visualizes the game
    def render(self):
        pass

    # Restarts the game
    def reset(self):
        pass

    def get_observation(self):
        # Get the screen capture of the game
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]
        # Greyscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        # Resize
        resized = cv2.resize(gray, (100, 83))

        # Add channels first
        channel = np.reshape(resized, (1, 83, 100))
        return raw

    def close_observation(self):
        pass

    def get_finish(self):
        # Get the finished screen
        finish_cap = np.array(self.cap.grab(self.finish_location))[:, :, :3]
        # We need to capture both the "ouch" = continue and the "game over screen"
        finish_strings = ['ouch', 'Game', 'level']
        return finish_cap


env = NGame()
plt.imshow(cv2.cvtColor(env.get_observation(), cv2.COLOR_BGR2RGB))
plt.show()
