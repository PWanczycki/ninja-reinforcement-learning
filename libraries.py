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
