import numpy as np
import json
import math
import time

GATE_COORDINATES = (3, -3)


class World:
    def __init__(self):
        self.reset()

    def reset(self):
        self.coords = (0, 0)
        self.prevCoords = (0, 0)
        self.state = (0, 0)
        self.total_reward = 0
        self.total_steps = 100
        self.sheeps = set()

        self.actions = 7
        self.prevAction = None
        self.world = np.zeros((21, 21))
        self.world_state = None
        self.shouldReturn = False
        self.holding_wheat = False

    # only allow agent to use the first 5 as actions
    def getValidActions(self):
        return [0, 1, 2, 3, 4]

    def game_status(self):
        if self.total_steps > 0:
            if self.total_reward > 200:
                return "win"
            else:
                return "playing"
        else:
            if self.total_reward > 0:
                return "win"
            else:
                return "lose"

    def observe(self):
        return self.world.reshape(-1)

    def agentInPen(self):
        x, z = self.state
        return 5 > x > 0 and -1 > z > -5

    def sheepInPen(self, x, z):
        return 6 > x > 0 and -1 > z > -5

    def returnToStart(self):
        x, z = self.state
        time.sleep(0.3)

        if self.agentInPen():
            if self.shouldReturn:
                self.shouldReturn = False
                return 5
            else:
                return 6

        if x > 9 and z < -1:
            return 3
        elif x < 8 and z > -3:
            return 2
        elif z > -3:
            return 0
        else:
            return 3

    def update_state(self, world_state, action, agent_host):
        pass
