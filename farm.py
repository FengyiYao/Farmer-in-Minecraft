from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #2: Run simple mission using raw XML

from builtins import range
from farmworld import World
from myagent import Experience
from enum import Enum
from net import N_ACTIONS, Net
import MalmoPython
import os
import sys
import time
import math
import json
import random
import numpy as np
import torch
from torch import nn, optim

actionMap = {0: 'movenorth 1', 1: 'movesouth 1',
             2: 'moveeast 1', 3: 'movewest 1', 4: 'hotbar.2 1', 5: 'hotbar.1 1', 6: 'tp 0.5 4 0.5'}


def build_model(world, num_actions):
    model = Net(num_actions)
    return model.cuda()


def teleport_to_sheep(world):
    if world.world_state:
        for entity in world.world_state["entities"]:
            if entity["name"] == "Sheep":
                return "tp " + str(entity["x"]) + " 4 " + str(entity["z"] - 1)
    return ""


def take_action(agent_host, world, action):
    if action == 'hold_wheat':
        # assume wheat is in slo t2
        agent_host.sendCommand("hotbar.2 1")
        agent_host.sendCommand("hotbar.2 0")
    elif action == 'hide_wheat':
        agent_host.sendCommand("hotbar.1 1")
        agent_host.sendCommand("hotbar.1 0")
    elif action == 'teleport_to_sheep':
        agent_host.sendCommand(teleport_to_sheep(world))
    else:
        agent_host.sendCommand(action)


def correct_coords(world, agent_host, action):
    x, z = world.coords
    if x % 0.5 != 0 or z % 0.5 != 0:
        x_delta = -0.5 if action == 3 else 0.5
        z_delta = -0.5 if action == 0 else 0.5
        agent_host.sendCommand(
            "tp " + str(int(x) + x_delta) + " 4 " + str(int(z) + z_delta))


if __name__ == "__main__":
    world = World()
    model = build_model(world.world, world.actions)
    max_memory = 1000
    data_size = 50
    experience = Experience(model, max_memory=max_memory)
    agent_host = MalmoPython.AgentHost()
    # -- set up the mission -- #
    mission_file = './farm.xml'
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)


    max_retries = 10
    num_repeats = 1000
    mission_avg_rewards = []
    mission_max_rewards = []
    mission_num_actions = []
    mission_losses = []

    f = open("numactions.txt", "a")
    f1 = open("average_rewards.txt", "w")
    f2 = open("max_rewards.txt", "w")
    f3 = open("loss.txt", "w")
    torch.save(model.state_dict(), 'model%d.pth' % int(time.time()))
    for i in range(num_repeats):
        print("Mission %d of %d" % (i + 1, num_repeats))
        if i % 20 == 19:
            torch.save(model.state_dict(), 'model%d.pth' % i)
        rewards = []
        world.reset()

        envstate = world.observe()

        print()
        print('Repeat %d of %d' % (i+1, num_repeats))

        # Record video config. Record just the first 10 missions or last 10 in the training
        if i < 10 or 40 < i < 50 or 90 < i < 100:
            my_mission_record = MalmoPython.MissionRecordSpec("sheep_lurer_recording_" + str(i) + ".tgz")
            my_mission.requestVideo(800, 500)
            my_mission_record.recordMP4(30, 1000000); #records video with 30fps and at 1000000 bit rate
            my_mission.setViewpoint( 1 )
        
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10001))
        for retry in range(max_retries):
            try:
                agent_host.startMission(my_mission, my_clients, my_mission_record, 0, "%s-%d" % ('Moshe', i))
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2.5)

        print("Waiting for the mission to start", end=' ')
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
        print()
         # Hide the wheat to start each mission.
        agent_host.sendCommand("hotbar.1 1")
        agent_host.sendCommand("hotbar.1 0")
        # -- run the agent in the world -- #
        num_actions = 0
        while world_state.is_mission_running:
            time.sleep(0.01)
            prev_envstate = envstate

            if world.shouldReturn:
                print('Return action: ', end="")
                action = world.returnToStart()
            elif np.random.rand() < 0.10:
                print('Random action: ', end="")
                action = random.choice(world.getValidActions())
            else:
                print('Predicted action: ', end="")
                action = np.argmax(experience.predict(prev_envstate))
            print(action)

            take_action(agent_host, world, actionMap[action])
            num_actions += 1

            world_state = agent_host.getWorldState()
            envstate, reward, game_status = world.update_state(
                world_state, action, agent_host)
            print(reward, game_status)
            rewards.append(reward)

            # Correct the agent's coordinates in case a sheep pushed it
            correct_coords(world, agent_host, actionMap[action])

            game_over = game_status == 'win' or game_status == 'lose'
            episode = [prev_envstate, action, reward, envstate, game_over]

            if not world.shouldReturn:
                experience.remember(episode)
                inputs, targets = experience.get_data(data_size=data_size)
                '''h = model.fit(
                    inputs,
                    targets,
                    epochs=8,
                    batch_size=16,
                    verbose=0,
                )'''
                loss_fn = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                epochs = 10
                for t in range(epochs):
                    for x, y in zip(inputs, targets):
                        x = torch.FloatTensor(x.reshape((1, 1, 21, 21))).cuda()
                        y = torch.FloatTensor(y.reshape((1, len(actionMap)))).cuda()
                        optimizer.zero_grad()
                        y_pred = model(x)
                        loss = loss_fn(y_pred, y)
                        loss.backward()
                        optimizer.step()    
                print(f"loss: {loss}")
                mission_losses.append(loss)
               

            if game_over:
                agent_host.sendCommand("quit")
                break
        # -- clean up -- #

        # compute average reward, and max reward
        template = "Iteration: {:d} | Average Reward: {:.4f} | Max Reward: {:.4f}"
        avg_reward = sum(rewards) / len(rewards)
        mission_avg_rewards.append(avg_reward)
        max_reward = max([r for r in rewards if r != -1]) # ignore -1 rewards
        mission_max_rewards.append(max_reward)
        mission_num_actions.append(num_actions)
        print(template.format(i, avg_reward if rewards else 0, max_reward if rewards else 0))
        print("num actions: ", num_actions)
        time.sleep(0.5)  # (let the Mod reset)
    print("All mission average rewards: ", mission_avg_rewards)
    print("All mission max rewards: ", mission_max_rewards)
    print("All mission number of actions: ", mission_num_actions)
    print("All mission loss: ", mission_losses)
    f1.write("\n".join((str(x) for x in mission_avg_rewards)))
    f2.write("\n".join((str(x) for x in mission_max_rewards)))
    f.write("\n".join((str(x) for x in mission_num_actions)))
    f3.write("\n".join((str(x) for x in mission_losses)))
    f.close()
    f2.close()
    f1.close()
    f3.close()
    torch.save(model.state_dict(), 'model.pth')

    print("Done.")
