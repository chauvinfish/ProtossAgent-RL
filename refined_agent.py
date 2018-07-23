import random
import math
import os

import numpy as np
import pandas as pd
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_RESERACH_ZERGLING_SPEED = actions.FUNCTIONS.Research_ZerglingMetabolicBoost_quick.id
_BUILD_ZERGLING = actions.FUNCTIONS.Train_Zergling_quick.id
_BUILD_HYDRALISKDEN = actions.FUNCTIONS.Build_HydraliskDen_screen.id
_BUILD_SPWANINGPOOL = actions.FUNCTIONS.Build_SpawningPool_screen.id
_BUILD_HATCHERY = actions.FUNCTIONS.Build_Hatchery_screen.id
_BUILD_DRONE = actions.FUNCTIONS.Train_Drone_quick.id
_BUILD_ROACH = actions.FUNCTIONS.Train_Roach_quick.id
_BUILD_ROACHWARREN = actions.FUNCTIONS.Build_RoachWarren_screen.id
_BUILD_HYDRALISK = actions.FUNCTIONS.Train_Hydralisk_quick.id
_RESERACH_ROACH_SPEED = actions.FUNCTIONS.Research_GlialRegeneration_quick.id
_RESEARCH_HYDRALISK_SPEED = actions.FUNCTIONS.Research_MuscularAugments_quick.id
_RESEARCH_HYDRALISK_RANGE = actions.FUNCTIONS.Research_GroovedSpines_quick.id
_RESEARCH_BANELING_SPEED = actions.FUNCTIONS.Research_CentrifugalHooks_quick.id
_BUILD_BANELING = actions.FUNCTIONS.Train_Baneling_quick.id
_BUILD_BANELINGNEST = actions.FUNCTIONS.Build_BanelingNest_screen.id
_BUILD_EXTRACTOR = actions.FUNCTIONS.Build_Extractor_screen.id
_BUILD_LAIR = actions.FUNCTIONS.Morph_Lair_quick.id
_BUILD_OVERLORD = actions.FUNCTIONS.Train_Overlord_quick.id
_BUILD_TUMORQUEEN = actions.FUNCTIONS.Build_CreepTumor_Queen_screen.id
_INJECTLARVA = actions.FUNCTIONS.Effect_InjectLarva_screen.id
_SELECT_LARVA = actions.FUNCTIONS.select_larva.id
_SELECT_IDLE_DRONE =actions.FUNCTIONS.select_idle_worker.id
_BUILD_QUEEN = actions.FUNCTIONS.Train_Queen_quick.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5
_LARVA_NUM = 10
_SUPPLY_CAP = 4

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341
_GAS = 342
_EXTRACTOR = 88
_HATCHERY = 86
_DRONE = 104
_SPWANINGPOOL = 89
_HYDRALISKDEN = 91
_ROACHWARREN = 97
_BANELINGNEST = 96
_LAIR =  100
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]
_QUEEN = 126

DATA_FILE = 'refined_agent_data'

ACTION_BUILD_EXTRACTOR = 'buildextractor'
ACTION_BUILD_DRONE = 'builddrone'
ACTION_BUILD_ZERGLING = 'buildzergling'
ACTION_BUILD_ROACH = 'buildroach'
ACTION_BUILD_HYDRALISK = 'buildhydralisk'
ACTION_BUILD_SPWANINGPOOL = 'buildspwaningpoll'
ACTION_BUILD_HYDRALISKDEN = 'buildhydraliskden'
ACTION_BUILD_ROACHWARREN = 'buildroachwarren'
ACTION_BUILD_BANELINGNEST = 'buildbanelingnest'
ACTION_BUILD_HATCHERY = 'buildhatchery'
ACTION_BUILD_LAIR = 'buildlair'
ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'
ACTION_RESEARCH_ZERGLING_SPEED = 'researchzerglingspeed'
ACTION_RESEARCH_ROACH_SPEED = 'researchroachspeed'
ACTION_RESEARCH_HYDRALISK_SPEED = 'researchhydraliskspeed'
ACTION_RESEARCH_HYDRALISK_RANGE = 'researchhydraliskrange'
ACTION_RESEARCH_BANELING_SPEED = 'researchbanelingspeed'
ACTION_BUILD_OVERLORD = 'buildoverlord'
ACTION_BUILD_TUMORQUEEN = 'buildtumorqueen'
ACTION_INJECTLARVA = 'injectlarva'
ACTION_BUILD_BANELING = 'buildbaneling'
ACTION_IDLE_DRONE = 'idledrone'
ACTION_BUILD_QUEEN = 'buildqueen'
smart_actions = [
ACTION_DO_NOTHING,
ACTION_BUILD_HATCHERY,
ACTION_BUILD_LAIR,
ACTION_BUILD_EXTRACTOR,
ACTION_BUILD_DRONE,
ACTION_BUILD_ZERGLING,
ACTION_BUILD_HYDRALISK,
ACTION_BUILD_ROACH,
ACTION_BUILD_BANELING,
ACTION_BUILD_SPWANINGPOOL,
ACTION_BUILD_HYDRALISKDEN,
ACTION_BUILD_ROACHWARREN,
ACTION_BUILD_BANELINGNEST,
ACTION_IDLE_DRONE,
ACTION_BUILD_OVERLORD,
ACTION_BUILD_QUEEN

]
drone_actions = [
ACTION_BUILD_BANELINGNEST,
ACTION_BUILD_ROACHWARREN,
ACTION_BUILD_HYDRALISKDEN,
ACTION_BUILD_SPWANINGPOOL,
ACTION_BUILD_EXTRACTOR

]
queen_actions = [
ACTION_INJECTLARVA,
ACTION_BUILD_TUMORQUEEN
]
larva_actions = [
ACTION_BUILD_DRONE,
ACTION_BUILD_ZERGLING,
ACTION_BUILD_HYDRALISK,
ACTION_BUILD_ROACH,
ACTION_BUILD_OVERLORD
]
building_actions = [
ACTION_BUILD_LAIR,
ACTION_RESEARCH_ZERGLING_SPEED,
ACTION_RESEARCH_ROACH_SPEED,
ACTION_RESEARCH_HYDRALISK_SPEED,
ACTION_RESEARCH_HYDRALISK_RANGE,
ACTION_RESEARCH_BANELING_SPEED,
ACTION_BUILD_QUEEN
]
morph_actions = [
ACTION_BUILD_BANELING
]
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=[]):
        self.check_state_exist(observation)
        
        self.disallowed_actions[observation] = excluded_actions
        
        state_action = self.q_table.ix[observation, :]
        
        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            action = np.random.choice(state_action.index)
            
        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return
        
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        
        s_rewards = self.q_table.ix[s_, :]
        
        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max()
        else:
            q_target = r  # next state is terminal
            
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class SparseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgent, self).__init__()
        
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        
        self.previous_action = None
        self.previous_state = None

        self.cc_y = None
        self.cc_x = None
        self.zerglingspeed_upgraded = 0
        self.roachspeed_upgraded = 0
        self.hydraliskspeed_upgraded = 0
        self.hydraliskrange_upgraded = 0
        self.banelingspeed_uprgaded = 0

        self.move_number = 0
        
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
        
    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        
        return [x + x_distance, y + y_distance]
    
    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]
        
        return [x, y]
    
    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
            
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)
        
    def step(self, obs):
        super(SparseAgent, self).step(obs)
        
        if obs.last():
            reward = obs.reward
        
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
            
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
            
            self.previous_action = None
            self.previous_state = None
            
            self.move_number = 0
            
            return actions.FunctionCall(_NO_OP, [])
        
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

        if obs.first():
            player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
            self.cc_y, self.cc_x = (unit_type == _HATCHERY).nonzero()

        cc_y,cc_x = (unit_type == _HATCHERY).nonzero()
        cc_count = int(round(len(cc_y)/247))
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))
        #NO WANDER SUPPLYDEPOT LOOKS HALF SMALL AS BARRACKS
        spwaningpool_y , spwaningpool_x = (unit_type == _SPWANINGPOOL).nonzero()
        spwaningpool_count = 1 if spwaningpool_y.any() else 0

        roachwarren_y,roachwarren_x =(unit_type == _ROACHWARREN).nonzero()
        roachwarren_count = 1 if roachwarren_y.any() else 0

        hydraliskden_y, hydraliskden_x = (unit_type == _HYDRALISKDEN).nonzero()
        hydraliskden_count = 1 if hydraliskden_y.any() else 0

        banelingnest_y, banelingnest_x = (unit_type == _BANELINGNEST).nonzero()
        banelingnest_count = 1 if banelingnest_y.any() else 0

        lair_y, lair_x = (unit_type == _LAIR).nonzero()
        lair_count = 1 if lair_y.any() else 0

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))

            
        supply_used = obs.observation['player'][3]
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        worker_supply = obs.observation['player'][6]
        
        supply_free = supply_limit - supply_used
        
        if self.move_number == 0:
            self.move_number += 1
            
            current_state = np.zeros(24)
            current_state[0] = cc_count
            current_state[1] = obs.observation['player'][_SUPPLY_CAP]
            current_state[2] = obs.observation['player'][_LARVA_NUM]
            current_state[3] = obs.observation['player'][_ARMY_SUPPLY]
            current_state[12] = lair_count
            current_state[13] = spwaningpool_count
            current_state[14] = roachwarren_count
            current_state[15] = hydraliskden_count
            current_state[16] = banelingnest_count
            #current_state[17] = zerglingspeed_upgraded
            #current_state[18] = banelingspeed_upgraded
            current_state[19] = obs.observation['player'][1]
            current_state[20] = obs.observation['player'][2]
            #current_state[21] = hydraliskspeed_upgraded
            #current_state[22] = hydraliskrange_upgraded
            #current_state[23] = roachspeed_upgraded

            hot_squares = np.zeros(4)        
            enemy_y, enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))
                
                hot_squares[((y - 1) * 2) + (x - 1)] = 1
            
            if not self.base_top_left:
                hot_squares = hot_squares[::-1]
            
            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]
    
            green_squares = np.zeros(4)        
            friendly_y, friendly_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            for i in range(0, len(friendly_y)):
                y = int(math.ceil((friendly_y[i] + 1) / 32))
                x = int(math.ceil((friendly_x[i] + 1) / 32))
                
                green_squares[((y - 1) * 2) + (x - 1)] = 1
            
            if not self.base_top_left:
                green_squares = green_squares[::-1]
            
            for i in range(0, 4):
                current_state[i + 8] = green_squares[i]
    
            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))
        
            excluded_actions = []
           # if supply_depot_count == 2 or worker_supply == 0:
           #     excluded_actions.append(1)
           #
           # if supply_depot_count == 0 or barracks_count == 2 or worker_supply == 0:
           #     excluded_actions.append(2)

            #if supply_free == 0 or barracks_count == 0:
             #   excluded_actions.append(3)
                
            #if army_supply == 0:
            #    excluded_actions.append(4)
             #   excluded_actions.append(5)
              #  excluded_actions.append(6)
               # excluded_actions.append(7)

            rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)

            self.previous_state = current_state
            self.previous_action = rl_action
        
            smart_action, x, y = self.splitAction(self.previous_action)
            if smart_action == ACTION_IDLE_DRONE:
                if _SELECT_IDLE_DRONE in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_IDLE_DRONE, [_NOT_QUEUED])
            elif smart_action in drone_actions:
                unit_y, unit_x = (unit_type == _DRONE).nonzero()
                    
                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]
                    
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                
            elif smart_action in larva_actions:
                unit_y, unit_x = (unit_type == _HATCHERY).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif smart_action in building_actions:
                if smart_action == ACTION_BUILD_LAIR or smart_action== ACTION_BUILD_QUEEN:
                    unit_y, unit_x = (unit_type == _HATCHERY).nonzero()
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        target = [unit_x[i], unit_y[i]]

                        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

                elif smart_action == ACTION_RESEARCH_HYDRALISK_RANGE or smart_action == ACTION_RESEARCH_HYDRALISK_SPEED:
                    if smart_action == ACTION_BUILD_LAIR:
                        unit_y, unit_x = (unit_type == _HYDRALISKDEN).nonzero()
                        if unit_y.any():
                            i = random.randint(0, len(unit_y) - 1)
                            target = [unit_x[i], unit_y[i]]

                            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                elif smart_action == ACTION_RESEARCH_BANELING_SPEED:
                    unit_y, unit_x = (unit_type == _BANELINGNEST).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        target = [unit_x[i], unit_y[i]]

                        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                elif smart_action == ACTION_RESEARCH_ZERGLING_SPEED:
                    unit_y, unit_x = (unit_type == _SPWANINGPOOL).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        target = [unit_x[i], unit_y[i]]

                        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                elif smart_action == ACTION_RESEARCH_ROACH_SPEED:
                    unit_y, unit_x = (unit_type == _ROACHWARREN).nonzero()
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        target = [unit_x[i], unit_y[i]]

                        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            elif smart_action in queen_actions:
                unit_y, unit_x = (unit_type == _QUEEN).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_ATTACK:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        
        elif self.move_number == 1:
            self.move_number += 1
            
            smart_action, x, y = self.splitAction(self.previous_action)
            if smart_action == ACTION_IDLE_DRONE:
                if _HARVEST_GATHER in obs.observation['available_actions']:
                    DICE = 1#round(random.random())
                    if DICE == 0:
                        unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
                    else:unit_y, unit_x = (unit_type == _EXTRACTOR).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])

            elif smart_action == ACTION_BUILD_QUEEN:
                if _BUILD_QUEEN in obs.observation['available_actions']:
                    return actions.FunctionCall(_BUILD_QUEEN, [_QUEUED])
            elif smart_action == ACTION_BUILD_EXTRACTOR:
                if  _BUILD_EXTRACTOR in obs.observation['available_actions']:
                    if self.cc_y.any():
                        unit_y, unit_x = (unit_type == _GAS).nonzero()
                        if unit_y.any():
                            i = random.randint(0, len(unit_y) - 1)

                            m_x = unit_x[i]
                            m_y = unit_y[i]

                            target = [int(m_x), int(m_y)]
                            return actions.FunctionCall(_BUILD_EXTRACTOR, [_NOT_QUEUED, target])
            elif smart_action == ACTION_BUILD_SPWANINGPOOL:
                if  spwaningpool_count == 0 and _BUILD_SPWANINGPOOL in obs.observation['available_actions']:
                    if self.cc_y.any():
                        target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                        return actions.FunctionCall(_BUILD_SPWANINGPOOL, [_NOT_QUEUED, target])
            elif smart_action == ACTION_BUILD_HATCHERY:
                if  _BUILD_HATCHERY in obs.observation['available_actions']:
                    if self.cc_y.any():
                        target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                        return actions.FunctionCall(_BUILD_HATCHERY, [_NOT_QUEUED, target])
            elif smart_action == ACTION_BUILD_ROACHWARREN:
                if  roachwarren_count == 0 and _BUILD_ROACHWARREN in obs.observation['available_actions']:
                    if self.cc_y.any():
                        target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)

                        return actions.FunctionCall(_BUILD_ROACHWARREN, [_NOT_QUEUED, target])
            elif smart_action == ACTION_BUILD_BANELINGNEST:
                if banelingnest_count == 0 and _BUILD_BANELINGNEST in obs.observation['available_actions']:
                    if self.cc_y.any():

                        target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)

                        return actions.FunctionCall(_BUILD_BANELINGNEST, [_NOT_QUEUED, target])
            elif smart_action == ACTION_BUILD_HYDRALISKDEN:
                if hydraliskden_count == 0 and _BUILD_BANELINGNEST in obs.observation['available_actions']:
                    if self.cc_y.any():
                        target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                        return actions.FunctionCall(_BUILD_HYDRALISKDEN, [_NOT_QUEUED, target])
            elif smart_action in larva_actions:
                if _SELECT_LARVA in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_LARVA,[])
            elif smart_action == ACTION_RESEARCH_ROACH_SPEED:
                if _RESERACH_ROACH_SPEED in obs.observation['available_actions']:
                    return actions.FunctionCall(ACTION_RESEARCH_ROACH_SPEED, [_QUEUED])

            elif smart_action == ACTION_ATTACK:
                do_it = True
                
                if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                    do_it = False
                
                if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                    do_it = False
                
                if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)
                    
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])
                
        elif self.move_number == 2:
            self.move_number = 0
            
            smart_action, x, y = self.splitAction(self.previous_action)



            if smart_action == ACTION_BUILD_OVERLORD:
                if _BUILD_OVERLORD in obs.observation['available_actions']:
                    return actions.FunctionCall(_BUILD_OVERLORD, [_QUEUED])
            elif smart_action == ACTION_BUILD_DRONE:
                if _BUILD_DRONE in obs.observation['available_actions']:
                    return actions.FunctionCall(_BUILD_DRONE, [_QUEUED])

            elif smart_action == ACTION_BUILD_ZERGLING:
                if _BUILD_ZERGLING in obs.observation['available_actions']:
                    return actions.FunctionCall(_BUILD_ZERGLING, [_QUEUED])
            elif smart_action == ACTION_BUILD_ROACH:
                if _BUILD_ROACH in obs.observation['available_actions']:
                    return actions.FunctionCall(_BUILD_ROACH, [_QUEUED])
            elif smart_action == ACTION_BUILD_HYDRALISK:
                if _BUILD_HYDRALISK in obs.observation['available_actions']:
                    return actions.FunctionCall(_BUILD_HYDRALISK, [_QUEUED])
        return actions.FunctionCall(_NO_OP, [])

def main(unused_argv):
	agent = SparseAgent()
	try:
		while True:
			#command line parameters for game session

			with sc2_env.SC2Env(map_name="AbyssalReef",
				players=[sc2_env.Agent(sc2_env.Race.zerg),
				sc2_env.Bot(sc2_env.Race.terran,
				sc2_env.Difficulty.very_easy)],
				agent_interface_format=features.AgentInterfaceFormat(
					feature_dimensions=features.Dimensions(screen=84, minimap=64),
					use_feature_units=True),
					step_mul=16,
					game_steps_per_episode=0,
					visualize=True,
                    disable_fog=False
                                ) as env:

				agent.setup(env.observation_spec(), env.action_spec())
				timesteps = env.reset()
				agent.reset()

				while True:
					step_actions = [agent.step(timesteps[0])]
					if timesteps[0].last():
						break
					timesteps = env.step(step_actions)

	except KeyboardInterrupt:
		pass

if __name__ == "__main__":
	app.run(main)