import cv2
from keras import optimizers
from keras.initializers import normal, he_normal, he_uniform
from keras.layers import Dense, Flatten
from keras import regularizers
from keras.layers.convolutional import Convolution3D, Convolution2D, MaxPooling3D
from keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib
import LearnUtil.CustomLosses as cl
from Agents.Discrete_External_Invoke_DQN_Agent import Discrete_External_Invoke_DQN_Agent
from LearnUtil.MetaSavingFunctions import get_meta_data_types_to_save, get_meta_data_types_functions
import random
from misio.pacman.pacman import GameState
from misio.pacman.game import Actions, Directions
import numpy as np
#matplotlib.interactive(True)
def analyzeStateFunc(state):
    observation = state
    done = True if state.data._win or state.data._lose else False
    legalActions = state.getLegalActions()
    legalActions = [actionTranslator(action, True) for action in legalActions if actionTranslator(action, True)!=-1]
    return observation, done, legalActions

backTranslateDic = {Directions.NORTH: 0,
            Directions.SOUTH: 1,
            Directions.EAST: 2,
            Directions.WEST: 3,
            Directions.STOP: -1}
translateDic = dict((v,k) for k,v in backTranslateDic.items())

def actionTranslator(action, backTranslate = False, legalActions = None):
    if not backTranslate:
        if legalActions is not None and action not in legalActions:
            return translateDic[-1]
        return translateDic[action]
    else:
        if legalActions is not None and backTranslateDic[action] not in legalActions:
            return -1
        return backTranslateDic[action]

def model_function(input_shape, action_size, data_format, learning_rate):
    model = Sequential()

    model.add(
        Convolution3D(8, (1, 1,4), strides=(1, 1,4), name='first', data_format=data_format, padding='same',
                      activation='leaky_relu', kernel_initializer=he_uniform(),
                      input_shape=input_shape))
    model.add(
        Convolution3D(16, (3, 3, 1), strides=(1, 1, 1), name='sec', data_format=data_format, padding='same',
                      activation='leaky_relu', kernel_initializer=he_uniform(),
                      input_shape=input_shape))
    model.add(MaxPooling3D((2,2,1), padding='same', data_format='channels_first'))
    model.add(
        Convolution3D(32, (3, 3,1), strides=(1, 1,1), name='th', data_format=data_format, padding='same',
                      activation='leaky_relu', kernel_initializer=he_uniform()))
    model.add(MaxPooling3D((2, 2, 1), padding='same', data_format='channels_first'))
    model.add(Flatten(name='flat'))

    model.add(Dense(64, name='dense', activation='leaky_relu', kernel_initializer=he_uniform()))
    model.add(
        Dense(action_size, name='out', activation='linear', kernel_initializer=he_normal()))
    model.compile(optimizer=optimizers.Adam(learning_rate), loss=cl.huber_loss)
    print(model.summary())
    return model
#model_function((1,63,63,4), 5, 'channels_first', 0.1)

def normalizeReward(reward):
    return reward/500

def convert_observation(observation):
    halfx=6
    halfy=6
    template = np.zeros(shape=(1+2*halfy, 1+2*halfx, 4))

    pos_x, pos_y = observation.getPacmanPosition()

    walls = np.array(observation.getWalls().data,dtype=np.uint8)
    width, height = np.shape(walls)

    min_indx = max(0,pos_x - halfx)
    max_indx = min(pos_x + halfx+1, width)

    min_indy = max(0,pos_y - halfy)
    max_indy = min(pos_y + halfy+1, height)

    walls = walls[min_indx:max_indx, min_indy:max_indy]
    food = np.array(observation.getFood().data, dtype=np.uint8)[min_indx:max_indx, min_indy:max_indy]



    insert_min_indx = - min(0, pos_x-halfx)
    insert_max_indx =  1+2*halfx + (min(0, width - (pos_x + halfx+1)))
    insert_min_indy = - min(0, pos_y - halfy)
    insert_max_indy = 1+2*halfy + (min(0, height - (pos_y + halfy+1)))

    template[insert_min_indy:insert_max_indy, insert_min_indx:insert_max_indx, 0] = np.transpose(walls)
    template[insert_min_indy:insert_max_indy, insert_min_indx:insert_max_indx, 1] = np.transpose(food)
    capsules = observation.getCapsules()
    for x,y in capsules:
        gx = x - pos_x + halfx
        gy = y - pos_y + halfy
        if gx >=0 and gx < 1+2*halfx and gy >=0 and gy < 1+2*halfy:
            template[gy,gx,2] = 1
    ghosts=observation.getGhostStates()
    #positions = observation.getGhostPositions()
    for ghost in ghosts:
        x,y = ghost.getPosition()
        gx =int(x)-pos_x+halfx
        gy =int(y)-pos_y+halfy

        if gx >=0 and gx < 1+2*halfx and gy >=0 and gy < 1+2*halfy:
            template[gy,gx,3] = 1 if ghost.scaredTimer == 0 else 0.25+0.5*((40-ghost.scaredTimer)/40)

    template=np.flip(template,axis=0)
    # if np.random.rand() > 0.0:
    #     plt.imshow(template[:,:,1])
    #     plt.show()
    # #     print("okkk")
    return template


def preprocess_input(memory):
    return memory.get_nth_newest()


X = Discrete_External_Invoke_DQN_Agent(model_function=model_function, save_dir='../weights/Pacman/',
                                       max_epos=1000000, action_size=4, memory=20000,
                                       state_shape=(13,13,4),
                                       frames_input=1, lr=3e-5, batch_size=70,
                                       model_name='PacmanModel', load_weights=True, epos_snap=450,
                                       meta_data_types_to_save=get_meta_data_types_to_save(),
                                       meta_data_types_functions=get_meta_data_types_functions(), epos_data_types_to_save=['QValues'],
                                       action_idle=0, action_idle_multiply=1,  #visualize_components=True,

                                       explore_jump=0.0, expl_rate=0.02, expl_min=0.02,
                                       #guided_sample=9/500, guided_sample_base_chance=2e-7, replays_per_episode=2,
                                       save_memory_length=None,
                                       ddqn=False, ddqn_cross_learn=False, epos_equalize_models= 300, discount=0.925,
                                       init_reward=0,  #visualize_layer='sec', visualize_input=True,
                                       past_epos=105750,

                                       preprocess_state_func=convert_observation, preprocess_input_func=preprocess_input,
                                       operation_memory=1,
                                       analyzeStateFunc=analyzeStateFunc, actionTranslator=actionTranslator, normalizeReward=normalizeReward
                                       )
def getAgent():
    return X
