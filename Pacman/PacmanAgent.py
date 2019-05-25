import cv2
from keras import optimizers
from keras.initializers import normal
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

import LearnUtil.CustomLosses as cl
from Agents.Discrete_External_Invoke_DQN_Agent import Discrete_External_Invoke_DQN_Agent
from LearnUtil.MetaSavingFunctions import get_meta_data_types_to_save, get_meta_data_types_functions
from gym.envs.registration import register, make
import random



def analyzeStateFunc(state):
    return 'observation', 'done'

def actionTranslator(action, backTranslate = False):
    return 'translatedAction'

def model_function(input_shape, action_size, data_format, learning_rate):
    model = Sequential()

    model.add(
        Convolution2D(16, (3, 3), strides=(3, 3), name='first', data_format=data_format, padding='same',
                      activation='relu', kernel_initializer=normal(stddev=0.01),
                      input_shape=input_shape))
    model.add(
        Convolution2D(32, (3, 3), strides=(2, 2), name='second', data_format=data_format, padding='same',
                      activation='relu', kernel_initializer=normal(stddev=0.01)))
    model.add(Flatten(name='flat'))

    model.add(Dense(64, name='dense', activation='relu', kernel_initializer=normal(stddev=0.01)))
    model.add(
        Dense(action_size, name='out', activation='linear', kernel_initializer=normal(stddev=0.01)))
    model.compile(optimizer=optimizers.Adam(learning_rate), loss=cl.huber_loss)
    return model


def convert_input_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = image[0:400, 44:244]
    image2 = cv2.resize(image2, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return image2


def preprocess_input(memory):
    # get next input (use newest memories)
    inp = cv2.subtract(memory.get_nth_newest(), memory.get_nth_newest(1))
    # inp = cv2.subtract(np.int16(memory.get_nth_newest()), np.int16(memory.get_nth_newest(1)))
    # inp = np.where(inp<0, -inp, inp).astype(np.uint8)
    thresh, inp = cv2.threshold(inp, 1, 255, cv2.THRESH_BINARY)
    # inp = cv2.morphologyEx(inp, cv2.MORPH_CLOSE, (3,3), iterations=1)
    inp = (inp / 255).astype('float32')
    return inp


X = Discrete_External_Invoke_DQN_Agent(model_function=model_function, save_dir='../weights/Pacman/',
                       max_epos=10000, action_size=5,
                       state_size=60, second_size=30, dim=2,
                       frames_input=1, lr=1e-4,
                       model_name='PacmanModel', load_weights=False, epos_snap=450,
                       meta_data_types_to_save=get_meta_data_types_to_save(),
                       meta_data_types_functions=get_meta_data_types_functions(), epos_data_types_to_save=['QValues'],
                       action_idle=0, action_idle_multiply=2,
                       explore_jump=0.45, expl_rate=0.7, expl_min=0.05,
                       save_memory_length=None,
                       ddqn=False, init_reward=0,
                       past_epos=0,
                       preprocess_state_func=convert_input_image, preprocess_input_func=preprocess_input,
                       operation_memory=1,
                        analyzeStateFunc=analyzeStateFunc, actionTranslator=actionTranslator
                       )
def getAgent():
    return X
