import cv2
from keras import optimizers
from keras.initializers import normal
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

import LearnUtil.CustomLosses as cl
from Agents.Discrete_DQN_Agent import Discrete_DQN_Agent
from LearnUtil.MetaSavingFunctions import get_meta_data_types_to_save, get_meta_data_types_functions
from gym.envs.registration import register, make
import random

def model_function(input_shape, action_size, data_format, learning_rate):
    model = Sequential()

    model.add(
        Convolution2D(16, (3, 3), strides=(3, 3), name='first', data_format=data_format, padding='same',
                      activation='relu', kernel_initializer=normal(stddev=0.01),
                      input_shape=input_shape))
    model.add(
        Convolution2D(32, (3, 3), strides=(3, 3), name='second', data_format=data_format, padding='same',
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


for game in ['FlappyBird']:
    nondeterministic = False
    register(
        id='{}-v1'.format(game),
        entry_point='gym_ple_custom.PLEEnv:PLEEnv',
        kwargs={'game_name': game, 'display_screen':False, 'rng': random.randint(0, 999999999)},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
        nondeterministic=nondeterministic,
    )


X = Discrete_DQN_Agent(env=make('FlappyBird-v1'), model_function=model_function, save_dir='../weights/FlappyBird/',
                       max_epos=100000, action_size=2,
                       state_size=100, second_size=50, dim=2,
                       frames_input=4, lr=1e-5,
                       model_name='NoPooling', load_weights=True, epos_snap=450,
                       meta_data_types_to_save=get_meta_data_types_to_save(),
                       meta_data_types_functions=get_meta_data_types_functions(), epos_data_types_to_save=['QValues'],
                       visualize_input=False, visualize_layer='second', visualize_components=False,
                       visualize_timeout=15,
                       action_idle=1, action_idle_multiply=5,
                       explore_jump=0.00, expl_rate=0.00, expl_min=0.00,
                       save_memory_length=None,
                       ddqn=False, init_reward=5,
                       past_epos=62100,
                       render=False,
                       preprocess_state_func=convert_input_image, preprocess_input_func=preprocess_input,
                       operation_memory=2, input_order=-1
                       )
X.run()
