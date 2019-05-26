import copy as cp
import time
from pprint import pprint

import cv2
import numpy as np
from misio.pacman.learningAgents import ReinforcementAgent


import LearnUtil.Visualize_Activations as va
from LearnUtil.InputUtils import InputUtils
from LearnUtil.Memory import Memory
from LearnUtil.Model_Saving import Model_Saving


class Discrete_External_Invoke_DQN_Agent(ReinforcementAgent):

    def __init__(self, model_function, save_dir, max_epos, action_size,  # action size - how many possible actions
                 state_shape,  #input sizes
                 frames_input=1, lr=0.0001, batch_size=50,  # learning params-frames(how many past frames model sees)
                 model_name=None, load_weights=False, epos_snap=750,  # for saving/loading purposes
                 meta_data_types_to_save=None, meta_data_types_functions=None, epos_data_types_to_save=None,
                 visualize_input=False, visualize_layer=None, visualize_components=False, visualize_timeout=30,
                 action_idle=0, action_idle_multiply=1,  # random exploration modificators
                 epos_explore_jump=150, explore_jump=0.55, expl_rate=1, expl_decay=0.999, expl_min=0.05,
                 memory=5000, save_memory_length=None,  # memory-learning memory size, save-memory chunk size
                 ddqn=False, ddqn_cross_learn=False, epos_equalize_models=None, discount=0.95, init_reward=5,  # dqn parameters
                 past_epos=0,  # how long model was trained already
                 preprocess_state_func=lambda x: x, preprocess_input_func=lambda x: x,
                 start_train_long_memory_batch_mult=50, replays_per_episode=1,
                 operation_memory=2, input_order=1,  # one frame processing memory (might use frames before)
                 guided_sample=None, guided_sample_base_chance=1.5e-7,  # in replay epos scaling chance to see specific reward
                 analyzeStateFunc = lambda x:(x[0],x[1],x[2]), actionTranslator = lambda x, y: x, normalizeReward = lambda x:x
                 #state analyzer return observation/done
                 # and actionTranslator is two way model symbol/env symbol action with param True if translate to model
                 ):
        ReinforcementAgent.__init__(self, numTraining=max_epos, epsilon=0.0, gamma=discount, alpha=lr)
        self._past_epos = past_epos
        self._frames_input = frames_input
        self._state_shape = state_shape
        self._action_size = action_size
        self._ddqn = ddqn  # if ddqn is true then Double q learning is used
        self._ddqn_cross_learn= ddqn_cross_learn
        self._epos_equalize_models = epos_equalize_models  # every n epos set weights of target model to model
        self._input_utils = InputUtils(frames_input, state_shape)
        self._model_build(load_weights, save_dir, epos_snap, model_name, meta_data_types_to_save,
                          meta_data_types_functions,
                          epos_data_types_to_save, model_function, lr)
        self._operation_memory = Memory(operation_memory)
        self._short_memory = Memory(frames_input, previous_states=1)
        self._memory = Memory(memory, previous_states=frames_input - 1)
        if save_memory_length is not None:
            self._saving_memory = Memory(save_memory_length, save_and_reset=True, model_saving=self._ms)
        else:
            self._saving_memory = None


        self._max_epos = max_epos

        self._batch_size = batch_size
        self._start_train_long_memory_batch_mult = start_train_long_memory_batch_mult

        self._exploration_rate = expl_rate
        self._exploration_decay = expl_decay
        self._exploration_min = expl_min
        self._epos_explore_jump = epos_explore_jump
        self._explore_jump = explore_jump

        # action idle is action that should be picked the most, changes most slightly or when no idea
        # action idle is inactive when action_idle_multiply=1, multiply shouldnt be less than 1
        self._action_idle = action_idle
        self._action_idle_multiply = action_idle_multiply

        self._discount_rate = discount
        self._rew_bonus = 0
        self._init_rew = init_reward

        self._preprocess_state_func = preprocess_state_func
        self._preprocess_input_func = preprocess_input_func
        self._saveQVals = True if epos_data_types_to_save is not None and 'QValues' in epos_data_types_to_save else False
        self._visualize_input = visualize_input
        self._visualize_layer = visualize_layer
        self._visualize_components = visualize_components
        self._visualize_timeout = visualize_timeout


        self._time = time.time()
        self._input_order = input_order
        self._ms.epos_data_types_to_reset.append('reward')
        self._guided_sample = guided_sample
        self._memory.chance_factor = guided_sample_base_chance
        self.analyzeStateFunc = analyzeStateFunc
        self.actionTranslator = actionTranslator
        self.normalizeReward = normalizeReward
        self.replaysPerEpisode = replays_per_episode
        self.i_episode = past_epos
        if self._visualize_layer is not None:
            cv2.namedWindow('show', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('show', 1000, 1000)

    def _model_build(self, load_weights, save_dir, epos_snap, model_name, meta_data_types_to_save,
                     meta_data_types_functions, epos_data_types_to_save, model_function, learning_rate):

        if load_weights:
            self._ms = Model_Saving(None, save_dir, epos_snap, model_name, meta_data_types_to_save,
                                    meta_data_types_functions, epos_data_types_to_save)
            self._model = self._ms.load_model(self._past_epos)
            print("Loaded values!")
        else:

            self._model = model_function(input_shape=(self._input_utils.get_input_shape()),
                                         action_size=self._action_size, data_format='channels_first',
                                         learning_rate=learning_rate)
            self._ms = Model_Saving(self._model, save_dir, epos_snap, model_name, meta_data_types_to_save,
                                    meta_data_types_functions, epos_data_types_to_save)

        if self._ddqn:
            if load_weights:
                self._target_ms = Model_Saving(None, save_dir, epos_snap, model_name+'_target', [],
                                        {}, [])
                self._target_model = self._target_ms.load_model(self._past_epos)
                print("Loaded values!")
            else:
                self._target_model = cp.deepcopy(self._model)
                self._target_model.set_weights(self._model.get_weights())
                self._target_ms = Model_Saving(self._target_model, save_dir, epos_snap, model_name+'_target', [],
                                           {}, [])
        else:
            self._target_model = None

        nodes = [layer.output for layer in self._model.layers]
        pprint(nodes)

    def _build_expected_rewards_and_fit_q_network(self, batch, q_network, target_network):

        future_inputs = [[memories[x]['new_state'] for x in range(self._frames_input)][::self._input_order]
                         for memories in batch]

        current_inputs = [[memories[x]['state'] for x in range(self._frames_input)][::self._input_order]
                          for memories in batch]

        q_input = self._input_utils.reshape_input(future_inputs + current_inputs,
                                                  add_size=2 * self._batch_size)
        q_predictions = q_network.predict(q_input)
        if self._ddqn:
            decide_potential_future_vals = q_predictions[0:self._batch_size]
            estimate_potential_future_vals = target_network.predict(
                self._input_utils.reshape_input(future_inputs, add_size=self._batch_size))
        else:
            decide_potential_future_vals = estimate_potential_future_vals = q_predictions[0:self._batch_size]
        q_current_predictions = q_predictions[self._batch_size:2 * self._batch_size]

        for i in range(self._batch_size):
            target = batch[i][0]['reward']

            if not batch[i][0]['done']:
                target += self._discount_rate * estimate_potential_future_vals[i][
                    np.argmax(decide_potential_future_vals[i])]

            q_current_predictions[i][batch[i][0]['action']] = target

        q_network.fit(self._input_utils.reshape_input(current_inputs, add_size=self._batch_size), q_current_predictions,
                      epochs=1, verbose=2)

    def _get_batch(self, i_episode):
        batch = self._memory.random_sample(self._batch_size)

        # exchange first sample with one latest fail with probability scalling up with time and batch_size
        if self._guided_sample is not None:
            batch = self._memory.replace_first_with_specified_value_by_chance(batch,
                                                                              self._memory.episode_batch_size_chance(
                                                                                  i_episode, self._batch_size),
                                                                              'reward', self._guided_sample)
        return batch

    def _replay(self, i_episode):

        if self._ddqn and self._epos_equalize_models is not None and i_episode % self._epos_equalize_models == 0:
            self._target_model.set_weights(self._model.get_weights())  # update target model

        if len(self._memory.get_memory()) >= self._batch_size * self._start_train_long_memory_batch_mult:
            batch = self._get_batch(i_episode)
            self._build_expected_rewards_and_fit_q_network(batch, q_network=self._model,
                                                           target_network=self._target_model)
            if self._ddqn and self._ddqn_cross_learn:
                batch_target = self._get_batch(i_episode)
                self._build_expected_rewards_and_fit_q_network(batch_target, q_network=self._target_model,
                                                               target_network=self._model)

    def _decide_action(self, times, legalActions=None):
        model_input = self._input_utils.reshape_input(self._short_memory.array_memory()[::self._input_order])
        prediction = self._model.predict(model_input)[0]

        self._currentRunQVals[times] = prediction

        if self._visualize_input:
            self._visualizeInput(model_input)


        # if not control then random
        if np.random.rand() < self._exploration_rate:
            if legalActions is not None:
                actions = ([self._action_idle]*(self._action_idle_multiply-1) if self._action_idle in legalActions else []) + legalActions
            else:
                actions = [self._action_idle] * (
                            self._action_idle_multiply - 1)  + list(range(self._action_size))


            return np.random.choice(actions)

        # if not random then decide
        chosen = np.argmax(prediction)
        if legalActions is not None:
            while(chosen not in legalActions):
                prediction[chosen] = np.min(prediction)-1
                chosen = np.argmax(prediction)

        return chosen

    def _visualizeInput(self, model_input):
        acts = va.get_activations(self._model, model_input,
                                  layer_name=self._visualize_layer)
        va.show_activations(acts, timeout=self._visualize_timeout)
        #vis = np.concatenate((model_input[0][0], model_input[0][1]))#, model_input[0][2], model_input[0][3]))
        vis = model_input[0,0,:,:,1:4]
        #print(model_input[0][0][0])
        cv2.namedWindow('win', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('win', 400, 400)
        cv2.imshow('win', vis)
        cv2.waitKey(self._visualize_timeout)

    def _visualizeComponents(self):
        cv2.imshow('pr', cv2.resize(self._operation_memory.get_nth_newest(1)[2:], (0, 0), fx=4, fy=4))

        cv2.imshow('ob', cv2.resize(self._operation_memory.get_nth_newest(0)[1:], (0, 0), fx=4, fy=4))
        cv2.imshow('inp', cv2.resize(self._short_memory.get_nth_newest()[:,:,1:], (0, 0), fx=4, fy=4))
        cv2.waitKey(self._visualize_timeout)

    def _fill_memory(self, observation, memory):
        while memory.free_memory() > 0:
            memory.store(observation)

    def _prepare_run(self, observation=None):
        self._times = 0  # how many steps played
        self._total_rew = self._init_rew  # total reward throughout one game
        self._currentRunQVals = {}  # one game Q values for actions

        #observation = self._env.reset()  # get first state
        if observation is not None:
            observation = self._preprocess_state_func(observation)
            self._fill_memory(observation, self._operation_memory)
            inp = self._preprocess_input_func(self._operation_memory)
            self._fill_memory(inp, self._short_memory)
            return inp
        return None

    def _preprocess_and_store(self, entity, func, memory):
        entity = func(entity)
        memory.store(entity)

    def _long_memories(self, state, action, reward, new_state, done, i_episode):
        self._memory.remember(state, action, reward, new_state, done)
        if self._saving_memory is not None:
            self._saving_memory.remember(state, action, reward, new_state, done, i_episode)

    def getAction(self, state):
        observation, done, legalActions = self.analyzeStateFunc(state)
        if self._operation_memory.memory_length() == 0:
            self.i_episode += 1
            self._ms.epos_snapshot(self.i_episode)
            if self._ddqn:
                self._target_ms.epos_snapshot(self.i_episode)
            self._prepare_run(observation)

        action = self.actionTranslator(self._decide_action(self._times), False, legalActions)


        self.doAction(state,action)
        return action
    def update(self, state, action, nextState, reward):
        reward = self.normalizeReward(reward)
        observation, done, legalActions = self.analyzeStateFunc(nextState)
        action = self.actionTranslator(action, True)
        self._preprocess_and_store(observation, self._preprocess_state_func,
                                   self._operation_memory)
        self._short_memory.store(self._preprocess_input_func(self._operation_memory))
        if self._visualize_components:
            self._visualizeComponents()

        self._long_memories(self._short_memory.get_nth_newest(1), action, reward,
                            self._short_memory.get_nth_newest(), done, self.i_episode)

        self._total_rew += reward
        self._times += 1
        if self._times%500==0:
            print("still playing")


    def final(self, state):
        ReinforcementAgent.final(self, state)
        self._operation_memory._memory.clear()
        self._short_memory._memory.clear()
        if self._saveQVals:
            self._ms.collect_epos_data('QValues', self._currentRunQVals, self.i_episode)

        self._ms.collect_epos_data('reward', self._total_rew, self.i_episode)
    
        
        for i in range(self.replaysPerEpisode):
            self._replay(self.i_episode)  # train model
        self._exploration_rate = max(self._exploration_rate * (self._exploration_decay ** self._times),
                                     self._exploration_min)  # update exploration rate
        if (self.i_episode + 1) % self._epos_explore_jump == 0:
            self._exploration_rate += self._explore_jump

        time_per_score = (time.time() - self._time) / self._times  # get and update timings
        self._time = time.time()
        print("Episode {} / {}. Score: {} Total: {}, explore: {}, time per frame: {}".format(self.i_episode,
                                                                                             self._max_epos,
                                                                                             self._times,
                                                                                             self._total_rew,
                                                                                             self._exploration_rate,
                                                                                             time_per_score))