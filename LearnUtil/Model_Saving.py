import os
import pickle
from functools import reduce
import copy as cp
import keras


class Model_Saving:
    _var_file_name = 'var.txt'
    _weights_name = '_weights_'
    _metadata_name = '_metadata'

    def __init__(self, model, save_dir, epos_snap, model_name=None, meta_data_types_to_save=None,
                 meta_data_types_functions=None, epos_data_types_to_save=None, epos_data_types_to_reset=None):
        self._save_dir = save_dir
        self._epos_snap = epos_snap
        self._data = {}
        self._model = model
        self._model_path = self._save_dir + model_name
        self._weights_path = self._model_path + self._weights_name
        self._var_path = self._save_dir + self._var_file_name

        self._metadata_path = self._model_path + self._metadata_name
        assert len(meta_data_types_to_save) == len(meta_data_types_functions)
        self._meta_data_types_to_save = meta_data_types_to_save if meta_data_types_to_save is not None else []
        self._meta_data_types_functions = meta_data_types_functions if meta_data_types_functions is not None else {}
        if epos_data_types_to_reset is None:
            epos_data_types_to_reset = cp.deepcopy(epos_data_types_to_save)
        self._epos_data_types_to_save = epos_data_types_to_save if epos_data_types_to_save is not None else []
        self.epos_data_types_to_reset = epos_data_types_to_reset

        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = str(self._var_file())

    def load_model(self, past_epos):
        self._model = keras.models.load_model(self._model_path)
        self._model.load_weights(self._weights_path + str(past_epos))
        return self._model

    def _var_file(self):
        numb = 0
        if not os.path.isfile(self._var_path):
            fo = open(self._var_path, "w")
            fo.write(numb)
            fo.close()
        else:
            fo = open(self._var_path, "r")
            self._numb = int(next(fo))
            fo.close()
        numb += 1
        fo = open(self._var_path, "w")
        fo.write(str(numb))
        fo.close()

    def collect_group_data(self, types_to_save, data):
        for type in types_to_save:
            self._data[type] = data[type]

    def collect_epos_data(self, data_type, data, i_episode):
        if self._data.get(data_type) is None:
            self._data[data_type] = {}
        self._data[data_type][i_episode] = data

    def collect_data(self, data_type, data):
        self._data[data_type] = data

    def increase_data_value(self, data_type, data_value):
        if self._data.get(data_type) is not None:
            self._data[data_type] += data_value
        else:
            self._data[data_type] = data_value

    def save_something(self, something, name):
        with open(self._model_path + name, mode='wb') as file:
            pickle.dump(something, file)

    def epos_snapshot(self, i_episode):
        if i_episode % self._epos_snap == 0:

            self._model.save_weights(self._weights_path + str(i_episode),
                                     overwrite=True)
            self._model.save(self._model_path, overwrite=True,
                             include_optimizer=True)

            meta = open(self._metadata_path, "a")

            def add_data_sequence(a, b):
                return a + ', ' + b + ': {}'.format(self._meta_data_types_functions[b](self._data))

            meta.write(reduce(add_data_sequence, ['Episode: ' + str(i_episode)] + self._meta_data_types_to_save) + '\n')
            meta.close()

            for epos_type in self._epos_data_types_to_save:
                if self._data.get(epos_type) is not None:
                    with open(self._model_path + '_' + epos_type + '_' + str(i_episode), mode='wb') as file:
                        pickle.dump(self._data[epos_type], file)

            for epos_type in self.epos_data_types_to_reset:
                if self._data.get(epos_type) is not None:
                    self._data.pop(epos_type)
