def best(data):
    return max([val for key,val in data['reward'].items()])


def average(data):
    from scipy import mean
    return mean(list(data['reward'].values()))

def std(data):
    from scipy import std
    return std(list(data['reward'].values()))

def overzero(data):
    return len([x for x in data['reward'].values() if x > 0]) / len(data['reward'])

def overone(data):
    return len([x for x in data['reward'].values() if x > 1]) / len(data['reward'])

def get_meta_data_types_to_save():
    return ['Best', 'Average', 'Overzero', 'Overone', 'Std']


def get_meta_data_types_functions():
    return {'Best': best, 'Average': average, 'Overzero': overzero, 'Overone': overone, 'Std': std}
