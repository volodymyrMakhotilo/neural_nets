from minisom import MiniSom
from minisom import asymptotic_decay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from numpy import dot
from numpy.linalg import norm

def load_data():
    live = pd.read_csv('data/preprocessed/live/live.csv')
    labels = live.loc[:, live.columns[0]]
    data = live.loc[:, live.columns[1:]]
    return data.to_numpy(), labels.to_numpy()



def main():
    data, labels = load_data()

    features_num = data.shape[1]

    '''
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 decay_function=asymptotic_decay,
                 neighborhood_function='gaussian', topology='rectangular',
                 activation_distance='euclidean', random_seed=None):
    '''

    config = {'x': 3,  # Grid m
              'y': 3,  # Grid n
              'input_len': features_num,  # SOM neuron dimension
              'sigma': 1.0,
              'learning_rate': 0.5,
              'decay_function': asymptotic_decay,
              'neighborhood_function': 'gaussian',
              'topology': 'rectangular',
              'activation_distance': 'euclidean', 'random_seed': None,
              }

    def order_arr(arr, depth=2):
        arr = np.array([row[::-1] for row in arr[::-1].reshape(config['x'], config['y'], depth)]).reshape(-1, depth)
        return arr


    som = MiniSom(**config)

    som.pca_weights_init(data)
    som.train(data, 1000, verbose=False)
    print('quantization_error', som.quantization_error(data))
    print('topographic_error', som.topographic_error(data))

    winner_coordinates = np.array([som.winner(sample) for sample in data]).T

    print(som.labels_map(data, labels))

    print(som.distance_map('mean'))
    #print(np.array(list(np.arange(config['y']))*config['x'])+1)
    #print(np.array(list(np.arange(config['y'], 0, step=-1))*config['x']).reshape(3, 3).T.reshape(-1))
    #print(som.distance_map().reshape(-1))



    winner_cord_tuple = np.array([(cord_i, cord_j) for cord_i, cord_j in zip(winner_coordinates[0], winner_coordinates[1])])
    winner_weight = np.array([som.get_weights()[cord_i, cord_j] for cord_i, cord_j in zip(winner_coordinates[0], winner_coordinates[1])])

    diffs = winner_weight-data
    diffs = diffs / np.max(diffs)
    diff_norm = np.linalg.norm(diffs, axis=1, keepdims=True)


    points_relative_to_neurons = diff_norm * np.random.choice([-1, 1], size=winner_cord_tuple.shape)
    points_relative_to_neurons = points_relative_to_neurons / np.max(points_relative_to_neurons) * 0.25
    points_relative_to_neurons = points_relative_to_neurons + winner_cord_tuple + 1

    points_x = points_relative_to_neurons[:, 0]
    points_y = points_relative_to_neurons[:, 1]

    d, counts = np.unique(winner_cord_tuple, axis=0, return_counts=True)

    print(order_arr(d, depth=2))
    print(order_arr(counts, depth=1).T)
    counts = order_arr(counts, depth=1).T
    counts = counts - np.min(counts)
    counts = counts * 10

    sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    sns.scatterplot(x=np.array(list(np.arange(config['y'], 0, step=-1))*config['x']).reshape(3, 3).T.reshape(-1),
                    y=np.array(list(np.arange(config['y']))*config['x'])+1,
                    hue=1-som.distance_map('mean').reshape(-1),
                    vmax=(0, 1),
                    palette='light:b',
                    size=counts.reshape(-1),
                    sizes=(np.min(counts), np.max(counts)))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    desired_ticks = [0, 1, 2, 3, 4]

    # Add grid lines at the specified ticks
    plt.xticks(desired_ticks)
    plt.yticks(desired_ticks)
    plt.grid(which='both', linestyle='-', linewidth=1, color='gray', alpha=0.5)

    sns.scatterplot(x=points_x, y=points_y)

    plt.show()





if __name__ == '__main__':
    main()