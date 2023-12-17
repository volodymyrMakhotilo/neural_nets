import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def euclidean(a, b):
    return np.sqrt(np.sum((a - b) **  2))

class LVQ:
    def __init__(self, data: pd.DataFrame, target, metric, alpha):
        self.X, self.y, self.weights = self.init_weights(data, target)
        self.metric = metric
        self.alpha = alpha

    def find_bmu(self, sample):
        bmu_metric, bmu_index = np.inf, 0
        for idx, weight in enumerate(self.weights):
            if bmu_metric > self.metric(weight, sample):
                bmu_metric = self.metric(weight, sample)
                bmu_index = idx
        return bmu_index

    def update(self, x,  prediction, true_y):
        if prediction == true_y:
            self.weights[prediction, :] += self.alpha * (x - self.weights[prediction, :])
        else:
            self.weights[prediction, :] -= self.alpha * (x - self.weights[prediction, :])

    def learn(self, sample_X, sample_y):
        bmu_index = self.find_bmu(sample_X)
        self.update(sample_X, bmu_index, sample_y)

    def train(self, epochs):
        for _ in range(epochs):
            rand_idx = np.random.randint(self.X.shape[0])
            sample_X, sample_y = self.X[rand_idx, :], self.y[rand_idx]
            self.learn(sample_X, sample_y)

    def init_weights(self, df, target):
        samples = df.groupby(target).sample(1)
        df = df[~df.index.isin(samples.index)]
        return df.drop(target, axis=1).to_numpy(), df[target].to_numpy(), samples.drop(target, axis=1).to_numpy()

    def classify(self, X_val):
        predictions = []
        for val_row in X_val:
            predictions.append(self.find_bmu(val_row))
        return predictions


if __name__ == '__main__':
    wine_train = pd.read_csv('Lab1/data/preprocessed/wine/train_wine.csv')
    wine_val = pd.read_csv('Lab1/data/preprocessed/wine/test_wine.csv')
    # Classes should start from 0
    wine_train['class'] = wine_train['class'] - 1
    wine_val['class'] = wine_val['class'] - 1

    lvq = LVQ(wine_train, 'class', euclidean, 0.01)
    lvq.train(1000)

    # Помер від кринжу від цього коду
    X_train = wine_train.drop('class', axis=1).to_numpy()
    y_train = wine_train['class'].to_numpy()
    predictions_train = np.array(lvq.classify(X_train))
    print('train:', np.sum(predictions_train == y_train) / len(predictions_train))

    X_val = wine_val.drop('class', axis=1).to_numpy()
    y_val = wine_val['class'].to_numpy()

    predictions = np.array(lvq.classify(X_val))
    print('val:', np.sum(predictions == y_val) / len(predictions))
