import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.metrics import metrics_classification



def euclidean(a, b):
    return np.sqrt(np.sum((a - b) **  2))

class LVQ:
    def __init__(self, data: pd.DataFrame, X_val, y_val, target, metric, alpha):
        self.X, self.y, self.weights = self.init_weights(data, target)
        self.X_val, self.y_val = X_val, y_val
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
        for epoch in range(epochs):
            rand_idx = np.random.randint(self.X.shape[0])
            sample_X, sample_y = self.X[rand_idx, :], self.y[rand_idx]
            self.learn(sample_X, sample_y)
            if (epoch % 50) == 0:
                self.verbose(self.X, self.y, epoch, 'train')
                self.verbose(self.X_val, self.y_val, epoch, 'val')

    def verbose(self, X, y, epoch, prompt):
        predictions = np.array(lvq.classify(X))
        print(epoch, prompt, 'acc:', np.sum(predictions == y) / len(predictions), metrics_classification(y, predictions))

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
    wine_train = pd.read_csv('data/preprocessed/wine/train_wine.csv')
    wine_val = pd.read_csv('data/preprocessed/wine/test_wine.csv')
    # Classes should start from 0
    wine_train['class'] = wine_train['class'] - 1
    wine_val['class'] = wine_val['class'] - 1

    X_val = wine_val.drop('class', axis=1).to_numpy()
    y_val = wine_val['class'].to_numpy()

    lvq = LVQ(wine_train, X_val, y_val,  'class', euclidean, 0.01)
    lvq.train(1001)




