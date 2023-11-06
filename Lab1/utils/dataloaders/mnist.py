import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(5))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def normilize(df, columns):
    input_data = df[columns].astype(np.float64)
    df[columns] = round((input_data - input_data.min()) / 255, 4)
    return df


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"observations: {dataframe.shape[0]}")
    print(f"variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


# read data
df = pd.read_csv("../../data/MNIST/mnist_test.csv")
target_name = 'label'


# dataset info
check_df(df)


# get num and cat attributes
cat_features, num_features, cat_but_car = grab_col_names(df)
input_features = num_features + cat_features + cat_but_car


if len(cat_features) != 0:
    df[cat_features] = df[cat_features].astype('category')

test_size = 0.10
df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[target_name])


df = normilize(df, df.columns[1:])
test_df = normilize(test_df, test_df.columns[1:])

df.to_csv('../../data/preprocessed/MNIST/train_mnist.csv', index=False)
test_df.to_csv('../../data/preprocessed/MNIST/test_mnist.csv', index=False)