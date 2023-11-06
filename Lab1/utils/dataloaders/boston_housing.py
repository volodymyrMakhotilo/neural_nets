import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def get_highly_correlated_features(df, num_features, corr_threshold=0.90):
    corr_df = df[num_features].corr(method='pearson')
    mask = np.tri(corr_df.shape[0], k=-1, dtype=bool)
    corr_df = corr_df.where(mask)
    highcorr_df = corr_df[corr_df.abs() > corr_threshold].stack().dropna().reset_index()
    highcorr_df = highcorr_df[highcorr_df['level_0'] != highcorr_df['level_1']]
    highcorr_features = list(set(highcorr_df['level_0'].to_numpy().flatten()))
    return highcorr_features

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
    df[columns] = (input_data - input_data.min()) / (input_data.max() - input_data.min())
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

def PCA_outlier_remove(df, outlier_prop):
    pca = PCA(n_components=2)
    pca_df = pd.DataFrame(pca.fit_transform(t_df.drop([target_name], axis=1, errors='ignore')),
                          columns=[f'V{i}' for i in range(2)])
    pca_df[target_name] = df[target_name].values

    q1_0, q3_0 = pca_df['V0'].quantile([outlier_prop, 1 - outlier_prop])
    iqr_0 = q3_0 - q1_0
    lw_0 = q1_0 - 1.5 * iqr_0
    uw_0 = q3_0 + 1.5 * iqr_0

    q1_1, q3_1 = pca_df['V1'].quantile([outlier_prop, 1 - outlier_prop])
    iqr_1 = q3_1 - q1_1
    lw_1 = q1_1 - 1.5 * iqr_1
    uw_1 = q3_1 + 1.5 * iqr_1

    outliers_mask = (pca_df['V0'] > uw_0) | (pca_df['V0'] < lw_0) | (pca_df['V1'] > uw_1) | (pca_df['V1'] < lw_1)
    pca_df['is_outlier'] = False
    pca_df.loc[outliers_mask, 'is_outlier'] = True

    return df[~pca_df['is_outlier'].values].reset_index(drop=True), pca_df, pca

# read data
df = pd.read_csv("../../data/boston_housing/HousingData.csv")
df = df.dropna()
target_name = 'MEDV'

# # dataset info
# check_df(df)

# get num and cat attributes
cat_features, num_features, cat_but_car = grab_col_names(df)
input_features = num_features + cat_features + cat_but_car


if len(cat_features) != 0:
    df[cat_features] = df[cat_features].astype('category')

test_size = 0.20
df, test_df = train_test_split(df, test_size=test_size, random_state=42)

highcorr_features = get_highly_correlated_features(df, num_features, corr_threshold=0.90)

num_features = list(set(num_features) - set(highcorr_features))
df = df.drop(highcorr_features, axis=1, errors='ignore')


t_df = df.copy()
if len(cat_features) > 0:
    t_df = pd.concat([t_df, pd.get_dummies(t_df[cat_features])], axis=1)
    t_df = t_df.drop(cat_features, axis=1)

df, pca_df, pca = PCA_outlier_remove(df, 0.12)

df = normilize(df, df.columns[:-1])
test_df = normilize(test_df, test_df.columns[:-1])

df.to_csv('../../data/preprocessed/boston_housing/train_boston_housing.csv', index=False)
test_df.to_csv('../../data/preprocessed/boston_housing/test_boston_housing.csv', index=False)



fig, ax = plt.subplots(1, figsize=(10, 10))
sb.scatterplot(data=pca_df, x='V0', y='V1', hue='is_outlier', ax=ax)
plt.tight_layout()

fig, ax = plt.subplots(1, figsize=(10, 10))
sb.heatmap(df.corr(), annot=True)
plt.tight_layout()

plt.show()
