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
df = pd.read_csv("../data/bank/bank.csv")
# df = pd.read_csv("../data/boston_housing/housing.csv")

# drop unnecessary attributes
df = df.drop(labels=['default', 'contact', 'day', 'month', 'pdays', 'previous', 'loan', 'poutcome'], axis=1)

# dataset info
# check_df(df)

# get num and cat attributes
cat_features, num_features, cat_but_car = grab_col_names(df)
input_features = num_features + cat_features + cat_but_car


target_name = 'deposit'
pca_outlier_remove = True

if len(cat_features) != 0:
    df[cat_features] = df[cat_features].astype('category')

test_size = 0.20
df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[target_name])


highcorr_features = get_highly_correlated_features(df, num_features, corr_threshold=0.90)
num_features = list(set(num_features) - set(highcorr_features))
df = df.drop(highcorr_features, axis=1, errors='ignore')


t_df = df.copy()
if len(cat_features) > 0:
    t_df = pd.concat([t_df, pd.get_dummies(t_df[cat_features])], axis=1)
    t_df = t_df.drop(cat_features, axis=1)

if pca_outlier_remove:
    pca = PCA(n_components=None)
    pca_df = pd.DataFrame(pca.fit_transform(t_df.drop([target_name], axis=1, errors='ignore')), columns=[f'V{i}'for i in range(t_df.shape[1])])
    pca_df[target_name] = df[target_name]

    q1, q3 = pca_df['V0'].quantile([0.25, 0.75])
    iqr = q3 - q1
    lw = q1 - 1.5 * iqr
    uw = q3 + 1.5 * iqr
    outliers_mask = (pca_df['V0'] > uw) | (pca_df['V0'] < lw)
    pca_df['is_outlier'] = False
    pca_df.loc[outliers_mask, 'is_outlier'] = True
    df = df[~pca_df['is_outlier'].values].reset_index(drop=True)


# df.to_csv('../data/preprocessed/boston_housing/train_boston_housing.csv', index=False)
# test_df.to_csv('../data/preprocessed/boston_housing/test_boston_housing.csv', index=False)

df.to_csv('../data/preprocessed/bank/train_bank.csv', index=False)
test_df.to_csv('../data/preprocessed/bank/test_bank.csv', index=False)


fig, ax = plt.subplots(1, figsize=(10, 10))
sb.scatterplot(data=pca_df, x='V0', y='V1', hue='is_outlier', ax=ax)
plt.tight_layout()


ev_df = pd.DataFrame(zip(range(len(input_features)), pca.explained_variance_ratio_), columns=['component', 'value'])
fig, ax = plt.subplots(1, figsize=(10, 10))
sb.lineplot(data=ev_df, x='component', y='value',ax=ax)
plt.tight_layout()

fig, ax = plt.subplots(1, figsize=(10, 10))
sb.scatterplot(data=pca_df, x='V0', y=target_name, ax=ax)
plt.tight_layout()

fig, ax = plt.subplots(1, figsize=(10, 10))
sb.scatterplot(data=pca_df, x='V0', y='V1', hue=target_name, ax=ax)
plt.tight_layout()