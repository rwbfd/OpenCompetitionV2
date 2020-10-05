from model_fitter import LRFitter, KNNFitter
import pandas as pd
from sklearn.datasets import load_breast_cancer


bc = load_breast_cancer()
train, label = bc.data, bc.target
df = pd.DataFrame(train)
df['label'] = label
df_train, df_test = df.iloc[:400, :], df.iloc[400:, :]

lrf = LRFitter()
knnf = KNNFitter()

#params = {'max_iter': 5000}
#lrf.train(df_train, df_test, params)

lrf.search(df_train, df_test)
#knnf.search(df_train, df_test)