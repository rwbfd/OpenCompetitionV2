from sklearn.datasets import load_breast_cancer
from cuml.preprocessing.model_selection import train_test_split
from sklearn.model_selection import KFold

kf = KFold(n_splits=3)


bc = load_breast_cancer()
X, y = cudf.DataFrame(bc.data), cudf.Series(bc.target)
x_train, x_test, y_train, y_test = train_test_split(X, y)

df_train = x_train
df_train['label'] = y_train

df_test = x_test
df_test['label'] = y_test

clf = CumlLRFitter()
cvf = CumlSVMFitter()
ckf = CumlKNNFitter()

#clf.search(df_train, df_test)
#clf.search_k_fold(k_fold=kf, data=df_train)
#clf.train_k_fold(k_fold=kf, train_data=df_train, test_data=df_test)

#ckf.search(df_train, df_test)
#ckf.search_k_fold(k_fold=kf, data=df_train)
#ckf.train_k_fold(k_fold=kf, train_data=df_train, test_data=df_test)

#cvf.search(df_train, df_test)
#cvf.search_k_fold(k_fold=kf, data=df_train)
#cvf.train_k_fold(k_fold=kf, train_data=df_train, test_data=df_test)