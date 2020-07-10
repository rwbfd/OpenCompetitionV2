# coding = 'utf-8'
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def get_eval_index(y, y_predict, metric='acc'):
    if metric == 'acc':
        return accuracy_score(y, y_predict)
    elif metric == 'precision':
        return precision_score(y, y_predict)
    elif metric == 'recall':
        return recall_score(y, y_predict)
    elif metric == 'macro_f1':
        return f1_score(y, y_predict, average='macro')
    elif metric == 'micro_f1':
        return f1_score(y, y_predict, average='micro')
    else:
        raise Exception("Not implemented yet.")


def fit_model_and_get_predict(y_train, x_train, x_test):
    cls = LogisticRegression(random_state=0)
    cls.fit(x_train, y_train)
    y_predict = cls.predict(x_test)
    return y_predict
