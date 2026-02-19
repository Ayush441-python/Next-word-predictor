import numpy as np 
from utils import print_metrics

def evaluate_model(model, X_test, y_test):
    y_pred=(model.predict(X_test)).astype(int)
    print_metrics(y_test,y_pred)
    