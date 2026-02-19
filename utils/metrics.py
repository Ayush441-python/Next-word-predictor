from sklearn.metrics import confusion_matrix, classification_report

def print_metrics(y_true, y_pred):
    print("confussion martix :",confusion_matrix(y_true,y_pred))
    print("class report :",classification_report(y_true,y_pred))


    
