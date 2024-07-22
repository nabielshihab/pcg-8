import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, balanced_accuracy_score

def plot_cm(cm, labels=['Charged Off', 'Fully Paid'], **kwargs):
    """
    plot confusion matrix
    
    Parameters:
    -----------
    cm : {np.array} confusion matrix array
    ticklabels : {list} x and y tick labels
    **kwargs : plotting keyword arguments
    """
    fig, ax = plt.subplots(figsize=(3,3))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax, **kwargs)
    ax.set(title='Confusion Matrix', xlabel='Predicted Label', ylabel='Actual Label', xticklabels=labels, yticklabels=labels)
    
    # extract true positives, etc 
    tn, fp, fn, tp = cm.ravel()
    
    if cm.shape == (2,2):
        print('True Positives\t: ', tp)
        print('True Negatives\t: ', tn)
        print('False Positives\t: ', fp)
        print('False Negatives\t: ', fn)


def append_scores(df, model_name, scores):
    """
    Append model scores to a dataframe.
    The dataframe columns should be 'Model', 'Train Precision', 'Train Bal. Accuracy', 'Test Precision', and 'Test Bal. Accuracy'

    Parameters:
    -----------
    df : {dataframe} a dataframe of metric scores
    model_name : {str} model name
    scores : {dict} dictionary of metric scores
    
    Returns:
    -----------
    df : {dataframe} an updated dataframe
    """
    if model_name in df['Model'].values:
        print('The model already exists')
    else:
        df.loc[df.shape[0], :] = [model_name, scores['Train Precision'], scores['Test Precision']]
    
    return df

def get_scores(y_train, y_train_pred, y_test, y_test_pred):
    """
    extract the precision scores as well as the confusion matrices. 

    Parameters:
    -----------
    y_train : {dataframe} a dataframe of train dataset target values.
    y_train_pred  {dataframe} a dataframe of train dataset predicted values.
    y_test : {dataframe} a dataframe of test dataset target values.
    y_test_pred : {dataframe} a dataframe of test dataset predicted values.
    
    Returns:
    -----------
    dict_scores : {dictionary} a dictionary containing model metric scores plus the confusion matrices.    
    """
    train_precision = precision_score(y_train, y_train_pred)
    cm_train = confusion_matrix(y_train, y_train_pred)

    test_precision = precision_score(y_test, y_test_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    dict_scores = {
        'Train Precision': train_precision,
        'Train CM': cm_train,
        'Test Precision': test_precision,
        'Test CM': cm_test,
    }
    
    return dict_scores
    
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    First, it makes predictions on train and test datasets, then get the metric scores.

    Parameters:
    -----------
    model : a machine learning model
    y_train : {dataframe} a dataframe of train dataset target values.
    X_train  {dataframe} a dataframe of train dataset features.
    y_test : {dataframe} a dataframe of test dataset target values.
    X_test : {dataframe} a dataframe of test dataset features.
    
    Returns:
    -----------
    dict_scores : {dictionary} a dictionary containing model metric scores plus the confusion matrices.
    """
    # make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # get scores
    dict_scores = get_scores(y_train, y_train_pred, y_test, y_test_pred)
    
    return dict_scores


def smooth_curve(points, factor=0.75):
    """
    perform curve smoothing.

    Parameters:
    -----------
    points: {list} curve points
    factor: {float} smoothing factor
    
    Returns:
    --------
    smoothed_points: {list} smoothed curve
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_loss(history):
    """
    plot training and validation loss during model training

    Parameters:
    -----------
    history: {list} keras.callbacks.History
    """
    fig, ax = plt.subplots()
    ax.plot(smooth_curve(history.history['loss']), label='Train loss')
    ax.plot(smooth_curve(history.history['val_loss']), label='Val loss')
    ax.set(xlabel='Epoch', ylabel='Binary crossentropy')
    ax.legend()
    fig.show()


def plot_metric(history, metric='precision'):
    """
    plot training and validation metric during model training

    Parameters:
    -----------
    history: {list} keras.callbacks.History
    metrics: {str} one of the keys in history.history dictionary
    """
    fig, ax = plt.subplots()
    ax.plot(smooth_curve(history.history[metric]), label=f'Train {metric}')
    ax.plot(smooth_curve(history.history[f'val_{metric}']), label=f'Val {metric}')
    ax.set(xlabel='Epoch', ylabel=metric.title())
    ax.legend()
    fig.show()