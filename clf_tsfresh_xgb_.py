



def predict_plot_cm(model, X, y, le, save=0, str='', save_path="", figname="confusion"):
    """Predict and plot confusion matrix
    Parameters
    ----------
    model: Pipeline
        Pipeline with fitted model
    X: array-like
        Test data
    y: array-like
        Test labels
    le: LabelEncoder
        Label encoder
    str: str
        String describing whether the data is train or test
    save: bool
        Save figure
    save_path: str
        Path to save figure
    figname: str
        Name of figure
    Returns
    -------
    None
    """
    y_pred = model.predict(X)
    plot_cm(y, y_pred, le, save=save, save_path=save_path, figname=figname)
    proportion_correct = accuracy_score(y, y_pred)
    print(f'{str} Accuracy: {proportion_correct:.4f}')
    return y_pred

def load_features_le(train_data_f, test_data_f, le_f):
    """Load precomputed features and label encoder

    Parameters
    ----------
    train_data_f: str
        Path to train data
    test_data_f: str
        Path to test data
    lef: str
        Path to label encoder

    Returns
    -------
    X_train: array-like
        Train data
    y_train: array-like
        Train labels
    X_test: array-like
        Test data
    y_test: array-like
        Test labels
    le: LabelEncoder
        Label encoder
    """

    # Load precomputed features
    train_data = np.loadtxt(train_data_f, delimiter=",")
    test_data = np.loadtxt(test_data_f, delimiter=",")

    # Preprocess data
    X_train = train_data[:,0:-1]
    y_train= train_data[:,-1].reshape(-1,1).ravel()
    X_test = test_data[:,0:-1]
    y_test = test_data[:,-1].reshape(-1,1).ravel()

    # Load label encoder
    with open(le_f, 'r') as f:
        mapping = json.load(f)
        le = LabelEncoder()

    nb_classes = len(mapping.keys())
    mapping['classes'] = [mapping[str(int(i))] for i in range(nb_classes)]
    le.classes_ = np.array(mapping['classes'])
    return X_train, y_train, X_test, y_test, le 

def main(): 
    # If label encodes is not availble (e.g. this script is separated, uncommenitng the folling lines allows to restore the same encoder that encoded the training data)
    with open(f"data/le_name_mapping.json", "r") as f:
        label_dict = json.load(f)
    le = LabelEncoder()
    classes = np.array([label_dict[str(i)] for i in range(9)])
    le.classes_ = classes

    train_data_f = "data/train_tsfresh.csv"
    test_data_f = "data/test_tsfresh.csv"
    le_f = "data/le_name_mapping.json"

    X_train, y_train, X_test, y_test, le = load_features_le(train_data_f, test_data_f, le_f)

    # Create XGBoost model
    model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Make predictions for train data
    y_train_pred = model.predict(X_train)
    y_train_pred = predict_plot_cm(model, X_train, y_train, le, save=save_fig, str='Train', save_path=fig_path, figname="train_confusion")

    # Make predictions for test data
    y_test_pred = model.predict(X_test)
    y_test_pred = predict_plot_cm(model, X_test, y_test, le, save=save_fig, str='Test', save_path=fig_path, figname="test_confusion")

    if save_model:
        dump(model, 'xgb-tsfresh-clf-model.joblib')
    return

if __name__ == '__main__':
    
    main()
