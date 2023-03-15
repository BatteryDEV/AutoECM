from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize, scale
from sklearn import datasets

iris = datasets.load_iris()
X_all = iris.data
y_all = iris.target

# Do a train/test split, radomly 1/3 of the data is used for testing
# USe skleanr train test split function
from sklearn.model_selection import train_test_split

X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)


def hyperopt_train_test(params):
    X_ = X[:]
    if "normalize" in params:
        if params["normalize"] == 1:
            X_ = normalize(X_)
        del params["normalize"]
    if "scale" in params:
        if params["scale"] == 1:
            X_ = scale(X_)
        del params["scale"]
    clf = DecisionTreeClassifier(**params)
    return cross_val_score(clf, X, y).mean()


space4dt = {
    "max_depth": hp.choice("max_depth", range(1, 20)),
    "max_features": hp.choice("max_features", range(1, 5)),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "scale": hp.choice("scale", [0, 1]),
    "normalize": hp.choice("normalize", [0, 1]),
}


def f(params):
    acc = hyperopt_train_test(params)
    return {"loss": -acc, "status": STATUS_OK}


trials = Trials()

best = fmin(f, space4dt, algo=tpe.suggest, max_evals=300, trials=trials)
print("best:")
print(best)
print("done!")
