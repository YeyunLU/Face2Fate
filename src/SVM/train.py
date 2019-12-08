from utils import *
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
import numpy as np

SAVE_PATH="data/new_trained_rbf_svms.pkl"
SAVE_TRAIN_DATA_PATH = "data/new_train_data.pkl"

if os.path.isfile(SAVE_TRAIN_DATA_PATH):
    data = joblib.load(SAVE_TRAIN_DATA_PATH)
else:
    data = loadData()
    joblib.dump(data, SAVE_TRAIN_DATA_PATH)

svms = {}

for region_name, features in data.items():
    print("training svm for %s      "% (region_name))

    X = []
    y = []
    for feature_name, feature_shapes in features.items():
        for shape in feature_shapes:
            X.append(shape.flatten())
            y.append(feature_name)

    X = np.squeeze(np.array(X))
    y = np.array(y, dtype='S128')

    print("=================")
    # svms[region_name.encode()] = svm.SVC(kernel="rbf", probability=True, gamma="scale", C=50)
    svms[region_name.encode()] = svm.SVC(kernel="linear", probability=True)

    scores = cross_val_score(svms[region_name.encode()], X, y, cv=5)
    print("Cross val score: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # train for deployment
    svms[region_name.encode()].fit(X, y)

    print("\n")

print("training svm... Done")


# joblib.dump(svms, SAVE_PATH)
print("svm saved!")