from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    return model
