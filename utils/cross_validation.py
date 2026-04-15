from sklearn.model_selection import cross_val_score

def perform_cv(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)

    return {
        "Scores": scores,
        "Mean Accuracy": scores.mean()
    }
