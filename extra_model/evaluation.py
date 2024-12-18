import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from utils import calculate_ROC

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


def load_features(features_path):
    data = np.load(features_path)
    X = np.column_stack([data["pa_scores"], data["cc_scores"], data["aa_scores"], data["rz_scores"], data["wos_scores"]])
    return X

# Evaluate individual features - they are extremely expressive, allowing for a simple ranking test instead of learning
def evaluate_separate_features(X, y):
    scores = []
    for i in tqdm(range(X.shape[1]), desc="Evaluating individual features via ranking task measured by ROC-AUC"):
        feature_scores = X[:, i]
        sorted_predictions_eval = np.argsort(-1.0 * feature_scores)
        auc_score = calculate_ROC(sorted_predictions_eval, y)
        scores.append((i, max(auc_score, 1 - auc_score)))
    return scores


def model_based_evaluation(model, X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)

    y_scores = model.predict_proba(X_test)[:, 1]
    sorted_predictions_eval = np.argsort(-1.0 * y_scores)

    auc_score = calculate_ROC(sorted_predictions_eval, y_test)
    return auc_score


if __name__ == "__main__":
    features_path = "features.npz"

    # Load data
    X = load_features(features_path)
    current_min_edges = 3
    curr_vertex_degree_cutoff = 25
    current_delta = 5
    data_source="../SemanticGraph_delta_"+str(current_delta)+"_cutoff_"+str(curr_vertex_degree_cutoff)+"_minedge_"+str(current_min_edges)+".pkl"

    if os.path.isfile(data_source):
        with open(data_source, "rb") as pkl_file:
            _, unconnected_vertex_pairs, unconnected_vertex_pairs_solution, _, _, _, _ = pickle.load(pkl_file)
    y = unconnected_vertex_pairs_solution.copy()

    # Evaluate individual features
    feature_tags = ["Preferential Attachment (PA)",
                    "Clustering Coefficient (CC)",
                    "Adamic-Adar index (AA)",
                    "Ruzicka similarity (RZ)",
                    "Weighted Overlap Score (WOS)"]
    feature_scores = evaluate_separate_features(X, y)
    for feature_idx, auc in feature_scores:
        print(f"{feature_tags[feature_idx]}: ROC AUC = {auc:.4f}")

    print()

    for model, tag in [(LogisticRegression(max_iter=1000, solver='lbfgs'), "Logistic Regression"),
                       (GaussianNB(), "Naive Bayes")]:
        print(f"Evaluating via using in {tag}:")
        model_auc = model_based_evaluation(model, X, y)
        print(f"{tag} (PA + CC + AA + RZ + WOS) ROC AUC = {model_auc:.4f}")
