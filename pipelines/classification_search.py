from itertools import product
from models.classification.hist_gb_classifier import train_hist_gb_classifier, predict_probabilities
from evaluation.classification_eval import find_best_threshold

def get_classifier_branch_grid():
    learning_rates = [0.03, 0.05]
    max_leaf_nodes_list = [15, 31]
    min_samples_leaf_list = [20, 50]

    branches = []
    for learning_rate, max_leaf_nodes, min_samples_leaf in product(
        learning_rates, max_leaf_nodes_list, min_samples_leaf_list
    ):
        branches.append({
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
        })
    return branches

def is_better_classifier_result(current_metrics, best_metrics, pr_auc_margin=0.005, brier_margin=0.002):
    current_pr_auc = current_metrics["pr_auc"]
    best_pr_auc = best_metrics["pr_auc"]

    if current_pr_auc > best_pr_auc + pr_auc_margin:
        return True

    if current_pr_auc < best_pr_auc - pr_auc_margin:
        return False

    current_brier = current_metrics["brier_score"]
    best_brier = best_metrics["brier_score"]

    if current_brier < best_brier - brier_margin:
        return True
    if current_brier > best_brier + brier_margin:
        return False

    current_key = (
        current_metrics["f1_at_threshold"],
        current_metrics["recall_at_threshold"],
        current_metrics["precision_at_threshold"],
    )
    best_key = (
        best_metrics["f1_at_threshold"],
        best_metrics["recall_at_threshold"],
        best_metrics["precision_at_threshold"],
    )

    return current_key > best_key

def is_clearly_worse_in_same_direction(current_metrics, previous_metrics, pr_auc_margin=0.005, brier_margin=0.002, f1_margin=1e-4):
    pr_auc_worse = current_metrics["pr_auc"] < previous_metrics["pr_auc"] - pr_auc_margin
    brier_worse = current_metrics["brier_score"] > previous_metrics["brier_score"] + brier_margin
    f1_worse = current_metrics["f1_at_threshold"] < previous_metrics["f1_at_threshold"] - f1_margin

    return pr_auc_worse and brier_worse and f1_worse

def get_stage1_ranking_key(result):
    metrics = result["val_metrics"]
    return (
        metrics["pr_auc"],
        -metrics["brier_score"],
        metrics["f1_at_threshold"],
        metrics["recall_at_threshold"],
        metrics["precision_at_threshold"],
    )

def evaluate_classifier_config(X_train_flat, y_train_w, X_val_flat, y_val_w, params):
    model = train_hist_gb_classifier(
        X_train_flat,
        y_train_w,
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        early_stopping=False,
        verbose=0,
    )

    val_proba = predict_probabilities(model, X_val_flat)
    best_threshold, val_metrics = find_best_threshold(y_val_w, val_proba)

    return {
        "model": model,
        "params": params,
        "threshold": best_threshold,
        "val_metrics": val_metrics,
        "val_proba": val_proba,
    }

def run_staged_classifier_search(X_train_flat, y_train_w, X_val_flat, y_val_w):
    branch_grid = get_classifier_branch_grid()
    print("\nRunning branch-based hyperparameter search on train/validation...")

    stage1_results = []
    best_result = None
    fit_count = 1
    max_possible_fits = len(branch_grid) + 2 * min(3, len(branch_grid))

    for branch in branch_grid:
        print(f"Trying config {fit_count}/{max_possible_fits}...")
        params = {
            "learning_rate": branch["learning_rate"],
            "max_iter": 100,
            "max_leaf_nodes": branch["max_leaf_nodes"],
            "min_samples_leaf": branch["min_samples_leaf"],
        }

        current_result = evaluate_classifier_config(
            X_train_flat, y_train_w, X_val_flat, y_val_w, params
        )
        stage1_results.append(current_result)

        if best_result is None or is_better_classifier_result(
            current_result["val_metrics"], best_result["val_metrics"]
        ):
            best_result = current_result

        fit_count += 1

    top_k = min(3, len(stage1_results))
    top_stage1_results = sorted(
        stage1_results,
        key=get_stage1_ranking_key,
        reverse=True
    )[:top_k]

    print("\nRefining the strongest validation configurations...")
    for base_result in top_stage1_results:
        base_params = base_result["params"]
        previous_result = base_result

        for max_iter in [200, 300]:
            print(f"Trying config {fit_count}/{max_possible_fits}...")
            params = {
                "learning_rate": base_params["learning_rate"],
                "max_iter": max_iter,
                "max_leaf_nodes": base_params["max_leaf_nodes"],
                "min_samples_leaf": base_params["min_samples_leaf"],
            }

            current_result = evaluate_classifier_config(
                X_train_flat, y_train_w, X_val_flat, y_val_w, params
            )

            if is_better_classifier_result(
                current_result["val_metrics"], best_result["val_metrics"]
            ):
                best_result = current_result

            if is_clearly_worse_in_same_direction(
                current_result["val_metrics"], previous_result["val_metrics"]
            ):
                print("Stopping one branch early.")
                fit_count += 1
                break

            previous_result = current_result
            fit_count += 1

    return best_result