from __future__ import annotations

from typing import Dict

from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_classical_models(random_state: int = 42) -> Dict[str, object]:
    try:
        from xgboost import XGBClassifier
    except Exception:
        XGBClassifier = None

    try:
        from catboost import CatBoostClassifier
    except Exception:
        CatBoostClassifier = None

    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    nb = GaussianNB()
    dt = DecisionTreeClassifier(max_depth=None, random_state=random_state)
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=random_state),
    )
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, random_state=random_state),
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(random_state=random_state)

    models = {
        "KNN": knn,
        "NB": nb,
        "DT": dt,
        "LR": lr,
        "SVM": svm,
        "RF": rf,
        "GB": gb,
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=random_state),
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        models["XGB"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    if CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(
            verbose=0,
            random_seed=random_state,
            loss_function="Logloss",
        )

    base_estimators = [
        ("rf", rf),
        ("svm", make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=random_state))),
        ("gb", gb),
        ("knn", knn),
    ]
    models["StackingLR"] = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=random_state),
        n_jobs=-1,
        passthrough=False,
    )
    models["StackingDT"] = StackingClassifier(
        estimators=base_estimators,
        final_estimator=DecisionTreeClassifier(random_state=random_state),
        n_jobs=-1,
        passthrough=False,
    )
    return models
