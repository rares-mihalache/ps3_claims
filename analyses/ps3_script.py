# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import dalex as dx
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc, make_scorer, mean_tweedie_deviance
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_column, load_transform
from ps3.evaluation import evaluate_predictions

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# We divide by exposure because policies are observed for different lengths of
# time (up to one year). A policy observed for a few months has had less time to
# produce claims, so the raw claim amount is not comparable across policies.
# Dividing gives the claim amount per year of cover (the pure premium), and
# using exposure as the sample weight lets longer, more reliable observations
# count more.


df = create_sample_column(df, "IDpol", training_frac=0.8)
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# Add splines for BonusMalus and Density and use a Pipeline.
# include_bias=False in the splines and drop="first" in the one-hot encoder
# leave the GLM's own intercept as the only intercept.
numeric_cols = ["BonusMalus", "Density"]
numeric_transformer = Pipeline(
    [
        ("scale", StandardScaler()),
        ("spline", SplineTransformer(knots="quantile", include_bias=False)),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)
preprocessor.set_output(transform="pandas")
model_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=TweedieDist, l1_ratio=1, fit_intercept=True
            ),
        ),
    ]
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# Splines lower the training deviance from 73.60 to 73.22 and the test
# deviance from 74.66 to 74.44, so the extra flexibility for BonusMalus and
# Density also helps on unseen data. The problem set mentions VehPower, but the
# script already treats VehPower as categorical, so we follow the TODO and use
# Density.
# To check for overfitting we compare train and test deviance. A training
# deviance that keeps falling while the test deviance stays flat or rises means
# the model is fitting noise. Here the gap only grows from 1.06 to 1.22. Choices
# such as the number of knots should be tuned with cross-validation on the
# training set rather than by looking at the test set.

# %%
# Use a GBM instead as an estimator. The Tweedie objective with power 1.5
# matches the loss of the GLMs, and LightGBM handles the categorical columns
# of X_train_t directly.
model_pipeline = Pipeline(
    [("estimate", LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5))]
)
model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# With default settings (learning rate 0.1, 100 trees) the LGBM reaches a much
# lower training deviance (65.00) than the GLMs, but its test deviance is 73.99,
# a gap of 8.99. It clearly overfits.

# %%
# Tune the LGBM to reduce overfitting. To save compute time we only tune
# learning_rate and n_estimators, with 5-fold cross-validation. The score is
# the Tweedie deviance, the same loss we use to compare the models.
param_grid = {
    "estimate__learning_rate": [0.01, 0.05, 0.1],
    "estimate__n_estimators": [50, 100, 200],
}
tweedie_scorer = make_scorer(mean_tweedie_deviance, power=1.5, greater_is_better=False)
cv = GridSearchCV(model_pipeline, param_grid, scoring=tweedie_scorer, cv=5, verbose=1)
cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
print("best parameters:", cv.best_params_)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)

# Cross-validation picks a learning rate of 0.01 with 100 trees. The gap
# between train and test deviance falls from 8.99 to 1.52, so the tuned model
# no longer overfits. Its test deviance (74.00) is lower than for both GLMs
# (74.66 and 74.44) but no better than the default LGBM. It also predicts 9.09m
# of claims on the test set against 9.78m observed, about 7% too low.
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# ===================================================================
# Problem Set 4
# ===================================================================

# %%
# PS4 Ex 1: monotonicity constraint for BonusMalus
# Average claims per BonusMalus group, weighted by exposure: total claim amount
# divided by total exposure, so policies observed for longer count more.
bonus_malus = df_train.groupby("BonusMalus")[["ClaimAmountCut", "Exposure"]].sum()
bonus_malus["avg_claims"] = bonus_malus["ClaimAmountCut"] / bonus_malus["Exposure"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(
    bonus_malus.index,
    bonus_malus["avg_claims"],
    s=bonus_malus["Exposure"] / bonus_malus["Exposure"].max() * 300 + 5,
    alpha=0.6,
)
ax.set(
    title="Average claims per year of exposure by BonusMalus (training set)",
    xlabel="BonusMalus",
    ylabel="Claim amount per year of exposure",
)
plt.show()

# Without a constraint the LGBM can follow the noise in this plot: above about
# 110 many BonusMalus groups have little exposure and some have no claims at
# all, so a flexible model could predict a lower price for a worse claim
# history. That would be unfair to customers and easy to game, so we force the
# predictions to increase with BonusMalus.

# %%
# Constrained LGBM: 1 means predictions may only increase with BonusMalus,
# 0 means no constraint for the other features.
monotone_constraints = [1 if col == "BonusMalus" else 0 for col in X_train_t.columns]
print("monotone constraints:", monotone_constraints)

constrained_lgbm = Pipeline(
    [
        (
            "estimate",
            LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.5,
                monotone_constraints=monotone_constraints,
            ),
        )
    ]
)
# Same grid and scorer as the unconstrained LGBM, so the two are comparable.
cv_constrained = GridSearchCV(
    constrained_lgbm, param_grid, scoring=tweedie_scorer, cv=5, verbose=1
)
cv_constrained.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
print("best parameters:", cv_constrained.best_params_)

df_test["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(
            y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t
        )
        / np.sum(w_train_t)
    )
)
print(
    "testing loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(
            y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t
        )
        / np.sum(w_test_t)
    )
)

# %%
print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained"]),
    )
)

# %%
# Check that the constraint works: take the first policy in the test set, keep
# all its other features fixed, and change only BonusMalus from 50 to 230.
bonus_malus_grid = np.arange(50, 231)
customer = X_test_t.iloc[[0]]
fig, ax = plt.subplots(figsize=(10, 5))
for label, model in [
    ("Unconstrained LGBM", cv.best_estimator_),
    ("Constrained LGBM", cv_constrained.best_estimator_),
]:
    rows = pd.concat([customer] * len(bonus_malus_grid), ignore_index=True)
    rows["BonusMalus"] = bonus_malus_grid
    ax.plot(bonus_malus_grid, model.predict(rows), label=label)
ax.set(
    title="Predicted pure premium for the first test policy as BonusMalus changes",
    xlabel="BonusMalus",
    ylabel="Predicted pure premium",
)
ax.legend()
plt.show()

# %%
# The constrained LGBM has a slightly higher deviance than the unconstrained
# one (train 73.09 vs 72.48, test 74.12 vs 74.00), which is the price of
# restricting the model. In return, its predictions never fall as BonusMalus
# gets worse: for the first test policy the unconstrained model jumps to about
# 223 at a BonusMalus of 62 and drops back to about 155 at 64, while the
# constrained model only rises. It also predicts 9.17m of claims on the test
# set against 9.78m observed, slightly closer than the unconstrained 9.09m.

# %%
# PS4 Ex 2: learning curve of the best constrained LGBM. eval_set makes
# LightGBM score the train and test sets after every tree without training on
# them, so we can see how the loss develops as trees are added.
best_constrained = cv_constrained.best_estimator_
best_constrained.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)],
    estimate__eval_sample_weight=[w_train_t, w_test_t],
    estimate__eval_names=["train", "test"],
)
lgb.plot_metric(best_constrained[-1], figsize=(10, 5))
plt.show()


# At 100 trees the test loss is still falling, so the model has not converged:
# with a learning rate of 0.01, 100 trees are too few and the estimator is
# undertrained rather than overfitting. A longer run keeps improving the test
# loss until roughly 300 trees, but choosing the number of trees from the test
# curve would leak test information. Better options are a larger grid for
# n_estimators in the cross-validation, or early stopping on a validation split
# of the training data. The grid search also only tried 50, 100 and 200 trees
# and scored with an unweighted deviance, which may be why it settled on 100.
# Note that LightGBM's "tweedie" metric is a negative log-likelihood, so its
# values (about 45) are not on the same scale as the deviance above (about 74).

# %%
# PS4 Ex 3: compare the unconstrained and constrained LGBM on the test set
metrics_comparison = pd.concat(
    {
        "LGBM": evaluate_predictions(
            y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t
        )["value"],
        "Constrained LGBM": evaluate_predictions(
            y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t
        )["value"],
    },
    axis=1,
)
print(metrics_comparison)
# The unconstrained LGBM is slightly more accurate (deviance 74.00 vs 74.12,
# lower MAE and RMSE) and ranks risks better (Gini 0.237 vs 0.221). The
# constrained model underpredicts total claims a little less (-6.2% vs -7.0%).
# The constraint costs a small amount of accuracy and ranking power in exchange
# for predictions that never fall as BonusMalus gets worse. RMSE is much larger
# than MAE because a few very large claims dominate the squared errors.

# %%
# PS4 Ex 4: partial dependence plots with DALEX. An explainer wraps a fitted
# model together with the data, so DALEX can vary one feature at a time and
# average the predictions over the other features.
explainer_constrained = dx.Explainer(
    cv_constrained.best_estimator_, X_test_t, y_test_t, label="Constrained LGBM"
)
explainer_unconstrained = dx.Explainer(
    cv.best_estimator_, X_test_t, y_test_t, label="Unconstrained LGBM"
)

# PDPs for the numerical features (BonusMalus, Density)
pdp_num_constrained = explainer_constrained.model_profile(
    variable_type="numerical", random_state=42
)
pdp_num_unconstrained = explainer_unconstrained.model_profile(
    variable_type="numerical", random_state=42
)
pdp_num_constrained.plot(pdp_num_unconstrained)

# PDPs for the categorical features (VehBrand, VehGas, Region, Area, DrivAge,
# VehAge, VehPower)
pdp_cat_constrained = explainer_constrained.model_profile(
    variable_type="categorical", random_state=42
)
pdp_cat_unconstrained = explainer_unconstrained.model_profile(
    variable_type="categorical", random_state=42
)
pdp_cat_constrained.plot(pdp_cat_unconstrained)

# %%
# PS4 Ex 5: Shapley decomposition of one prediction, for the constrained LGBM
# and the initial GLM (t_glm1). The LGBM explainer is the one from Ex 4; the GLM
# explainer also uses the test set as data.
glm_explainer = dx.Explainer(
    t_glm1, X_test_t, y_test_t, label="GLM benchmark", verbose=False
)
explainers = [explainer_constrained, glm_explainer]

# Decompose the prediction for the first policy in the test set. Averaging over
# B = 25 random orderings of the features gives contributions that add up to the
# prediction minus the average prediction.
observation = X_test_t.iloc[[0]]
shap = [
    explainer.predict_parts(observation, type="shap", B=25, random_state=42)
    for explainer in explainers
]
for explainer, parts in zip(explainers, shap):
    print(
        "{}: prediction = {:.2f}, average prediction = {:.2f}".format(
            explainer.label, parts.prediction, parts.intercept
        )
    )
# Rows with B == 0 hold the average over the orderings.
contributions = pd.DataFrame(
    {
        explainer.label: parts.result.query("B == 0").set_index("variable")[
            "contribution"git restore -- "*.pyc"
)
print(contributions.round(2))
# Both decompositions in one plot (opens in the browser).
shap[0].plot(shap[1])

# Both models lower this policy's premium most through BonusMalus = 50
# (-30.06 in the LGBM, -66.42 in the GLM), and most features push in the same
# direction. The GLM's contributions are larger throughout, and VehBrand B12 is
# the only feature with opposite signs (-16.82 in the GLM, +1.54 in the LGBM).
