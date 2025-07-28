# 🚀 Spaceship Titanic - Classification Project

##  Project Overview
This project aims to predict whether a passenger was **Transported** to another dimension during the voyage of the fictional Spaceship Titanic. It uses machine learning techniques to analyze and classify the passengers based on various onboard service features, personal traits, and cabin information.

---

##  Dataset Description
The dataset is provided by Kaggle and includes the following key features:

- PassengerId: Unique identifier
- HomePlanet: Planet the passenger is from
- CryoSleep: Whether the passenger opted for cryosleep during the trip
- Cabin: Cabin number (split into deck, num, side)
- Destination: Destination planet
- Age, VIP: Passenger demographics
- Onboard spending: RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
- Transported: Target variable (1 if transported, 0 if not)
##  Exploratory Data Analysis (EDA)

Key insights:
- Several columns had missing values, especially Cabin, Age, and spending columns.
- Spending features were mostly 0 for passengers in CryoSleep.
- The target (Transported) was nearly balanced.
- Correlation analysis showed VRDeck, Spa, and RoomService were moderately related to transport outcome.

Visuals included:
- Count plots for categorical features
- Histograms for numerical distributions
- Heatmap of numerical correlations


#  Data Preprocessing & Feature Engineering

- Split Cabin into Deck, Num, and Side.
- Filled missing values:
  - Numerical (e.g., Age, expenses): 0 or median
  - Categorical (e.g., Destination): mode or 'Missing'
- Converted CryoSleep, VIP, etc., to integers.
- Encoded categorical features using LabelEncoder or get_dummies.

---

##  Model Selection & Training

Tested models:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- XGBoost Classifier    Evaluation Metric: **Accuracy Score**

	Model	Validation Accuracy
2	Random Forest	0.793560
4	XGBoost	0.784934
3	Support Vector Machine	0.774008
0	Logistic Regression	0.772858
1	Decision Tree	0.741806

 **XGBoost** performed best and was selected for final tuning and prediction.

---

##  Hyperparameter Tuning

Used **RandomizedSearchCV** to optimize XGBoost hyperparameters:
- Parameters tuned: n_estimators, max_depth, learning_rate, gamma, min_child_weight, subsample, etc.
- Used 3-fold cross-validation and accuracy as scoring metric.    ##  Model Evaluation

- Accuracy: ~0.79
- F1 Score, Precision, Recall: Balanced and strong
- ROC Curve and AUC used to evaluate model quality
- Feature Importance from XGBoost helped interpret predictions:
  - VRDeck, RoomService, Age, and HomePlanet were key features.

---

##  Explainable AI

- XGBoost’s built-in feature_importances_ showed top predictive features.
- This helped understand how certain services or demographics impacted the transport decision.   ##  Challenges

- Handling extensive missing data
- Engineering meaningful features (e.g., Cabin)
- Ensuring consistent preprocessing between train and test sets
- Balancing model accuracy and explainability

---

## Teamwork & Collaboration

- Simulated a collaborative pipeline:
  - One module for EDA
  - Another for preprocessing
  - Another for modeling and tuning
- Encouraged clean, modular, and reusable code.

##  Future Work

- Try ensemble models or deep learning (e.g., neural networks)
- Implement SHAP/LIME for advanced explainability
- Build a web app using Flask/Streamlit for real-time predictions
- Use more robust cross-validation (e.g., stratified K-Fold)

---

##  Output

- Final prediction stored in submission.csv with:
  - PassengerId
  - Transported (True/False)
