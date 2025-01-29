**Day 2: Classical Machine Learning & Statistical Foundations (Extended)**

Below is an expanded set of questions, challenges, and prompts for Day 2, tailored to reinforce your understanding of essential machine learning concepts. By the end of the day, you should be comfortable discussing everything from basic EDA to tree-based models, with a solid grasp on how to evaluate models using various metrics.

---

## 1. Exploratory Data Analysis (EDA)

### Goal

Learn how to explore, visualize, and gain insights from a dataset before modeling. This sets the stage for better feature engineering and understanding data nuances.

### Challenges & Prompts

1. **Data Cleaning & Preprocessing**
   - Identify missing values in a dataset (e.g., Iris, Titanic, or your own CSV).
   - Compare strategies: dropping vs. imputing with mean/median or domain-specific values.
   - Check for outliers and decide whether to remove or cap them.
2. **Descriptive Statistics & Correlation**
   - Calculate mean, median, standard deviation, and percentiles for numerical fields.
   - Visualize relationships (scatter plots, pair plots, correlation heatmaps).
   - Note any strong correlations or suspicious anomalies (e.g., perfect correlation or negative correlation).
3. **Distributions & Visualization**
   - Plot histograms, box plots, violin plots.
   - Identify skew in distributions and think about transformations (log, sqrt) if needed.

### Questions to Answer

- What strategy do you use for missing data in a dataset with 30% of values missing in a single column?
- How do outliers affect mean and standard deviation?
- How would you handle highly skewed variables in a regression model?

---

### 1.1 Data Cleaning & Preprocessing

**EXPLORATION**

**Challenge:** Data Cleaning & Preprocessing

**Objective:** Perform data cleaning and preprocessing on the Titanic dataset to prepare it for modeling.

**Steps & Implementation:**

1. **Identify Missing Values:**

   ```python:data_preprocessing/identify_missing.py
   import pandas as pd

   def identify_missing_values(file_path):
       df = pd.read_csv(file_path)
       missing_values = df.isnull().sum()
       print("Missing Values per Column:")
       print(missing_values)
       return df
   ```

   - **Explanation:**
     - Load the dataset using pandas.
     - Use `isnull().sum()` to count missing values per column.
     - Display the missing values to understand which columns require attention.

2. **Handling Missing Data: Dropping vs. Imputing**

   - **Dropping:**

     - Remove rows or columns with missing values.
     - **Pros:** Simple and quick.
     - **Cons:** Can lead to loss of valuable data, especially if missingness is high.

   - **Imputing:**
     - Fill missing values with statistical measures like mean or median, or use domain-specific values.
     - **Pros:** Retains data, which can be crucial for model performance.
     - **Cons:** Imputing can introduce bias if not done thoughtfully.

   ```python:data_preprocessing/handle_missing.py
   import pandas as pd
   from sklearn.impute import SimpleImputer

   def handle_missing_values(df, strategy='mean'):
       if strategy == 'drop':
           df_cleaned = df.dropna()
       else:
           imputer = SimpleImputer(strategy=strategy)
           numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
           df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
           df_cleaned = df
       return df_cleaned
   ```

   - **Implementation Details:**
     - If the strategy is 'drop', use `dropna()` to remove rows with missing values.
     - Otherwise, use `SimpleImputer` from scikit-learn to fill missing numeric values with the specified strategy (e.g., 'mean' or 'median').

3. **Handling Outliers: Detecting and Capping**

   - **Detection:**

     - Use statistical methods like the Interquartile Range (IQR) to identify outliers.

   - **Capping:**
     - Set thresholds (e.g., 5th and 95th percentiles) to cap extreme values.

   ```python:data_preprocessing/handle_outliers.py
   import pandas as pd
   import numpy as np

   def cap_outliers(df, column, lower_quantile=0.05, upper_quantile=0.95):
       lower = df[column].quantile(lower_quantile)
       upper = df[column].quantile(upper_quantile)
       df[column] = np.where(df[column] < lower, lower, df[column])
       df[column] = np.where(df[column] > upper, upper, df[column])
       return df
   ```

   - **Implementation Details:**
     - Calculate the lower and upper quantiles.
     - Use `np.where` to cap values below the lower quantile and above the upper quantile.

**Considerations:**

- **Choosing Between Dropping and Imputing:**

  - **Dropping** is straightforward but can lead to data loss, especially if many values are missing.
  - **Imputing** retains data but must be done carefully to avoid introducing bias.

- **Impact of Outliers:**
  - Outliers can skew statistical measures and negatively impact model performance.
  - Deciding to cap or remove depends on the context and the nature of the outliers.

**Example Scenario:**

- **High Missingness in 'Age' Column:**

  - **Strategy:** Impute missing ages with the median age to reduce bias and retain data.
  - **Reasoning:** The median is less affected by outliers compared to the mean.

- **Outliers in 'Fare' Column:**
  - **Strategy:** Cap fares at the 95th percentile to prevent extremely high values from skewing the model.

---

### 1.2 Descriptive Statistics & Correlation

**EXPLORATION**

**Challenge:** Descriptive Statistics & Correlation Analysis

**Objective:** Perform descriptive statistical analysis and correlation assessment on the Titanic dataset to uncover relationships between features.

**Steps & Implementation:**

1. **Calculate Descriptive Statistics:**

   ```python:data_analysis/descriptive_statistics.py
   import pandas as pd

   def descriptive_statistics(file_path):
       df = pd.read_csv(file_path)
       stats = df.describe()
       print("Descriptive Statistics:")
       print(stats)
       return df
   ```

   - **Explanation:**
     - Load the dataset.
     - Use `describe()` to compute mean, median (50% percentile), standard deviation, and other percentiles for numerical columns.

2. **Calculate Correlation Matrix:**

   ```python:data_analysis/correlation_matrix.py
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt

   def plot_correlation_matrix(file_path):
       df = pd.read_csv(file_path)
       corr = df.corr()
       plt.figure(figsize=(10, 8))
       sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
       plt.title("Correlation Heatmap")
       plt.show()
   ```

   - **Explanation:**
     - Compute the correlation matrix using `df.corr()`.
     - Visualize the correlations using seaborn's heatmap for better interpretability.

3. **Scatter Plots and Pair Plots:**

   ```python:data_analysis/pair_plots.py
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt

   def create_pair_plots(file_path):
       df = pd.read_csv(file_path)
       sns.pairplot(df, hue='Survived', diag_kind='kde')
       plt.show()
   ```

   - **Explanation:**
     - Use seaborn's `pairplot` to create scatter plots for pairs of features.
     - Color-code the plots based on the 'Survived' column to observe patterns.

**Insights:**

- **Strong Correlations:**
  - For example, 'Fare' might be positively correlated with 'Pclass' (lower class fares higher).
- **Negative Correlations:**

  - 'Age' might have a slight negative correlation with 'Survived' if younger passengers had higher survival rates.

- **Anomalies:**
  - If any feature shows a perfect negative or positive correlation with another, it might indicate redundancy.

**Considerations:**

- **Correlation vs. Causation:**

  - High correlation does not imply causation; further analysis is needed to understand the underlying relationships.

- **Multicollinearity:**
  - Features with high correlation can lead to multicollinearity issues in regression models, affecting coefficient estimates.

---

### 1.3 Distributions & Visualization

**EXPLORATION**

**Challenge:** Distributions & Visualization

**Objective:** Analyze the distribution of key features and apply transformations to handle skewness, enhancing model performance.

**Steps & Implementation:**

1. **Plotting Histograms and Box Plots:**

   ```python:data_visualization/distributions.py
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt

   def plot_distributions(file_path):
       df = pd.read_csv(file_path)
       numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

       for col in numerical_cols:
           plt.figure(figsize=(12, 5))

           # Histogram
           plt.subplot(1, 2, 1)
           sns.histplot(df[col].dropna(), kde=True)
           plt.title(f'Histogram of {col}')

           # Box Plot
           plt.subplot(1, 2, 2)
           sns.boxplot(x=df[col])
           plt.title(f'Box Plot of {col}')

           plt.show()
   ```

   - **Explanation:**
     - Iterate through numerical columns.
     - For each, plot a histogram with KDE to visualize the distribution.
     - Plot a box plot to identify outliers.

2. **Identifying Skewness:**

   ```python:data_visualization/skewness.py
   import pandas as pd

   def calculate_skewness(file_path):
       df = pd.read_csv(file_path)
       skewness = df.skew()
       print("Skewness of Numerical Features:")
       print(skewness)
       return skewness
   ```

   - **Explanation:**
     - Use `df.skew()` to quantify the skewness of numerical features.
     - Values significantly different from 0 indicate skewed distributions.

3. **Applying Transformations:**

   ```python:data_preprocessing/transformations.py
   import pandas as pd
   import numpy as np

   def transform_skewed_features(df, columns):
       for col in columns:
           if df[col].skew() > 1:
               df[col] = np.log1p(df[col])
           elif df[col].skew() < -1:
               df[col] = np.expm1(df[col])
       return df
   ```

   - **Explanation:**
     - For positively skewed features (skewness > 1), apply log transformation.
     - For negatively skewed features (skewness < -1), apply exponential transformation.
     - `log1p` and `expm1` are used to handle zero and negative values gracefully.

**Considerations:**

- **Transformation Impact:**

  - Log transformations can stabilize variance and make relationships more linear.
  - Exponential transformations are less common but can be useful in specific scenarios.

- **Feature Engineering:**

  - Beyond transformations, consider creating new features that capture the essence of skewed data.

- **Model Compatibility:**
  - Some models, like linear regression, benefit more from transformed features compared to tree-based models.

---

### Answers to Questions

1. **What strategy do you use for missing data in a dataset with 30% of values missing in a single column?**

   **Answer:**

   - **Assessment:**

     - First, analyze the nature and distribution of the missing data.
     - Determine if the missingness is random or has a pattern (Missing Completely at Random, Missing at Random, or Missing Not at Random).

   - **Strategy Selection:**

     - **Imputation:** Given that 30% of values are missing in a single column, imputation is preferable to dropping, as dropping would result in significant data loss.
       - **For Numerical Data:** Use median imputation to reduce the impact of outliers.
       - **For Categorical Data:** Use the mode or a new category like 'Unknown'.
     - **Advanced Imputation:** Consider using model-based imputation methods like K-Nearest Neighbors (KNN) or Multiple Imputation by Chained Equations (MICE) for better accuracy.

   - **Implementation Example:**

     ```python:data_preprocessing/impute_missing_values.py
     import pandas as pd
     from sklearn.impute import SimpleImputer

     def impute_missing_values(df, column, strategy='median'):
         imputer = SimpleImputer(strategy=strategy)
         df[[column]] = imputer.fit_transform(df[[column]])
         return df
     ```

2. **How do outliers affect mean and standard deviation?**

   **Answer:**

   - **Impact on Mean:**

     - Outliers can significantly skew the mean, pulling it towards the extreme values.
     - This makes the mean a less reliable measure of central tendency in the presence of outliers.

   - **Impact on Standard Deviation:**

     - Outliers increase the standard deviation, indicating greater variability in the data.
     - This can give a misleading impression of data spread if outliers are not representative of the general data distribution.

   - **Example:**

     ```python:data_analysis/outliers_effect.py
     import numpy as np

     data = [10, 12, 12, 13, 12, 11, 12, 100]
     mean = np.mean(data)
     std_dev = np.std(data)

     print(f"Mean: {mean}")        # Mean: 19.75
     print(f"Std Dev: {std_dev}")  # Std Dev: 29.806

     # Without the outlier
     data_clean = [10, 12, 12, 13, 12, 11, 12]
     mean_clean = np.mean(data_clean)
     std_dev_clean = np.std(data_clean)

     print(f"Mean without outlier: {mean_clean}")        # Mean: 11.714285714285714
     print(f"Std Dev without outlier: {std_dev_clean}")  # Std Dev: 1.2979585430679193
     ```

   - **Conclusion:**
     - Outliers can distort statistical measures, necessitating strategies to handle them appropriately.

3. **How would you handle highly skewed variables in a regression model?**

   **Answer:**

   - **Skewness Challenges:**

     - Highly skewed variables can lead to models that assume normality, affecting the performance and interpretability.
     - It can impact the residuals and violate regression assumptions.

   - **Handling Strategies:**

     1. **Transformation:**

        - **Log Transformation:** Effective for right-skewed data.

          ```python:data_preprocessing/log_transform.py
          import numpy as np

          def log_transform(df, column):
              df[column] = np.log1p(df[column])
              return df
          ```

        - **Square Root Transformation:** Useful for moderate skewness.
        - **Box-Cox Transformation:** More flexible, applicable for positive data.

          ```python:data_preprocessing/box_cox_transform.py
          import pandas as pd
          from scipy import stats

          def box_cox_transform(df, column):
              df[column], _ = stats.boxcox(df[column] + 1)  # Adding 1 to handle zeros
              return df
          ```

     2. **Removal of Outliers:**
        - Cap or remove extreme values to reduce skewness.
     3. **Binning:**
        - Transform continuous variables into categorical by binning, though it may lead to loss of information.
     4. **Using Robust Models:**
        - Models like tree-based methods are less sensitive to skewed distributions.

   - **Choosing the Right Strategy:**

     - **Evaluate Impact:** After transformation, check skewness again to ensure improvement.
     - **Model Requirements:** Some models require normally distributed features, while others do not.

   - **Implementation Example:**

     ```python:data_preprocessing/handle_skewed_variables.py
     import pandas as pd
     import numpy as np
     from scipy import stats

     def transform_skewed(df, column):
         skewness = df[column].skew()
         if skewness > 1:
             df[column] = np.log1p(df[column])
         elif skewness < -1:
             df[column] = np.expm1(df[column])
         return df
     ```

---

## 2. Regression & Classification Basics

### Goal

Refresh the core concepts of linear vs. logistic regression, along with classification metrics that interviewers often focus on.

### Challenges & Prompts

1. **Logistic Regression from (Near) Scratch**
   - Conceptualize the gradient descent steps for a simple binary classification problem.
   - Compare manual gradient checking vs. library outputs (e.g., scikit-learn).
   - Integrate regularization (L1/L2) and observe how it impacts coefficients.
2. **Metrics & Model Evaluation**
   - Manually compute precision, recall, F1-score, accuracy, and ROC-AUC.
   - For a given confusion matrix, show how each metric changes if you tweak the decision threshold.
   - Use k-fold cross-validation to see metric variance.
3. **Bias-Variance Intuition**
   - Explore how an overly complex model (e.g., high-degree polynomial regression) might overfit.
   - Notice changes in training error vs. validation error.

### Questions to Answer

- How does logistic regression differ from linear regression on a conceptual level?
- Why might you choose F1-score over accuracy in an imbalanced classification problem?
- When is a high recall more important than a high precision (and vice versa)?
- Can you explain the difference between using a validation set vs. a test set?

---

### 2.1 Logistic Regression from (Near) Scratch

**EXPLORATION**

**Challenge:** Logistic Regression from (Near) Scratch

**Objective:** Implement logistic regression manually and compare it with scikit-learn's implementation. Integrate regularization and observe its impact.

**Steps & Implementation:**

1. **Understanding Logistic Regression:**

   - **Concept:** Logistic regression models the probability of a binary outcome using the logistic function.
   - **Equation:**
     \[
     \text{Probability} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
     \]
   - **Loss Function:** Binary Cross-Entropy
     \[
     \text{Loss} = -\frac{1}{N} \sum\_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]
     \]

2. **Manual Implementation Using Gradient Descent:**

   ```python:ml/gradient_descent_logistic.py
   import numpy as np

   class LogisticRegressionGD:
       def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_=0.01):
           self.learning_rate = learning_rate
           self.num_iterations = num_iterations
           self.regularization = regularization
           self.lambda_ = lambda_
           self.weights = None
           self.bias = None

       def sigmoid(self, z):
           return 1 / (1 + np.exp(-z))

       def fit(self, X, y):
           n_samples, n_features = X.shape
           self.weights = np.zeros(n_features)
           self.bias = 0

           for _ in range(self.num_iterations):
               linear_model = np.dot(X, self.weights) + self.bias
               y_predicted = self.sigmoid(linear_model)

               dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))

               db = (1 / n_samples) * np.sum(y_predicted - y)

               if self.regularization == 'l2':
                   dw += (self.lambda_ / n_samples) * self.weights
               elif self.regularization == 'l1':
                   dw += (self.lambda_ / n_samples) * np.sign(self.weights)

               self.weights -= self.learning_rate * dw
               self.bias -= self.learning_rate * db

       def predict_proba(self, X):
           linear_model = np.dot(X, self.weights) + self.bias
           return self.sigmoid(linear_model)

       def predict(self, X, threshold=0.5):
           proba = self.predict_proba(X)
           return np.array([1 if i > threshold else 0 for i in proba])
   ```

   - **Explanation:**
     - Initialize weights and bias to zeros.
     - For each iteration:
       - Compute the linear combination \( w^T x + b \).
       - Apply the sigmoid function to get predicted probabilities.
       - Compute gradients for weights and bias.
       - Apply regularization if specified.
       - Update weights and bias using the gradients and learning rate.

3. **Comparing with scikit-learn:**

   ```python:ml/compare_with_sklearn.py
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score
   from sklearn.linear_model import LogisticRegression
   from ml.gradient_descent_logistic import LogisticRegressionGD

   # Load dataset
   import pandas as pd
   from sklearn.datasets import load_breast_cancer

   data = load_breast_cancer()
   X = data.data
   y = data.target

   # Split into train and test
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Initialize and fit manual logistic regression
   model_manual = LogisticRegressionGD(learning_rate=0.1, num_iterations=3000, regularization='l2', lambda_=0.1)
   model_manual.fit(X_train, y_train)
   predictions_manual = model_manual.predict(X_test)

   # Initialize and fit scikit-learn's logistic regression
   model_sklearn = LogisticRegression(C=1/model_manual.lambda_, solver='lbfgs', max_iter=3000)
   model_sklearn.fit(X_train, y_train)
   predictions_sklearn = model_sklearn.predict(X_test)

   # Compare accuracy
   accuracy_manual = accuracy_score(y_test, predictions_manual)
   accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

   print(f"Manual Logistic Regression Accuracy: {accuracy_manual}")
   print(f"scikit-learn Logistic Regression Accuracy: {accuracy_sklearn}")
   ```

   - **Explanation:**
     - Load the Breast Cancer dataset from scikit-learn.
     - Split the data into training and testing sets.
     - Train the manually implemented logistic regression model.
     - Train scikit-learn's logistic regression model with equivalent regularization parameters.
     - Compare the accuracy of both models to validate the manual implementation.

4. **Observing the Impact of Regularization:**

   - **Without Regularization:**

     ```python:ml/no_regularization.py
     model_manual_no_reg = LogisticRegressionGD(learning_rate=0.1, num_iterations=3000)
     model_manual_no_reg.fit(X_train, y_train)
     predictions_no_reg = model_manual_no_reg.predict(X_test)
     accuracy_no_reg = accuracy_score(y_test, predictions_no_reg)
     print(f"Manual Logistic Regression without Regularization Accuracy: {accuracy_no_reg}")
     ```

   - **With L2 Regularization:**

     ```python:ml/l2_regularization.py
     model_manual_l2 = LogisticRegressionGD(learning_rate=0.1, num_iterations=3000, regularization='l2', lambda_=0.1)
     model_manual_l2.fit(X_train, y_train)
     predictions_l2 = model_manual_l2.predict(X_test)
     accuracy_l2 = accuracy_score(y_test, predictions_l2)
     print(f"Manual Logistic Regression with L2 Regularization Accuracy: {accuracy_l2}")
     ```

   - **With L1 Regularization:**

     ```python:ml/l1_regularization.py
     model_manual_l1 = LogisticRegressionGD(learning_rate=0.1, num_iterations=3000, regularization='l1', lambda_=0.1)
     model_manual_l1.fit(X_train, y_train)
     predictions_l1 = model_manual_l1.predict(X_test)
     accuracy_l1 = accuracy_score(y_test, predictions_l1)
     print(f"Manual Logistic Regression with L1 Regularization Accuracy: {accuracy_l1}")
     ```

   - **Observations:**
     - Regularization can prevent overfitting by penalizing large coefficients.
     - L2 tends to distribute the penalty across all coefficients, while L1 can drive some coefficients to zero, effectively performing feature selection.

**Conclusion:**

- Manual implementation of logistic regression provides deeper insight into the mechanics of the algorithm.
- Regularization is crucial for controlling model complexity and preventing overfitting.
- Comparing with scikit-learn's implementation validates the correctness of the manual approach.

---

### 2.2 Metrics & Model Evaluation

**EXPLORATION**

**Challenge:** Metrics & Model Evaluation

**Objective:** Manually compute classification metrics and understand how they change with different decision thresholds. Use k-fold cross-validation to assess metric variance.

**Steps & Implementation:**

1. **Manually Computing Metrics:**

   ```python:metrics/manual_metrics.py
   import numpy as np

   def compute_confusion_matrix(y_true, y_pred):
       TP = np.sum((y_true == 1) & (y_pred == 1))
       TN = np.sum((y_true == 0) & (y_pred == 0))
       FP = np.sum((y_true == 0) & (y_pred == 1))
       FN = np.sum((y_true == 1) & (y_pred == 0))
       return TP, TN, FP, FN

   def precision(TP, FP):
       return TP / (TP + FP) if (TP + FP) != 0 else 0

   def recall(TP, FN):
       return TP / (TP + FN) if (TP + FN) != 0 else 0

   def f1_score(precision, recall):
       return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

   def accuracy(TP, TN, FP, FN):
       return (TP + TN) / (TP + TN + FP + FN)

   def roc_auc(y_true, y_scores, thresholds):
       TPR = []
       FPR = []
       for thresh in thresholds:
           y_pred = (y_scores >= thresh).astype(int)
           TP, TN, FP, FN = compute_confusion_matrix(y_true, y_pred)
           TPR.append(recall(TP, FN))
           FPR.append(FP / (FP + TN) if (FP + TN) != 0 else 0)
       return TPR, FPR
   ```

   - **Explanation:**
     - Compute True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
     - Calculate precision, recall, F1-score, and accuracy based on TP, TN, FP, and FN.
     - For ROC-AUC, iterate over thresholds to compute True Positive Rate (TPR) and False Positive Rate (FPR).

2. **Adjusting Decision Threshold:**

   ```python:metrics/threshold_analysis.py
   import numpy as np
   from sklearn.metrics import roc_auc_score
   from metrics.manual_metrics import compute_confusion_matrix, precision, recall, f1_score, accuracy, roc_auc

   def threshold_analysis(y_true, y_scores, threshold):
       y_pred = (y_scores >= threshold).astype(int)
       TP, TN, FP, FN = compute_confusion_matrix(y_true, y_pred)
       prec = precision(TP, FP)
       rec = recall(TP, FN)
       f1 = f1_score(prec, rec)
       acc = accuracy(TP, TN, FP, FN)
       return prec, rec, f1, acc

   # Example usage
   y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
   y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.76, 0.3, 0.85, 0.5])

   thresholds = [0.3, 0.5, 0.7]
   for thresh in thresholds:
       prec, rec, f1, acc = threshold_analysis(y_true, y_scores, thresh)
       print(f"Threshold: {thresh}")
       print(f"Precision: {prec:.2f}, Recall: {rec:.2f}, F1-Score: {f1:.2f}, Accuracy: {acc:.2f}\n")
   ```

   - **Explanation:**
     - Vary the decision threshold and observe changes in precision, recall, F1-score, and accuracy.
     - Helps in understanding the trade-offs between different metrics.

3. **k-Fold Cross-Validation:**

   ```python:cv/k_fold_cv.py
   import numpy as np
   from sklearn.model_selection import KFold
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score
   import pandas as pd

   def k_fold_cross_validation(file_path, k=5):
       df = pd.read_csv(file_path)
       X = df.drop('Survived', axis=1)
       y = df['Survived']

       kf = KFold(n_splits=k, shuffle=True, random_state=42)
       accuracies = []

       for train_index, test_index in kf.split(X):
           X_train, X_test = X.iloc[train_index], X.iloc[test_index]
           y_train, y_test = y.iloc[train_index], y.iloc[test_index]

           model = LogisticRegression(max_iter=1000)
           model.fit(X_train, y_train)
           predictions = model.predict(X_test)
           acc = accuracy_score(y_test, predictions)
           accuracies.append(acc)

       print(f"{k}-Fold Cross-Validation Accuracies: {accuracies}")
       print(f"Average Accuracy: {np.mean(accuracies):.2f}")
   ```

   - **Explanation:**
     - Split the dataset into k folds.
     - For each fold, train the model on k-1 folds and test on the remaining fold.
     - Calculate and store accuracy for each fold to assess model stability.

**Considerations:**

- **Threshold Selection:**

  - Lowering the threshold increases recall but decreases precision, and vice versa.
  - Choose threshold based on the specific application needs (e.g., medical diagnostics might prioritize recall).

- **Cross-Validation:**
  - Provides a more robust estimate of model performance compared to a single train-test split.
  - Helps in identifying variance in model performance across different subsets of data.

---

### 2.3 Bias-Variance Intuition

**EXPLORATION**

**Challenge:** Bias-Variance Intuition

**Objective:** Understand the bias-variance tradeoff and its impact on model performance through experiments.

**Steps & Implementation:**

1. **Understanding Bias and Variance:**

   - **Bias:** Error due to overly simplistic models that fail to capture underlying patterns.
   - **Variance:** Error due to models that are too complex and capture noise in the training data.
   - **Tradeoff:** Balancing bias and variance to minimize total error.

2. **Experiment with Model Complexity:**

   ```python:ml/bias_variance_experiment.py
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.linear_model import LinearRegression
   from sklearn.preprocessing import PolynomialFeatures
   from sklearn.metrics import mean_squared_error
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import make_regression

   # Generate synthetic data
   X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

   degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   train_errors = []
   val_errors = []

   for degree in degrees:
       poly = PolynomialFeatures(degree)
       X_train_poly = poly.fit_transform(X_train)
       X_val_poly = poly.transform(X_val)

       model = LinearRegression()
       model.fit(X_train_poly, y_train)

       y_train_pred = model.predict(X_train_poly)
       y_val_pred = model.predict(X_val_poly)

       mse_train = mean_squared_error(y_train, y_train_pred)
       mse_val = mean_squared_error(y_val, y_val_pred)

       train_errors.append(mse_train)
       val_errors.append(mse_val)

   plt.figure(figsize=(10, 6))
   plt.plot(degrees, train_errors, label='Training Error')
   plt.plot(degrees, val_errors, label='Validation Error')
   plt.xlabel('Polynomial Degree')
   plt.ylabel('Mean Squared Error')
   plt.title('Bias-Variance Tradeoff')
   plt.legend()
   plt.show()
   ```

   - **Explanation:**
     - Generate synthetic regression data with noise.
     - Fit polynomial regression models of varying degrees.
     - Plot training and validation errors to observe overfitting and underfitting.

3. **Observations:**
   - **Low Degree (e.g., 1):** High bias, low variance (underfitting).
   - **High Degree (e.g., 10):** Low bias, high variance (overfitting).
   - **Optimal Degree (e.g., 3):** Balances bias and variance for minimal error.

**Conclusion:**

- **Bias-Variance Tradeoff:** Critical in model selection and tuning.
- **Model Complexity:** Must balance to avoid underfitting and overfitting.
- **Techniques to Manage:** Regularization, cross-validation, and choosing the right model complexity.

---

### Answers to Questions

1. **How does logistic regression differ from linear regression on a conceptual level?**

   **Answer:**

   - **Purpose:**

     - **Linear Regression:** Predicts a continuous outcome variable.
     - **Logistic Regression:** Predicts the probability of a binary outcome.

   - **Model Equation:**

     - **Linear Regression:** \( y = w^T x + b \)
     - **Logistic Regression:** \( P(y=1) = \sigma(w^T x + b) \), where \( \sigma \) is the sigmoid function.

   - **Loss Function:**

     - **Linear Regression:** Mean Squared Error (MSE).
     - **Logistic Regression:** Binary Cross-Entropy Loss.

   - **Output Interpretation:**

     - **Linear Regression:** Direct continuous value.
     - **Logistic Regression:** Probability value between 0 and 1, which can be thresholded to obtain class labels.

   - **Use Cases:**
     - **Linear Regression:** House price prediction, stock price forecasting.
     - **Logistic Regression:** Email spam detection, medical diagnosis (disease/no disease).

2. **Why might you choose F1-score over accuracy in an imbalanced classification problem?**

   **Answer:**

   - **Imbalanced Classes:** When one class significantly outnumbers the other (e.g., fraud detection), accuracy can be misleading.
   - **Accuracy Limitation:** A model that always predicts the majority class can achieve high accuracy but fails to capture the minority class.
   - **F1-Score Advantage:**

     - Combines precision and recall into a single metric.
     - Provides a balance between false positives and false negatives.
     - More informative when the class distribution is uneven.

   - **Example:**
     - In fraud detection, correctly identifying fraudulent transactions (minority class) is critical.
     - High F1-score ensures that the model maintains a balance between detecting frauds (recall) and not flagging legitimate transactions as fraud (precision).

3. **When is a high recall more important than a high precision (and vice versa)?**

   **Answer:**

   - **High Recall Importance:**

     - **Scenarios:** Medical diagnoses, security threat detection.
     - **Reason:** Missing a positive instance can have severe consequences (e.g., failing to detect a disease).
     - **Trade-Off:** May increase false positives, but ensures most actual positives are captured.

   - **High Precision Importance:**
     - **Scenarios:** Email spam filtering, recommendation systems.
     - **Reason:** False positives can annoy users or reduce system trust.
     - **Trade-Off:** May miss some positives, but ensures that the positives found are reliable.

4. **Can you explain the difference between using a validation set vs. a test set?**

   **Answer:**

   - **Validation Set:**

     - **Purpose:** Used during model training to tune hyperparameters and make decisions about model architecture.
     - **Usage:** Helps in assessing the model's performance and guiding iterative improvements.
     - **Frequency:** Accessed multiple times during the training process.

   - **Test Set:**

     - **Purpose:** Provides an unbiased evaluation of the final model's performance.
     - **Usage:** Used once after the model has been fully trained and validated.
     - **Frequency:** Accessed only once to simulate real-world performance.

   - **Key Differences:**

     - **Validation Set:** For model tuning and selection.
     - **Test Set:** For final evaluation and performance reporting.

   - **Avoiding Data Leakage:**
     - Ensuring the test set remains untouched during training to provide a true measure of model generalization.

---

## 3. Tree-Based Models

### Goal

Understand decision trees and random forests—often asked about in interviews due to their interpretability and good baseline performance.

### Challenges & Prompts

1. **Basic Decision Tree**
   - Train and compare performance with logistic regression on a small dataset.
   - Investigate parameters like max_depth, min_samples_split, or criterion (entropy vs. Gini).
2. **Random Forest**
   - Assess how bagging (bootstrap aggregation) helps reduce variance.
   - Compare results with a single decision tree.
3. **Interpretability & Overfitting**
   - Notice if your decision tree overfits, especially with few training examples.
   - Check feature importances in both decision trees and random forests.

### Questions to Answer

- Why do decision trees overfit if they are grown too deep?
- What's the importance of random feature selection in a random forest?
- How do ensemble methods (like random forests) address the bias-variance tradeoff?

---

### 3.1 Basic Decision Tree

**EXPLORATION**

**Challenge:** Basic Decision Tree

**Objective:** Train a decision tree classifier and compare its performance with logistic regression on the Titanic dataset. Investigate the impact of different parameters.

**Steps & Implementation:**

1. **Training Decision Tree vs. Logistic Regression:**

   ```python:ml/decision_tree_vs_logistic.py
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score, classification_report

   # Load dataset
   df = pd.read_csv('data/titanic_processed.csv')

   # Features and target
   X = df.drop('Survived', axis=1)
   y = df['Survived']

   # Train-Test Split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Logistic Regression
   log_reg = LogisticRegression(max_iter=1000)
   log_reg.fit(X_train, y_train)
   y_pred_log = log_reg.predict(X_test)
   acc_log = accuracy_score(y_test, y_pred_log)
   print(f"Logistic Regression Accuracy: {acc_log:.2f}")

   # Decision Tree
   tree = DecisionTreeClassifier(random_state=42)
   tree.fit(X_train, y_train)
   y_pred_tree = tree.predict(X_test)
   acc_tree = accuracy_score(y_test, y_pred_tree)
   print(f"Decision Tree Accuracy: {acc_tree:.2f}")

   # Classification Reports
   print("\nLogistic Regression Report:")
   print(classification_report(y_test, y_pred_log))

   print("\nDecision Tree Report:")
   print(classification_report(y_test, y_pred_tree))
   ```

   - **Explanation:**
     - Compare the accuracy and classification reports of logistic regression and decision tree classifiers.
     - Observe differences in precision, recall, and F1-scores across classes.

2. **Investigating Parameters:**

   ```python:ml/tuning_decision_tree.py
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.metrics import classification_report

   def tune_decision_tree(X_train, y_train, X_test, y_test):
       parameters = {
           'max_depth': [3, 5, 7, None],
           'min_samples_split': [2, 5, 10],
           'criterion': ['gini', 'entropy']
       }

       for max_depth in parameters['max_depth']:
           for min_samples_split in parameters['min_samples_split']:
               for criterion in parameters['criterion']:
                   tree = DecisionTreeClassifier(max_depth=max_depth,
                                                 min_samples_split=min_samples_split,
                                                 criterion=criterion,
                                                 random_state=42)
                   tree.fit(X_train, y_train)
                   y_pred = tree.predict(X_test)
                   acc = accuracy_score(y_test, y_pred)
                   print(f"max_depth={max_depth}, min_samples_split={min_samples_split}, criterion={criterion} => Accuracy: {acc:.2f}")
                   print(classification_report(y_test, y_pred))
   ```

   - **Explanation:**
     - Iterate over different combinations of `max_depth`, `min_samples_split`, and `criterion`.
     - Observe how these parameters affect model performance and interpretability.

3. **Observations:**
   - **Max Depth:**
     - Shallow trees may underfit, capturing only general patterns.
     - Deeper trees can capture more nuances but risk overfitting.
   - **Min Samples Split:**
     - Higher values prevent the tree from creating nodes with too few samples, reducing overfitting.
   - **Criterion:**
     - 'Gini' and 'entropy' may yield similar results but can influence the tree structure slightly differently.

**Conclusion:**

- **Decision Trees vs. Logistic Regression:**

  - Decision trees can capture non-linear relationships without feature engineering.
  - Logistic regression is simpler and interpretable but assumes linearity between features and the log-odds of the target.

- **Parameter Tuning:**
  - Critical for balancing bias and variance.
  - Proper tuning prevents overfitting and enhances generalization.

---

### 3.2 Random Forest

**EXPLORATION**

**Challenge:** Random Forest

**Objective:** Train a random forest classifier, assess how bagging reduces variance, and compare its performance with a single decision tree.

**Steps & Implementation:**

1. **Training Single Decision Tree vs. Random Forest:**

   ```python:ml/random_forest_vs_tree.py
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, classification_report

   # Load dataset
   df = pd.read_csv('data/titanic_processed.csv')

   # Features and target
   X = df.drop('Survived', axis=1)
   y = df['Survived']

   # Train-Test Split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Single Decision Tree
   tree = DecisionTreeClassifier(max_depth=5, random_state=42)
   tree.fit(X_train, y_train)
   y_pred_tree = tree.predict(X_test)
   acc_tree = accuracy_score(y_test, y_pred_tree)
   print(f"Decision Tree Accuracy: {acc_tree:.2f}")

   # Random Forest
   rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
   rf.fit(X_train, y_train)
   y_pred_rf = rf.predict(X_test)
   acc_rf = accuracy_score(y_test, y_pred_rf)
   print(f"Random Forest Accuracy: {acc_rf:.2f}")

   # Classification Reports
   print("\nDecision Tree Report:")
   print(classification_report(y_test, y_pred_tree))

   print("\nRandom Forest Report:")
   print(classification_report(y_test, y_pred_rf))
   ```

   - **Explanation:**
     - Compare the performance of a single decision tree with that of a random forest ensemble.
     - Observing higher accuracy and better classification metrics in the random forest indicates reduced variance.

2. **Assessing Variance Reduction:**

   - **Concept:**

     - **Bagging (Bootstrap Aggregating):** Random forests train multiple trees on different bootstrap samples and aggregate their predictions.
     - Reduces variance by averaging out individual tree fluctuations.

   - **Implementation Consideration:**
     - Increasing the number of trees (`n_estimators`) typically improves performance up to a point.

   ```python:ml/random_forest_tuning.py
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report

   def tune_random_forest(X_train, y_train, X_test, y_test):
       n_estimators = [50, 100, 200]
       max_depths = [5, 10, None]

       for n in n_estimators:
           for depth in max_depths:
               rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
               rf.fit(X_train, y_train)
               y_pred = rf.predict(X_test)
               acc = accuracy_score(y_test, y_pred)
               print(f"n_estimators={n}, max_depth={depth} => Accuracy: {acc:.2f}")
               print(classification_report(y_test, y_pred))
   ```

   - **Explanation:**
     - Experiment with different numbers of trees and tree depths.
     - Observe how increasing trees generally stabilizes accuracy and reduces variance.

3. **Feature Importances:**

   ```python:ml/feature_importances.py
   import pandas as pd
   import matplotlib.pyplot as plt

   def plot_feature_importances(model, feature_names):
       importances = model.feature_importances_
       indices = np.argsort(importances)[::-1]

       plt.figure(figsize=(12, 6))
       plt.title("Feature Importances")
       plt.bar(range(len(importances)), importances[indices], align='center')
       plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
       plt.tight_layout()
       plt.show()
   ```

   - **Explanation:**
     - Visualize the importance of each feature in the random forest model.
     - Helps in identifying which features contribute most to predictions.

**Observations:**

- **Random Forest Advantages:**

  - Higher accuracy compared to a single decision tree.
  - Reduced overfitting due to ensemble averaging.
  - Robustness to noise and outliers.

- **Single Decision Tree Limitations:**
  - Higher variance, sensitive to training data.
  - Prone to overfitting if not properly regularized.

**Conclusion:**

- **Random Forests:** Powerful ensemble method that mitigates the high variance of individual decision trees, leading to more stable and accurate predictions.
- **Parameter Tuning:** Essential for optimizing performance, with `n_estimators` and `max_depth` being critical parameters.

---

### 3.3 Interpretability & Overfitting

**EXPLORATION**

**Challenge:** Interpretability & Overfitting

**Objective:** Assess whether decision trees are overfitting and understand feature importances within decision trees and random forests.

**Steps & Implementation:**

1. **Assessing Overfitting in Decision Trees:**

   ```python:ml/overfitting_assessment.py
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.metrics import classification_report

   def assess_overfitting(X_train, y_train, X_test, y_test, max_depth=None):
       tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
       tree.fit(X_train, y_train)
       y_train_pred = tree.predict(X_train)
       y_test_pred = tree.predict(X_test)

       print(f"Decision Tree with max_depth={max_depth}")
       print("Training Classification Report:")
       print(classification_report(y_train, y_train_pred))
       print("Test Classification Report:")
       print(classification_report(y_test, y_test_pred))
   ```

   - **Explanation:**
     - Compare training and test performance.
     - Significant disparity indicates overfitting.

2. **Checking Feature Importances:**

   ```python:ml/feature_importance_comparison.py
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.ensemble import RandomForestClassifier
   import pandas as pd
   import matplotlib.pyplot as plt

   def compare_feature_importances(X_train, y_train, feature_names):
       # Decision Tree
       tree = DecisionTreeClassifier(max_depth=5, random_state=42)
       tree.fit(X_train, y_train)
       importances_tree = tree.feature_importances_

       # Random Forest
       rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
       rf.fit(X_train, y_train)
       importances_rf = rf.feature_importances_

       # Plotting
       indices = np.argsort(importances_rf)[::-1]
       plt.figure(figsize=(12, 6))
       plt.title("Feature Importances: Decision Tree vs Random Forest")
       plt.bar(range(len(importances_rf)), importances_rf[indices], align='center', alpha=0.5, label='Random Forest')
       plt.bar(range(len(importances_tree)), importances_tree[indices], align='center', alpha=0.5, label='Decision Tree')
       plt.xticks(range(len(importances_rf)), [feature_names[i] for i in indices], rotation=90)
       plt.legend()
       plt.tight_layout()
       plt.show()
   ```

   - **Explanation:**
     - Compare feature importances between a single decision tree and a random forest.
     - Random forests often provide more reliable feature importance estimates.

3. **Interpreting Results:**
   - **Overfitting Indicators:**
     - High accuracy on the training set but low accuracy on the test set.
     - Example: Accuracy of 0.98 on training vs. 0.65 on testing.
   - **Feature Importances:**
     - Features with higher importance scores significantly influence model predictions.
     - Helps in feature selection and understanding model behavior.

**Conclusion:**

- **Overfitting in Decision Trees:**

  - Deep trees can memorize training data, leading to poor generalization.
  - Pruning techniques and setting `max_depth` help mitigate overfitting.

- **Feature Importance Insights:**
  - Decision trees and random forests provide valuable insights into which features drive predictions.
  - Consistent high importance across models increases confidence in their significance.

---

### Answers to Questions

1. **Why do decision trees overfit if they are grown too deep?**

   **Answer:**

   - **Complexity:** Deep decision trees create highly complex models that capture intricate patterns in the training data, including noise and outliers.
   - **Memorization:** Instead of learning generalizable rules, deep trees tend to memorize the training instances, reducing their ability to perform well on unseen data.
   - **Reduced Generalization:** Overfitting leads to high variance, where the model's predictions fluctuate significantly with different training data, undermining its reliability.

2. **What's the importance of random feature selection in a random forest?**

   **Answer:**

   - **Reducing Correlation:** By selecting a random subset of features for each split, random forests ensure that individual trees are less correlated with each other.
   - **Diversity:** Increased diversity among trees leads to better ensemble performance, as errors made by one tree can be corrected by others.
   - **Variance Reduction:** Random feature selection contributes to variance reduction without increasing bias, enhancing the overall model's robustness.

3. **How do ensemble methods (like random forests) address the bias-variance tradeoff?**

   **Answer:**

   - **Variance Reduction:** Ensemble methods like random forests aggregate multiple models (trees) to average out individual model variances, leading to more stable predictions.
   - **Maintaining Low Bias:** While reducing variance, ensembles maintain low bias by including diverse models that capture various data patterns.
   - **Balancing Tradeoff:** By combining multiple models, ensembles achieve a balance where the total error (sum of bias and variance) is minimized, improving generalization performance.

---

## 4. (Optional) Basic Statistical Concepts

### Goal

Review core statistics to handle typical ML or data-related interview questions.

### Quick Quiz

1. **p-Value**
   - Define it in simple terms.
   - Understand how it's used (and sometimes misused) in hypothesis testing.
2. **Type I & Type II Errors**
   - Relate them to false positives and false negatives.
   - Note which scenario is more critical in health-related or financial contexts.
3. **Confidence Intervals**
   - Describe how they're constructed.
   - Differentiate between confidence intervals and prediction intervals.

### Questions to Answer

- How does the concept of confidence intervals tie into model uncertainty?
- Why might you choose a 95% confidence interval vs. 99%?
- In an A/B testing scenario, how do Type I and Type II errors manifest?

---

### Answers to Questions

1. **How does the concept of confidence intervals tie into model uncertainty?**

   **Answer:**

   - **Confidence Intervals (CIs):** Provide a range within which the true parameter (e.g., mean, coefficient) is expected to lie with a certain level of confidence (e.g., 95%).
   - **Model Uncertainty:** CIs quantify the uncertainty around parameter estimates, reflecting the variability due to sampling.
   - **Interpretation:** Narrow CIs indicate higher precision in estimates, while wide CIs suggest greater uncertainty.
   - **Usage in Models:** In regression models, CIs for coefficients help assess the reliability of feature importance and the stability of the model.

2. **Why might you choose a 95% confidence interval vs. 99%?**

   **Answer:**

   - **95% Confidence Interval:**

     - **Balance:** Offers a good balance between confidence level and precision.
     - **Applicability:** Suitable for most practical applications where moderate confidence is adequate.
     - **Interpretation:** There is a 95% probability that the CI contains the true parameter.

   - **99% Confidence Interval:**

     - **Higher Confidence:** Provides greater assurance that the interval contains the true parameter.
     - **Trade-Off:** Results in a wider interval, reducing precision.
     - **Use Cases:** Critical applications where the cost of missing the true parameter is high (e.g., medical trials).

   - **Decision Factors:**
     - **Risk Tolerance:** Higher confidence levels are chosen when the consequences of errors are significant.
     - **Resource Constraints:** Lower confidence levels may be preferred when precision is more critical or resources are limited.

3. **In an A/B testing scenario, how do Type I and Type II errors manifest?**

   **Answer:**

   - **Type I Error (False Positive):**

     - **Manifestation:** Concluding that variant B performs better than variant A when, in reality, there is no difference.
     - **Implication:** Leads to potentially unnecessary changes based on incorrect assumptions.

   - **Type II Error (False Negative):**

     - **Manifestation:** Failing to detect a genuine improvement in variant B over variant A.
     - **Implication:** Misses out on beneficial optimizations that could enhance performance or user experience.

   - **Balancing Errors:**
     - **Significance of Context:** The criticality of Type I vs. Type II errors depends on the business context and potential impacts.
     - **Adjusting Significance Level:** Lowering the significance level (e.g., from 0.05 to 0.01) reduces Type I errors but may increase Type II errors.

---

## 5. Connecting It Back to E-Commerce (Optional Brainstorm)

You're preparing for a data science role in e-commerce, so it helps to tie these Day 2 concepts to business scenarios:

1. **Product Classification**
   - Imagine using logistic regression or a tree-based model to classify products into categories based on text descriptions.
   - What metrics would be key (F1-score if categories are imbalanced)?
2. **Predicting Customer Churn**
   - A regression or classification approach could answer retention questions.
   - Think about how you'd combine EDA and modeling to identify which features are most correlated with churn.
3. **Decision Tree for Promotions**
   - Who should receive a discount or coupon? A tree-based model could segment users by purchase frequency, cart size, etc.

---

## Day 2 Action Items Recap

By the end of Day 2, you should:

1. Have walked through several EDA steps on a small dataset, taking notes on key insights.
2. Implemented (or at least outlined) logistic regression and computed classical classification metrics—both by hand (conceptually) and by referencing library methods.
3. Explored tree-based models (decision trees, random forests), noting how they compare to simpler methods in terms of overfitting and interpretability.
4. Brushed up on foundational statistical concepts (p-value, errors, confidence intervals).

Keep all your notes organized. In the next steps (Day 3 and beyond), you'll build on these foundations for more advanced ML techniques—like deep learning and larger-scale data engineering tasks.

---

> **Next Steps (Preview for Day 3):**  
> You'll jump into neural networks, covering everything from forward/backprop to CNNs or RNNs/Transformers. Stay focused on the fundamentals you revisited here—understanding data, preprocessing, and metrics will remain crucial when you move into deep learning scenarios.

Good luck with Day 2—this will set you up nicely to discuss both theoretical and practical aspects of classical machine learning in your upcoming interviews!
