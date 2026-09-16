# Machine Learning Algorithm Implementations

A clean and minimal collection of fundamental Machine Learning algorithm implementations built with Python and Scikit-Learn. Designed for learning, experimentation, and reference, each notebook is standalone, fully reproducible, and ready to run in Google Colab without manual file uploads.

## Implemented Algorithms

1. **Linear Regression**: Single-feature regression with train-test split, MSE, R² score, and regression line visualization.
2. **Logistic Regression**: Binary classification on breast cancer dataset, sigmoid activation demonstration, probability prediction, accuracy, confusion matrix, and classification report.
3. **Support Vector Machine (SVM)**: Radial basis function (RBF) kernel classification with feature scaling, evaluation metrics, and confusion matrix display.
4. **Decision Tree Classifier**: Multi-class classification on the Iris dataset with tree structure visualization via `plot_tree`.
5. **Random Forest Classifier**: Ensemble classification on the Wine dataset with evaluation metrics and feature importance visualization.
6. **Naive Bayes Classifier**: Comprehensive demonstration of three variants — GaussianNB (continuous data), MultinomialNB (count data), and BernoulliNB (binary data) — with a comparative performance bar chart.
7. **K-Nearest Neighbors (KNN)**: Classification with feature scaling, hyperparameter tuning across K=1 to 20, optimal K selection, and confusion matrix display.
8. **K-Means Clustering**: Unsupervised clustering with feature scaling, elbow method curve (inertia/WCSS) for optimal K selection, cluster assignment, and centroid visualization.

## Folder Structure

```text
Implementations/
├── Linear_Regression/
│   └── linear_regression.ipynb
├── Logistic_Regression/
│   └── logistic_regression.ipynb
├── SVM/
│   └── svm.ipynb
├── Decision_Tree/
│   └── decision_tree.ipynb
├── Random_Forest/
│   └── random_forest.ipynb
├── Naive_Bayes/
│   └── naive_bayes.ipynb
├── KNN/
│   └── knn.ipynb
├── K_Means/
│   └── k_means.ipynb
└── README.md
```

## Technologies Used

- **Python 3**
- **NumPy**
- **Matplotlib**
- **Scikit-Learn**

## How to Open in Google Colab

1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select the **GitHub** tab in the file open modal.
3. Enter your repository URL (once pushed) or upload any `.ipynb` file from this repository via the **Upload** tab.
4. Run all cells by clicking **Runtime** > **Run all** (`Ctrl + F9` / `Cmd + F9`).
