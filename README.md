# Task 2 – Netflix Content Type Prediction (Movie or TV Show)

Binary classification: predict whether a title is a **Movie** or a **TV Show**.

## Method
1. Feature selection (release year, date added, rating, country, director known, number of genres, genres)
2. Encoding: One-Hot (rating, country), standardization (numeric variables), multi-hot (genres), all inside a scikit-learn Pipeline
3. Training: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
4. Evaluation: accuracy, precision, recall, F1, AUC, 5-fold cross-validation, confusion matrix
5. Model comparison

## Key point: data leakage
The `duration` ("90 min" / "2 Seasons") and `listed_in` ("Dramas" / "TV Dramas") columns directly reveal the answer. They give ~100% accuracy without learning anything. The notebook demonstrates this, then keeps a feature set with no duration and only the genres shared by movies and TV shows.

| Feature set | Accuracy (CV) |
|---|---|
| A. Duration + raw genres | 1.000 |
| B. No duration, raw genres | 0.9997 |
| C. Normalized genres | 0.987 |
| **D. Shared genres (selected)** | **0.952** |
| Baseline (always "Movie") | 0.697 |

## Results (set D, 20% test split)
| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Logistic Regression | 0.948 | 0.913 | 0.973 |
| Decision Tree | 0.948 | 0.913 | 0.947 |
| Random Forest | 0.952 | 0.919 | 0.978 |
| Gradient Boosting | 0.955 | 0.925 | 0.980 |

## Run the project
```bash
pip install -r requirements.txt
jupyter notebook task2_content_type_prediction_EN.ipynb
```
