import pandas
from calculate_article_similarity import OUTPUT_CSV_PATH
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.impute import KNNImputer, SimpleImputer
import joblib


# if knn_imputer = True, compute missing keyword similarities from k nearest neighbors, else fill with zero
# if create_test_set = True, split training data and evaluate performance, else train on whole data
def train_random_forest(training_data_path: str, model_path: str, knn_imputer: bool = False, create_test_set: bool = False):
    training_data = pandas.read_csv(training_data_path, na_values=['NULL'])

    if knn_imputer:
        imputer = KNNImputer(n_neighbors=3)
    else:
        imputer = SimpleImputer(strategy='constant', fill_value=0.0)

    training_data = pandas.DataFrame(imputer.fit_transform(training_data), columns=training_data.columns)
    x = training_data[['sentence_similarity_2_to_1', 'sentence_similarity_1_to_2', 'keyword_similarity_2_to_1', 'keyword_similarity_1_to_2']]
    y = training_data['overall_score'].apply(round)     # round the floats to discrete int values

    random_forest = RandomForestClassifier(n_estimators=100)

    if create_test_set:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)    # 80% training and 20% test
        random_forest.fit(x_train, y_train)
        # evaluation
        y_predictions = random_forest.predict(x_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_predictions))
        print("Precision:", metrics.precision_score(y_test, y_predictions, average='micro'))
        print("Recall:", metrics.recall_score(y_test, y_predictions, average='micro'))
        print("F1:", metrics.f1_score(y_test, y_predictions, average='micro'))
    else:
        random_forest.fit(x, y)

    # save model to file
    joblib.dump(random_forest, model_path)


# load model from .joblib file
def load_model(model_path: str):
    return joblib.load(model_path)


if __name__ == "__main__":
    train_random_forest(OUTPUT_CSV_PATH, '../../models/random_forest_kw_zero.joblib', False, True)
    # train_random_forest(OUTPUT_CSV_PATH, '../../models/random_forest_kw_knn.joblib', True, True)
    train_random_forest(OUTPUT_CSV_PATH, '../../models/random_forest_no_test_kw_zero.joblib', False, False)
    # train_random_forest(OUTPUT_CSV_PATH, '../../models/random_forest_no_test_kw_knn.joblib', True, False)
