import pandas
import util
from calculate_article_similarity import OUTPUT_CSV_PATH
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.impute import KNNImputer, SimpleImputer
from scipy.stats import pearsonr
import joblib
import matplotlib.pyplot as plt


# if knn_imputer = True, compute missing keyword similarities from k nearest neighbors, else fill with zero
# if create_test_set = True, split training data and evaluate performance, else train on whole data
def load_training_data(training_data_path: str, knn_imputer: bool = False):
    training_data = pandas.read_csv(training_data_path, na_values=['NULL'])
    if knn_imputer:
        imputer = KNNImputer(n_neighbors=3)
    else:
        imputer = SimpleImputer(strategy='constant', fill_value=0.0)
    training_data = pandas.DataFrame(imputer.fit_transform(training_data), columns=training_data.columns)
    x = training_data[[util.DATA_BERT_SIM_21, util.DATA_BERT_SIM_12, util.DATA_USE_SIM_21, util.DATA_USE_SIM_12]]
    y = training_data[util.DATA_OVERALL_SCORE]
    return x, y


# load model from .joblib file
def load_model(model_path: str):
    return joblib.load(model_path)


def plot_model(y_labels, y_predictions, pdf_path: str):
    positions = list(set(y_labels))
    data = [[prediction for (label, prediction) in zip(y_labels, y_predictions) if label == pos] for pos in positions]
    plt.title('Model Performance (Mean = green, Median = red)')
    plt.xlabel('Labeled score')
    plt.ylabel('Predicted score')
    plt.plot([1, 4], [1, 4], color='black', label='Ideal behavior')
    violin_parts = plt.violinplot(data, positions, showmedians=True, showextrema=True, widths=0.1, showmeans=True)
    violin_parts['cmedians'].set_edgecolor('red')
    violin_parts['cmeans'].set_edgecolor('green')
    violin_parts['cbars'].set_linewidth(1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_path)


def train_random_forest(training_data_path: str, model_path: str, knn_imputer: bool = False, create_test_set: bool = False):
    x, y = load_training_data(training_data_path, knn_imputer)
    random_forest = RandomForestRegressor(n_estimators=100)

    if create_test_set:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # 80% training and 20% test
        random_forest.fit(x_train, y_train)
        # evaluation
        y_predictions = random_forest.predict(x_test)
        with open(f"{model_path}.txt", 'w') as file:
            file.write(f"Mean squared error: {metrics.mean_squared_error(y_test, y_predictions)}\n"
                       f"Mean absolute error: {metrics.mean_absolute_error(y_test, y_predictions)}\n"
                       f"Pearson correlation coefficient (r, p-value): {pearsonr(y_test, y_predictions)}")
        plot_model(y_test, y_predictions, f"{model_path}.pdf")
    else:
        random_forest.fit(x, y)

    # save model to file
    joblib.dump(random_forest, model_path)


if __name__ == "__main__":
    train_random_forest(OUTPUT_CSV_PATH, '../../models/random_forest_kw_zero.joblib', False, True)
    train_random_forest(OUTPUT_CSV_PATH, '../../models/random_forest_kw_knn.joblib', True, True)
    train_random_forest(OUTPUT_CSV_PATH, '../../models/random_forest_no_test_kw_zero.joblib', False, False)
    train_random_forest(OUTPUT_CSV_PATH, '../../models/random_forest_no_test_kw_knn.joblib', True, False)
