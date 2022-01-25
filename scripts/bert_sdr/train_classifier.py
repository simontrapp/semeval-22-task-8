import pandas
from .util import load_data, write_metrics_to_file, DATA_PAIR_ID_1, DATA_PAIR_ID_2
# from calculate_article_similarity import OUTPUT_CSV_PATH
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
import joblib
import matplotlib.pyplot as plt


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


def train_random_forest(training_data_path: str, model_path: str, create_test_set: bool = False):
    x, y, _ = load_data(training_data_path)
    random_forest = RandomForestRegressor(n_estimators=100)

    if create_test_set:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # 80% training and 20% test
        random_forest.fit(x_train, y_train)
        # evaluation
        y_predictions = random_forest.predict(x_test)
        write_metrics_to_file(model_path, y_test, y_predictions)
        plot_model(y_test, y_predictions, f"{model_path}.pdf")
    else:
        random_forest.fit(x, y)

    # save model to file
    joblib.dump(random_forest, model_path)


def predict_scores(model_path: str, test_data_path: str, output_path: str):
    rf_model = load_model(model_path)
    x, y, pairs = load_data(test_data_path)
    predictions = rf_model.predict(x)
    out_data = pandas.DataFrame(
        pairs[DATA_PAIR_ID_1].combine(pairs[DATA_PAIR_ID_2], lambda p1, p2: f"{int(p1)}_{int(p2)}"))
    out_data['prediction'] = predictions
    # noinspection PyTypeChecker
    out_data.to_csv(output_path, header=['pair_id', 'Overall'], index=False)
    write_metrics_to_file(output_path, y, predictions)


#if __name__ == "__main__":
#    # train and evaluate on test set
#    train_random_forest(OUTPUT_CSV_PATH, '../../models/random_forest_kw_zero.joblib', True)
#    # use the whole data for training the random forest
#    train_random_forest(OUTPUT_CSV_PATH, '../../models/random_forest_no_test_kw_zero.joblib', False)
#    # EXAMPLE: predict with a pretrained random forest model on a preprocessed test set (change later accordingly!)
#    predict_scores('../../models/random_forest_no_test_kw_zero.joblib', OUTPUT_CSV_PATH, '../../models/predictions.csv')
