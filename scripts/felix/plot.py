import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import pandas
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from util import COMPARISON_RESULT_CSV_PATH, COMPARISON_RESULT_PATH, PREPROCESSING_MODEL, COMPARISON_MODEL

matplotlib.use('Agg')  # Needed to prevent memory leak
plt.style.use('default')
sns.set_style("whitegrid")

PLOT_DIST = True


def plot_line(score_data, color, label, plot_dist=False):
    if plot_dist:
        sns.distplot(score_data, color=color, label=label)
    else:
        plt.plot([i for i in range(len(score_data))], score_data, color=color, label=label)


data = pandas.read_csv(COMPARISON_RESULT_CSV_PATH)

plot_line(data['score_overall'], 'red', 'Overall Score', PLOT_DIST)
plot_line(data['score_model'], 'blue', 'Model Score', PLOT_DIST)
plot_line(abs(data['score_overall'] - data['score_model']), 'green', 'Difference', PLOT_DIST)

if PLOT_DIST:
    plt.xlabel("Score/Value")
    plt.ylabel("Frequency")
else:
    plt.xlabel("Pair Index")
    plt.ylabel("Score/Value")

# print the scoring results
plt.title(f"Score Distributions | Preprocessing: {PREPROCESSING_MODEL.value} | Scoring: {COMPARISON_MODEL.value}")
plt.legend()
plt.savefig(f"{COMPARISON_RESULT_PATH}/score_plot.pdf")
plt.savefig(f"{COMPARISON_RESULT_PATH}/score_plot.png")
plt.close()

# print the score relations
plt.title(f"Score Relations | Preprocessing: {PREPROCESSING_MODEL.value} | Scoring: {COMPARISON_MODEL.value}")
plt.scatter(data['score_overall'], data['score_model'], label='Pair score', color=(1, 0, 0, 0.6))
plt.xlabel('Overall score (gold)')
plt.ylabel('Model score (linear normalization)')
plt.legend()
plt.savefig(f"{COMPARISON_RESULT_PATH}/score_relations.pdf")
plt.savefig(f"{COMPARISON_RESULT_PATH}/score_relations.png")
plt.close()
