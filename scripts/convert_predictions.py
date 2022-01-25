import pandas


def convert_predictions(result_path: str, invert: bool = False, round_values: bool = False):
    predictions = pandas.read_csv('models/predictions.csv')
    predictions['Overall'] = predictions['Overall'].apply(lambda x: max(1, min(4, x)))
    if invert:
        predictions['Overall'] = predictions['Overall'].apply(lambda x: 5 - x)
    if round_values:
        predictions['Overall'] = predictions['Overall'].apply(round)
    predictions.to_csv(result_path, index=False)


convert_predictions('models/predictions_cleaned.csv')
convert_predictions('models/predictions_cleaned_inverted.csv', invert=True)
convert_predictions('models/predictions_cleaned_rounded.csv', round_values=True)
convert_predictions('models/predictions_cleaned_inverted_rounded.csv', invert=True, round_values=True)
