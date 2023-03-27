from common import *

def F1Score(
    precision,
    recall,
):
    return ( 2 * precision * recall ) / ( precision + recall )

def score(
    labels,
    predictions,
    from_logits,
):

    metrics = {
        'AUC': tf.keras.metrics.AUC( from_logits = from_logits ),
        'categorical_crossentropy': tf.keras.metrics.CategoricalCrossentropy(
            from_logits = from_logits,
        ),
        'precision': tf.keras.metrics.Precision(),
        'recall': tf.keras.metrics.Recall(),
        'accuracy': tf.keras.metrics.TopKCategoricalAccuracy( k = 1 ),
        'accuracy_top3': tf.keras.metrics.TopKCategoricalAccuracy( k = 3 ),
        'accuracy_top10': tf.keras.metrics.TopKCategoricalAccuracy( k = 10 ),
    }


    scores = {}

    for key, score in metrics.items():
        scores[ key ] = score(
            labels,
            predictions,
        ).numpy()

    scores[ 'F1' ] = F1Score( scores['precision'], scores['recall'] )

    return scores



