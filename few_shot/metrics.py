import torch


def categorical_accuracy(y, y_pred):
    """Calculates categorical accuracy.

    # Arguments:
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
        y: Ground truth categories. Must have shape [batch_size,]
    """

    print("CAT: {}".format(y_pred.argmax(dim=-1)))
    print("correct = {}".format(torch.eq(y_pred.argmax(dim=-1),
                                         y).sum().item()))
    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]


NAMED_METRICS = {
    'categorical_accuracy': categorical_accuracy
}
