from transformers import AutoModelForTokenClassification, AutoConfig
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str, dropout: float = 0.1):
    """
    Create a token classification model with optimized configuration.
    
    Args:
        model_name: HuggingFace model name
        dropout: Dropout probability for classifier head
    """
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(LABEL2ID)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    # Set dropout for better regularization
    if hasattr(config, 'classifier_dropout'):
        config.classifier_dropout = dropout
    elif hasattr(config, 'hidden_dropout_prob'):
        config.hidden_dropout_prob = dropout
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
    )
    return model
