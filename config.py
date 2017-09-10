class DefaultConfig:
    n_epochs = 5
    batch_size = 300
    validation_batch_size = 450
    max_timesteps = 10
    only_most_frequent = 1000
    num_topics = 50
#
    temperature = 2 # for text generation
    num_reviews_to_produce = 5
#
    decay_rate = .95
    learning_rate = 0.1
    input_dropout = 1.0
    hidden_dropout = 0.5
#
    cell_size_char = 75
    cell_size_word = 75
    num_layers_char = 1
    num_layers_word = 1
#
    char_embedding_length = 50
    word_embedding_length = 50