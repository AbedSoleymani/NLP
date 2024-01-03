# NLP

## Transformers vs. LSTM:
1. **Parallelization:** Due to the exsistence of attention mechanism, transformers process the entire sequence simultaneously, allowing for more efficient parallelization during both training and inference. However, LSTMs process sequences sequentially, which can limit parallelization and make training slower, especially for long sequences.
2. **Long-Range Dependencies:** Transformers can capture long-range dependencies more effectively through self-attention mechanisms. However, LSTMs struggle with capturing very long-range dependencies due to the vanishing gradient problem.
3. **Bidirectionality:** Transformers, by design, are deeply bidirectional. The self-attention mechanism allows each position in the sequence to attend to all positions, both to the left and right. This bidirectional attention enables Transformers to capture contextual information from both sides of a token simultaneously. On the other hand, LSTMs are inherently unidirectional in their standard form. While there are bidirectional variants (BiLSTM), they process the sequence in both forward and backward directions separately. The hidden states from both directions are usually concatenated but not interlinked during the sequential processing.
4. **Contextual Awareness:** Transformers naturally capture bidirectional context through self-attention. Each position has access to information from all positions, allowing for a richer understanding of context. This bidirectionality is crucial for tasks where understanding the relationship between words in both directions is essential. Hoever, in a unidirectional LSTM, the information from the future (right side) doesn't directly influence the processing of the past (left side) and vice versa. Bidirectional LSTMs partially address this by considering both past and future contexts separately, but they are far less effective than transformers.

## BERT (Bidirectional Encoder Representations from Transformers)
BERT is nothing more than a stack of transformer encoders!
The training process of BERT is done in two stages:
1. **Masked Language Model (MLM):** BERT training involves randomly masking some of the words in a sentence (more specifically, 15% of the words in each sequence are replaced with a [MASK] token.) and training the model to predict these masked words based on the surrounding context. This task encourages the model to capture bidirectional dependencies. In technical terms, the prediction of the output words requires:
- Adding a classification layer on top of the encoder output.
- Multiplying the output vectors by the embedding matrix, transforming them into the vocabulary dimension.
- Calculating the probability of each word in the vocabulary with softmax.
2. **Next Sentence Prediction (NSP):** In addition to the MLM task, BERT is also trained on a next sentence prediction task. Given a pair of sentences, the model learns to predict whether the second sentence follows the first in the original document.
