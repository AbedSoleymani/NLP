# NLP

## Transformers vs. LSTM:
1. **Parallelization:** Due to the exsistence of attention mechanism, transformers process the entire sequence simultaneously, allowing for more efficient parallelization during both training and inference. However, LSTMs process sequences sequentially, which can limit parallelization and make training slower, especially for long sequences.
2. **Long-Range Dependencies:** Transformers can capture long-range dependencies more effectively through self-attention mechanisms. However, LSTMs struggle with capturing very long-range dependencies due to the vanishing gradient problem.
3. **Bidirectionality:** Transformers, by design, are deeply bidirectional. The self-attention mechanism allows each position in the sequence to attend to all positions, both to the left and right. This bidirectional attention enables Transformers to capture contextual information from both sides of a token simultaneously. On the other hand, LSTMs are inherently unidirectional in their standard form. While there are bidirectional variants (BiLSTM), they process the sequence in both forward and backward directions separately. The hidden states from both directions are usually concatenated but not interlinked during the sequential processing.
4. **Contextual Awareness:** Transformers naturally capture bidirectional context through self-attention. Each position has access to information from all positions, allowing for a richer understanding of context. This bidirectionality is crucial for tasks where understanding the relationship between words in both directions is essential. Hoever, in a unidirectional LSTM, the information from the future (right side) doesn't directly influence the processing of the past (left side) and vice versa. Bidirectional LSTMs partially address this by considering both past and future contexts separately, but they are far less effective than transformers.

![image](https://github.com/AbedSoleymani/NLP/assets/72225265/05743ac3-189b-4ef1-bd90-af76e0f4f6ce)


## BERT (Bidirectional Encoder Representations from Transformers)
BERT is nothing more than a stack of transformer encoders!
<img width="1332" alt="Screenshot 2024-01-02 at 7 54 22 PM" src="https://github.com/AbedSoleymani/NLP/assets/72225265/cfb149db-52f0-4844-9224-9329644dc104">

### Training
The training process of BERT is done in two stages:
1. **Masked Language Model (MLM):** BERT training involves randomly masking some of the words in a sentence (more specifically, $15$% of the words in each sequence are replaced with a [MASK] token.) and training the model to predict these masked words based on the surrounding context. This task encourages the model to capture bidirectional dependencies. In technical terms, the prediction of the output words requires:
   1. Adding a classification layer on top of the encoder output.
   2. Multiplying the output vectors by the embedding matrix, transforming them into the vocabulary dimension.
   3. Calculating the probability of each word in the vocabulary with softmax.
   <p align="center">
   <img width="500" alt="image" src="https://github.com/AbedSoleymani/NLP/assets/72225265/f09c8a15-f20a-4c79-a64c-e937bd7a2860">
   </p>
   The MLM loss function takes into consideration only the prediction of the masked values and ignores the prediction of the non-masked words. As a consequence, the model converges slower than directional models.
2. **Next Sentence Prediction (NSP):** In addition to the MLM task, BERT is also trained on a next sentence prediction task. Given a pair of sentences, the model learns to predict whether the second sentence follows the first in the original document. More specifically, $50$% of the inputs are a pair in which the second sentence is the subsequent sentence in the original document, while in the other $50$% a random sentence from the corpus is chosen as the second sentence. The assumption is that the random sentence will be disconnected from the first sentence.
   To help the model distinguish between the two sentences in training, the input is processed in the following way before entering the model:
   1. A [CLS] token is inserted at the beginning of the first sentence and a [SEP] token is inserted at the end of each sentence.
   2. A sentence embedding indicating Sentence A or Sentence B is added to each token. Sentence embeddings are similar in concept to token embeddings with a vocabulary of 2.
   3. A positional embedding is added to each token to indicate its position in the sequence. The concept and implementation of positional embedding are presented in the Transformer paper.
      
   To predict if the second sentence is indeed connected to the first, the following steps are performed:
   1. The entire input sequence goes through the Transformer model.
   2. The output of the [CLS] token is transformed into a $2\times1$ shaped vector, using a simple classification layer (learned matrices of weights and biases).
   3. Calculating the probability of IsNextSequence with softmax.
   <p align="center">
   <img width="800" alt="image" src="https://github.com/AbedSoleymani/NLP/assets/72225265/a703db50-51d3-4cc7-816f-295336d90caa">
   </p>
When training the BERT model, Masked LM and Next Sentence Prediction are trained together, with the goal of minimizing the combined loss function of the two strategies.

### Transfer Learning and Fine-Tuning
Using BERT for a specific task is relatively straightforward:
BERT can be used for a wide variety of language tasks, while only adding a small layer to the core model:
1. Classification tasks such as sentiment analysis are done similarly to Next Sentence classification, by adding a classification layer on top of the Transformer output for the [CLS] token.
2. In Question Answering tasks (e.g. SQuAD which stands for Stanford Question Answering Dataset), the software receives a question regarding a text sequence and is required to mark the answer in the sequence. Using BERT, a Q&A model can be trained by learning two extra vectors that mark the beginning and the end of the answer.
3. In Named Entity Recognition (NER), the software receives a text sequence and is required to mark the various types of entities (Person, Organization, Date, etc) that appear in the text. Using BERT, a NER model can be trained by feeding the output vector of each token into a classification layer that predicts the NER label.

In fine-tuning stage, as stated in the original paper, most model hyperparameters are the same as in pre-training, with the exception of the batch size, learning rate, and number of train- ing epochs. The dropout probability was always kept at `0.1`. The optimal hyperparameter values are task-specific, but we found the following range of possible values to work well across all tasks:
- Batch size: `16`, `32`
- Learning rate (Adam): `5e-5`, `3e-5`, `2e-5`
- Number of epochs: `2`, `3`, `4`

It also observed that large data sets (e.g., $100k+$ labeled training examples) were far less sensitive to hyperparameter choice than small data sets. Fine-tuning is typically very fast, so it is reasonable to simply run an exhaustive search over the above parameters and choose the model that performs best on the development set.

## GPT (Generative Pre-trained Transformer)
GPT is nothing more than a stack of transformer decoders!
![image](https://github.com/AbedSoleymani/NLP/assets/72225265/bf481961-c894-4192-8e51-d28ce62d7abc)

### Training
The goal is to train the GPT Architecture to understand what language is. This stage is more straight-forward than BERT. GPT is pre-trained using an unsupervised learning task on a massive corpus of text data.
The model just has to learn to predict the next word in a sequence given the context of previous words. This task is known as causal language modeling.
The image below demonstrates the difference between BERT and GPT in treating sentences during the training phase. BERT uses a bidirectional Transformer. OpenAI GPT uses a left-to-right Transformer.
<p align="center">
   <img width="500" alt="image" src="https://github.com/AbedSoleymani/NLP/assets/72225265/a7748504-3b19-4e8d-865e-6df61e6bc74d">
   </p>

Prior to GPT, most state-of-the-art NLP models were trained specifically on a particular task like sentiment classification, textual entailment etc. using supervised learning. However, supervised models have two major limitations:
1. They need large amount of annotated data for learning a particular task which is often not easily available.
2. They fail to generalize for tasks other than what they have been trained for.

GPT proposed learning a generative language model using unlabeled data and then fine-tuning the model by providing examples of specific downstream tasks like classification, sentiment analysis, textual entailment/summarization, text translation, and question answering.

Following are the implementation details of the **GPT-1**:
- Model used 768-dimensional state for encoding tokens into word embeddings. Position embeddings were also learnt during training.
- `12` layered model was used with `12` attention heads in each self-attention layer.
- For position wise feed forward layer `3072`-dimensional state was used.
- Adam optimiser was used with learning rate of `2.5e-4`.
- Attention, residual and embedding dropouts were used for regularisation, with dropout rate of `0.1`. Modified version of $L2$ regularisation was also used for non-bias weights.
- $GELU$ was used as activation function.
- The model was trained for `100` epochs on mini-batches of size `64` and sequence length of `512`. The model had 117M parameters in total.

**GPT-2** had 1.5 billion parameters. which was 10 times more than GPT-1 (117M parameters). Major differences from GPT-1 are:
- GPT-2 had `48` layers and used `1600` dimensional vectors for word embedding.
- Larger vocabulary of `50,257` tokens was used.
- Larger batch size of `512` and larger context window of `1024` tokens were used.
- Layer normalisation was moved to input of each sub-block and an additional layer normalisation was added after final self-attention block.
- At initialisation, the weight of residual layers was scaled by $\frac{1}{\sqrt{N}}$, where N was the number of residual layers.

The architecture of **GPT-3** is same as GPT-2. Few major differences from GPT-2 are:
- GPT-3 has `96` layers with each layer having `96` attention heads.
- Size of word embeddings was increased to `12888` for GPT-3 from `1600` for GPT-2.
- Context window size was increased from `1024 `for GPT-2 to `2048` tokens for GPT-3.
- Adam optimiser was used with `beta_1=0.9`, `beta_2=0.95`, and `epsilon= 10e-8`.
- Alternating dense and locally banded sparse attention patterns were used.

### Transfer Learning and Fine-Tuning
Supervised fine-tuning took as few as 3 epochs for most of the downstream tasks. This showed that the model had already learnt a lot about the language during pre-training. Thus, minimal fine-tuning was enough. Most of the hyper parameters from unsupervised pre-training were used for fine-tuning.

A significant achievement by GPT model was its decent zero-shot performance on various tasks.
Zero shot learning or behaviour refers to the ability of a model to perform a task without having seen any example of that kind in past. No gradients update happen during zero shot learning and the model is supposed to understand the task without looking at any examples.
The paper demonstrated that model had evolved in zero shot performance on different NLP tasks like question-answering, schema resolution, sentiment analysis etc. due to pre-training.

The most impressive feature of GPT-3 is that it’s a meta-learner; it has learned to learn. You can ask it in natural language to perform a new task and it “understands” what it has to do, which is less or more similar to how a human would!




