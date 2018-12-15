import pandas as pd
import numpy as np
import tensorflow as tf
import re
import pickle
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import nltk

# nltk.download('all')
print('TensorFlow Version: {}'.format(tf.__version__))

# 현재 1.1에 맞게 바꾸어 놓았음. 1월 2일 4:33/

## Insepcting the Data

reviews = pd.read_csv("title_cont_sum_norm_using_cont_cut.csv", encoding="cp949")

reviews.head()

reviews.rename(columns={reviews.columns[2]: "text"}, inplace=True)
reviews.rename(columns={reviews.columns[1]: "summary"}, inplace=True)

reviews = reviews.dropna(axis=0)  # na가 들어있는 행을 다 뻄

reviews.isnull().sum()

del reviews['Unnamed: 0']

reviews.head()

# Inspecting some of the reviews
for i in range(5):
    print("Review #", i + 1)
    print(reviews.summary[i])
    print(reviews.text[i])
    print()

## Preparing the Data

clean_texts = reviews.text
clean_summaries = reviews.summary

# Find the number of times each word was used and the size of the vocabulary
f = open("word_counts.txt", "rb")
word_counts = pickle.load(f)
f.close()

print("Size of Vocabulary:", len(word_counts))

f = open('w2v_title_sum_part.txt', 'rb')
embeddings_index = pickle.load(f)

print('Word embeddings:', len(embeddings_index))

word_counts.items()

# Find the number of words that are missing from CN, and are used more than our threshold.
missing_words = 0
threshold = 2

for word, count in word_counts.items():  # word_count 내에 단어들 순서는 항상 같다.
    if count >= threshold:
        if word not in embeddings_index:
            missing_words += 1

missing_ratio = round(missing_words / len(word_counts), 4) * 100  # word_embedings에 없는 단어 비율

print("Number of words missing from CN:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

with open("vocab_to_int.txt", "rb") as f:
    vocab_to_int = pickle.load(f)

with open("int_to_vocab.txt", "rb") as f:
    int_to_vocab = pickle.load(f)

print("Total number of unique words:", len(word_counts))  # 쓰인 단어 중복제외하고 쓰인 갯수
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(round(len(vocab_to_int) / len(word_counts) * 100, 2)))

vocab_to_int['원']

int_to_vocab[1]

# Need to use 300 for embedding dimensions to match CN's vectors.
embedding_dim = 300
nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]  # [i] 하면 각 행을 의미함
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))


def convert_to_ints(text, word_count, unk_count, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])  # vocab_to_int 단어들의 숫자들을 갖고옴
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])  # vocab_to_int 에 없으면 unknown!
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)  # sentence_int를 int에 추가한다 !
    return ints, word_count, unk_count

# Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

unk_percent = round(unk_count / word_count, 4) * 100

print("Total number of words in headlines:", word_count)  # 여기서 이 결과는 각 단어들이 나온 모든 횟수를 다 더한것 !
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))
# int_summaries는 단어들을 idx로 변환한 애들임 !

def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])


lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

print("Summaries:")
print(lengths_summaries.describe())
print()
print("Texts:")
print(lengths_texts.describe())  # 문장마다 단어 몇 개가 쓰였는지 !!!

# Inspect the length of texts
print(np.percentile(lengths_texts.counts, 90))
print(np.percentile(lengths_texts.counts, 95))
print(np.percentile(lengths_texts.counts, 99))

# Inspect the length of summaries
print(np.percentile(lengths_summaries.counts, 90))
print(np.percentile(lengths_summaries.counts, 95))
print(np.percentile(lengths_summaries.counts, 99))  # 몇개의 단어가 쓰였는지를 의미 !!!


def unk_counter(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count


# Sort the summaries and texts by the length of the texts, shortest to longest
# Limit the length of summaries and texts based on the min and max ranges.
# Remove reviews that include too many UNKs

sorted_summaries = []
sorted_texts = []
max_text_length = 46
max_summary_length = 15
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0

for length in range(min(lengths_texts.counts), max_text_length):
    for count, words in enumerate(int_summaries):
        if (len(int_summaries[count]) >= min_length and
                len(int_summaries[count]) <= max_summary_length and
                len(int_texts[count]) >= min_length and
                unk_counter(int_summaries[count]) <= unk_summary_limit and
                unk_counter(int_texts[count]) <= unk_text_limit and
                length == len(int_texts[count])
        ):
            sorted_summaries.append(int_summaries[count])
            sorted_texts.append(int_texts[count])

# Compare lengths to ensure they match
print(len(sorted_summaries))
print(len(sorted_texts))


## Building the Model

def model_inputs():
    '''Create palceholders for inputs to the model'''

    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')

    return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length


def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''

    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])  # ???
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    '''Create the encoding layer'''

    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                    input_keep_prob=keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw,
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output, 2)

    return enc_output, enc_state


def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer,
                            vocab_size, max_summary_length):
    '''Create the training logits'''

    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=summary_length,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer)

    training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                           output_time_major=False,
                                                           impute_finished=True,
                                                           maximum_iterations=max_summary_length)
    return training_logits


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_summary_length, batch_size):
    '''Create the inference logits'''

    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)

    inference_logits, *_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                             output_time_major=False,
                                                             impute_finished=True,
                                                             maximum_iterations=max_summary_length)

    return inference_logits


def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length,
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                     input_keep_prob=keep_prob)

    output_layer = Dense(vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                     enc_output,
                                                     text_length,
                                                     normalize=False,
                                                     name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell,
                                                          attn_mech,
                                                          rnn_size)

    initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state[0],
                                                                    _zero_state_tensors(rnn_size,
                                                                                        batch_size,
                                                                                        tf.float32))

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,
                                                  summary_length,
                                                  dec_cell,
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size,
                                                  max_summary_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell,
                                                    initial_state,
                                                    output_layer,
                                                    max_summary_length,
                                                    batch_size)

    return training_logits, inference_logits


def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
    '''Use the previous functions to create the training and inference logits'''

    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix

    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
    # !
    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)  # !
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)  # !

    training_logits, inference_logits = decoding_layer(dec_embed_input,
                                                       embeddings,
                                                       enc_output,
                                                       enc_state,
                                                       vocab_size,
                                                       text_length,
                                                       summary_length,
                                                       max_summary_length,
                                                       rnn_size,
                                                       vocab_to_int,
                                                       keep_prob,
                                                       batch_size,
                                                       num_layers)

    return training_logits, inference_logits


def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts) // batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))

        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


# Set the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75

# Build the graph
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():
    # Load the model inputs
    input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                      targets,
                                                      keep_prob,
                                                      text_length,
                                                      summary_length,
                                                      max_summary_length,
                                                      len(vocab_to_int) + 1,
                                                      rnn_size,
                                                      num_layers,
                                                      vocab_to_int,
                                                      batch_size)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
print("Graph is built.")


## Making Our Own Summaries

def text_to_seq(text):
    '''Prepare the text for the model'''

    # text = clean_text(text)
    text = clean_texts[random]
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


randoms = np.arange(len(clean_texts))

len(clean_summaries)

checkpoint = "./best_model.ckpt"
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    extract_sum = []
    predict_abstract_sum = []
    real_abstract_sum = []

    for random in randoms:  # 새로 추가된 부분
        original_sum = clean_summaries[random]
        input_sentence = clean_texts[random]
        text = text_to_seq(clean_texts[random])

        # Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                          summary_length: [np.random.randint(5, 8)],
                                          text_length: [len(text)] * batch_size,
                                          keep_prob: 1.0})[0]

        # Remove the padding from the tweet
        pad = vocab_to_int["<PAD>"]

        input_words = " ".join([int_to_vocab[i] for i in text])
        response_words = " ".join([int_to_vocab[i] for i in answer_logits if i != pad])

        extract_sum.append(input_sentence)
        predict_abstract_sum.append(response_words)
        real_abstract_sum.append(original_sum)

        print('\nText')
        print(' Original Text:', input_sentence)
        print(' Input Words: {}'.format(input_words))

        print('\nSummary:')
        print(" Response Index :{}".format(answer_logits))
        print(' Response Words: {}'.format(response_words))
        print(' Original Summary:', original_sum)
        print('\n\n\n')

extract_sum = pd.DataFrame(extract_sum, columns=['extract_sum'])
predict_abstract_sum = pd.DataFrame(predict_abstract_sum, columns=['predict_abstract_sum'])
real_abstract_sum = pd.DataFrame(real_abstract_sum, columns=['real_abstract_sum'])
total = pd.concat([extract_sum, predict_abstract_sum, real_abstract_sum], axis=1)


total.to_csv("crap_test_set_all_result_with_train_data.csv", index=False)

### 테스트 데이터 고고

test = pd.read_csv("sns_sen.csv", encoding="cp949")
test = test.dropna()

test = pd.DataFrame(np.array(test), columns=['index', 'text'])

# 앞서 우리의 모델로는 잘나왔떤 결과의 인덱스들 ! (엑셀인뎃스니깐 고치자)
randoms = np.arange(len(test))

# 임의 데이터로 테스트
def text_to_seq(text):
    '''Prepare the text for the model'''
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


checkpoint = "./best_model.ckpt"
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    real_sentence = []
    predict_abstract_sum = []

    for random in randoms:  # 새로 추가된 부분
        input_sentence = test.text[random]
        text = text_to_seq(test.text[random])

        # Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                          summary_length: [np.random.randint(7, 11)],
                                          text_length: [len(text)] * batch_size,
                                          keep_prob: 1.0})[0]

        # Remove the padding from the tweet
        pad = vocab_to_int["<PAD>"]

        input_words = " ".join([int_to_vocab[i] for i in text])
        response_words = " ".join([int_to_vocab[i] for i in answer_logits if i != pad])

        predict_abstract_sum.append(response_words)
        real_sentence.append(input_sentence)

        print('\nText')
        print(' Original Text:', input_sentence)
        print(' Input Words: {}'.format(input_words))

        print('\nSummary:')
        print(" Response Index :{}".format(answer_logits))
        print(' Response Words: {}'.format(response_words))
        print('\n\n\n')

predict_abstract_sum = pd.DataFrame(predict_abstract_sum, columns=['predict_abstract_sum'])
real_sentence = pd.DataFrame(real_sentence, columns=['real_sentence'])

total = pd.concat([predict_abstract_sum, real_sentence], axis=1)
total.to_csv("sns_result2.csv", index=False)