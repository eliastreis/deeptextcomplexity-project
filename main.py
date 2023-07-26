# general imports
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# machine learning libraries
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Concatenate
from tensorflow.keras import Model
from tensorflow.keras import backend as K
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
except:
    pass
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

# language processing/NLP libraries
import fasttext.util
import nltk
nltk.download('punkt')
import spacy
nlp = spacy.load('de_core_news_lg')
import textstat
textstat.set_lang("de")
from sentence_transformers import SentenceTransformer  # BERT
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
from textblob_de import TextBlobDE as TextBlob

fasttext.util.download_model('de', if_exists='ignore')
ft = fasttext.load_model('cc.de.300.bin')

# fasttext word vector size. Must be at most 300.
WORD_VECTOR_SIZE = 300


def sentence_extract_numerical_features(sentence):
    """
    This function takes a German sentence as input and computes a variety of numerical linguistic features based on the
    German sentence. All the computed linguistic features, metrics, statistics and ratios of the preceding are commented
    or described in the function. This function guarantees to always return a list of numerical values of the same size.
    :param sentence: German sentence
    :return: numpy array containing numerical linguistic features
    """
    features = []

    blob = TextBlob(sentence)
    sentence_length = len(blob.tokens)

    # sentence length
    features.append(sentence_length)
    features.append(len(blob))

    # (avg) number of syllables per sentence
    features.append(textstat.syllable_count(sentence))
    features.append(textstat.avg_syllables_per_word(sentence))

    # avg num of chars per word avg_character_per_word
    features.append(textstat.avg_character_per_word(sentence))

    # difficult word ratio
    features.append(textstat.difficult_words(sentence)/sentence_length)

    # https://github.com/shivam5992/textstat
    # automated readability index
    features.append(textstat.flesch_reading_ease(sentence))
    # SMOG Index
    features.append(textstat.smog_index(sentence))
    #Flesch-Kincaid Grade Level
    features.append(textstat.flesch_kincaid_grade(sentence))
    # Coleman-Liau Index
    features.append(textstat.coleman_liau_index(sentence))
    # Automated Readability Index
    features.append(textstat.automated_readability_index(sentence))
    # Dale-Chall Readability Score (not available in german)
    # features.append(textstat.dale_chall_readability_score(sentence))
    # Fog scale (not available in german)
    # features.append(textstat.gunning_fog(sentence))

    # Number of words with more then 3 syllables
    def more_then_3_syllables(sentence):
        count = 0
        for word in sentence.split(" "):
            syllable_count = textstat.syllable_count(word)
            if syllable_count >= 3:
                count += 1
        return count

    features.append(more_then_3_syllables(sentence))

    # Depth of semantic tree
    def walk_tree(node, depth = 0):
      if node.n_lefts + node.n_rights > 0: #number of left(right)ward immediate children of the word in the syntactic dependency parse
          return max(walk_tree(child, depth + 1) for child in node.children)
      else:
          return depth

    doc = nlp(sentence)
    features.append(np.max([walk_tree(sent.root) for sent in doc.sents]))

    # all tags; see https://github.com/markuskiller/textblob-de/blob/dev/textblob_de/ext/_pattern/text/search.py
    TAGS = ["CC", "CD", "CJ", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "JJ*",
            "LS", "MD", "NN", "NNS", "NNP", "NNPS", "NN*", "NO", "PDT", "PR",
            "PRP", "PRP$", "PR*", "PRP*", "PT", "RB", "RBR", "RBS", "RB*", "RP",
            "SYM", "TO", "UH", "VB", "VBZ", "VBP", "VBD", "VBN", "VBG", "VB*",
            "WDT", "WP*", "WRB", "X", ".", ",", ":", "(", ")"]

    different_word_tag_types_count = 0

    # get all tags in sentence
    tags_in_sentence = [e[1] for e in blob.pos_tags]
    for e in blob.tokens:
        if e in [".", ",", ":", "(", ")"]:
            tags_in_sentence.append(e)

    # number of occurrences of every tag and their ratio (in [0,1]) based on the length of the sentence.
    for tag in TAGS:
        appearances = tags_in_sentence.count(tag)
        features.append(appearances)
        features.append(appearances/sentence_length)
        if appearances >= 1:
            different_word_tag_types_count += 1

    # different word types count and ratio
    features.append(different_word_tag_types_count)
    features.append(different_word_tag_types_count/sentence_length)

    # the following computed values are used to compute different ratios below
    # based on lexical, syntactic and morphological features from https://www.aclweb.org/anthology/C12-1065.pdf
    N_unique_word_types = [token.lemma_ for token in nlp(sentence)]
    T_unique_words = {}
    N_unique_word_types_tmp = []
    T_unique_words_tmp = {}
    measure_of_textual_lexical_diversity = 0
    noun_adj_verb_adverbs = []
    noun_adj_verb_adverbs_dict = {}
    adj_adverbs = []
    adj_adverbs_dict = {}
    nouns = []
    nouns_dict = {}
    verbs = []
    verbs_dict = {}
    sein_cnt = 0
    haben_cnt = 0
    DIVIDE_BY_ZERO_ALTERNATIVE_VALUE = 0

    for w in N_unique_word_types:
        if w in T_unique_words:
            T_unique_words[w] += 1
        else:
            T_unique_words[w] = 1

        N_unique_word_types_tmp.append(w)
        if w in T_unique_words_tmp:
            T_unique_words_tmp[w] += 1
        else:
            T_unique_words_tmp[w] = 1
        if len(T_unique_words_tmp)/len(N_unique_word_types_tmp) <= 0.72:
            measure_of_textual_lexical_diversity += 1
            N_unique_word_types_tmp = []
            T_unique_words_tmp = {}

        b = TextBlob(w)
        if len(b.pos_tags) == 0:
            continue
        if b.pos_tags[0][1] in ["NN", "NNS", "NNP", "NNPS", "NN*", "NO"]:
            nouns.append(w)
            if w in nouns_dict:
                nouns_dict[w] += nouns_dict[w]
            else:
                nouns_dict[w] = 1
        if b.pos_tags[0][1] in ["VB", "VBZ", "VBP", "VBD", "VBN", "VBG", "VB*"]:
            verbs.append(w)
            if w in verbs_dict:
                verbs_dict[w] += verbs_dict[w]
            else:
                verbs_dict[w] = 1
        if b.pos_tags[0][1] in ["NN", "NNS", "NNP", "NNPS", "NN*", "NO", "JJ", "JJR", "JJS", "JJ*", "VB", "VBZ", "VBP", "VBD", "VBN", "VBG", "VB*", "RB", "RBR", "RBS", "RB*"]:
            noun_adj_verb_adverbs.append(w)
            if w in noun_adj_verb_adverbs_dict:
                noun_adj_verb_adverbs_dict[w] += noun_adj_verb_adverbs_dict[w]
            else:
                noun_adj_verb_adverbs_dict[w] = 1

        if b.pos_tags[0][1] in ["JJ", "JJR", "JJS", "JJ*", "RB", "RBR", "RBS", "RB*"]:
            adj_adverbs.append(w)
            if w in adj_adverbs_dict:
                adj_adverbs_dict[w] += adj_adverbs_dict[w]
            else:
                adj_adverbs_dict[w] = 1

        if w == "sein":
            sein_cnt += 1
        if w == "haben":
            haben_cnt += 1

    # T = unique word count (aka number of word types)
    T = len(T_unique_words)
    # N = total words count (total number word tokens in text)
    N = len(N_unique_word_types)
    # type-token ratio = (T / N)
    features.append(T/N)
    # Root Type-Token Ratio = (T / (N * N))
    features.append(T/(N*N))
    # Corrected Root Type-Token Ratio = (T / (2 * (N * N)))
    features.append(T/ (2 * (N*N)))
    # logarithmic type-token ratio = (log(T) / log(N))
    if N > 1: features.append(np.log2(T) / np.log2(N))
    else: features.append(DIVIDE_BY_ZERO_ALTERNATIVE_VALUE)
    # Uber Index = (Log^2 T /Log(N/T)).
    if T > 0 and N/T != 1: features.append((np.log2(T)*np.log2(T))/ np.log2(N/T))
    else: features.append(DIVIDE_BY_ZERO_ALTERNATIVE_VALUE)
    # Measure of Textual Lexical Diversity = compute type token ratio word for word, and if the type to token ratio goes below 0.72, then add 1 to the variable
    features.append(measure_of_textual_lexical_diversity)
    # lexical density = (nouns, adjectives, verbs, adverbs) count / N
    features.append(len(noun_adj_verb_adverbs)/N)
    # lexical word variation = unique (nouns, adjectives, verbs, adverbs) count / (nouns, adjectives, verbs, adverbs) count
    if len(noun_adj_verb_adverbs) > 0: features.append(len(noun_adj_verb_adverbs_dict)/len(noun_adj_verb_adverbs))
    else: features.append(DIVIDE_BY_ZERO_ALTERNATIVE_VALUE)
    # noun variation = unique nouns count / (nouns, adjectives, verbs, adverbs) count
    if len(noun_adj_verb_adverbs) > 0: features.append(len(nouns_dict)/len(noun_adj_verb_adverbs))
    else: features.append(DIVIDE_BY_ZERO_ALTERNATIVE_VALUE)
    # Adjective Variation, Adverb Variation Modifier Variation = unique (adj + adv) count / (nouns, adjectives, verbs, adverbs) count
    if len(noun_adj_verb_adverbs) > 0: features.append(len(adj_adverbs_dict)/len(noun_adj_verb_adverbs))
    else: features.append(DIVIDE_BY_ZERO_ALTERNATIVE_VALUE)
    # Verb Variation 1 = unique (verb) count / verbs count
    if len(verbs) > 0: features.append(len(verbs_dict)/len(verbs))
    else: features.append(DIVIDE_BY_ZERO_ALTERNATIVE_VALUE)
    # Verb Variation 2 = unique (verb) count / (nouns, adjectives, verbs, adverbs) count
    if len(noun_adj_verb_adverbs) > 0: features.append(len(verbs_dict)/len(noun_adj_verb_adverbs))
    else: features.append(DIVIDE_BY_ZERO_ALTERNATIVE_VALUE)
    # Squared Verb Variation 1 = (unique (verb) count)^2 / verb count
    if len(verbs) > 0: features.append((len(verbs_dict)*len(verbs_dict))/len(verbs))
    else: features.append(DIVIDE_BY_ZERO_ALTERNATIVE_VALUE)
    # Corrected Verb Variation 1 = unique (verb) count / sqrt(2 * verb count)
    if len(verbs) > 0: features.append(len(verbs_dict)/np.sqrt(2*len(verbs)))
    else: features.append(DIVIDE_BY_ZERO_ALTERNATIVE_VALUE)
    # sein to Verb Token Ratio
    if len(verbs) > 0: features.append(sein_cnt/len(verbs))
    else: features.append(DIVIDE_BY_ZERO_ALTERNATIVE_VALUE)
    # haben to Verb Token Ratio
    if len(verbs) > 0: features.append(haben_cnt/len(verbs))
    else: features.append(DIVIDE_BY_ZERO_ALTERNATIVE_VALUE)
    # Verb Token Ratio = verb count / N
    features.append(len(verbs)/N)
    # Noun Token Ratio = noun count / N
    features.append(len(nouns)/N)
    # Verb-Noun Token Ratio = (verb, noun) count / N
    features.append((len(verbs) + len(nouns))/N)

    # num of NPs per sentence
    features.append(len([e for e in doc.noun_chunks]))

    return np.array(features)


def sentence_to_word_embedding_representation(sentence):
    """
    This function takes a German sentence as input and returns a numpy array containing numerical word vectors that are
    generated using FastText. The length of each word vector is based on the variable WORD_VECTOR_SIZE.
    :param sentence: German sentence
    :return: numpy array containing numpy arrays of equal length
    """
    blob = TextBlob(sentence)
    words = blob.tokens
    return np.array([np.array(ft.get_word_vector(e)) for e in words])


def sentence2vec_representation(sentence):
    """
    This function takes a German sentence as input and returns a numerical sentence vector computed using the BERT
    language model
    :param sentence: German sentence
    :return: numpy array containing the numerical sentence vector of the sentence
    """
    return embedder.encode(sentence)

def pre_process_test_data(path):
    """
    This function takes a path (e.g. string to a path) as input, opens the excel file specified at that path and pre-
    processes the data in a format that can be used by the neural network model. The excel file must contain a "Sentence"
    and a "MOS" column. Otherwise the behavior of this function is undefined.
    :param path: path to a well formatted excel file (e.g. string to a path)
    :return: four values, the first being a numpy array where every entry contains the word vector representation of
    each sentence, the second being a numpy array where every entry contains the numerical features of each sentence,
    third a numpy array containing the BERT sentence vector of each vector, and fourth a numpy array containing the MOS
    score of every sentence.
    """
    data = pd.read_excel(path)

    # fastText numerical word vectors
    data["word_vector_sentences"] = data["Sentence"].apply(sentence_to_word_embedding_representation)
    # numerical features of every sentence
    data["numeric_features"] = data["Sentence"].apply(sentence_extract_numerical_features)
    # BERT numerical sentence vector
    data["sentence_vector_sentences"] = data["Sentence"].apply(sentence2vec_representation)

    data_x1 = np.array(data["word_vector_sentences"].tolist())
    data_x2 = np.array(data["numeric_features"].tolist())
    data_x3 = np.array(data["sentence_vector_sentences"].tolist())

    data_y = np.array(data["MOS"].tolist())

    return (data_x1, data_x2, data_x3, data_y)

def pre_process_training_data(path):
    """
    This function takes a path (e.g. string to a path) as input, opens the excel file specified at that path and pre-
    processes the data in a format that can be used by the neural network model. The excel file must contain a "Sentence"
    and a "MOS" column. Otherwise the behavior of this function is undefined. The data is then split into a training and
    a validation set in a way such that percentage wise both sets contain the about the same amount of sentences of
    every MOS score. The training set will contain 5/6th of the data, the validation set 1/6th.
    :param path: path to a well formatted excel file (e.g. string to a path)
    :return: two tuples with each tuple containing four values.
    The first entry of the tuple being a numpy array where every entry contains the word vector representation of
    each sentence, the second entry being a numpy array where every entry contains the numerical features of each
    sentence, the third entry being a numpy array containing the BERT sentence vector of each vector, and the fourth
    entry being a numpy array containing the MOS score of every sentence.
    """
    data = pd.read_excel(path)
    data.sort_values("MOS", inplace=True)

    # fastText numerical word vectors
    data["word_vector_sentences"] = data["Sentence"].apply(sentence_to_word_embedding_representation)
    # numerical features of every sentence
    data["numeric_features"] = data["Sentence"].apply(sentence_extract_numerical_features)
    # BERT numerical sentence vector
    data["sentence_vector_sentences"] = data["Sentence"].apply(sentence2vec_representation)

    # 2/3 training, 1/3 validation data sets
    training = data[data.index % 6 != 0]
    validation = data[data.index % 6 == 0]

    # shuffle based on seed
    training = shuffle(training, random_state=784128105)
    validation = shuffle(validation, random_state=2098634387)

    training_x1 = np.array(training["word_vector_sentences"].tolist())
    training_x2 = np.array(training["numeric_features"].tolist())
    training_x3 = np.array(training["sentence_vector_sentences"].tolist())
    training_y = np.array(training["MOS"].tolist())
    validation_x1 = np.array(validation["word_vector_sentences"].tolist())
    validation_x2 = np.array(validation["numeric_features"].tolist())
    validation_x3 = np.array(validation["sentence_vector_sentences"].tolist())
    validation_y = np.array(validation["MOS"].tolist())

    return (training_x1, training_x2, training_x3, training_y), (validation_x1, validation_x2, validation_x3, validation_y)



def root_mean_squared_error(y_true, y_pred):
    """
    This function computes the root mean squared error of two inputs
    :param y_true: numerical input
    :param y_pred: numerical input
    :return: root mean squared error of the two input values.
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def evaluate_sentences(evaluation_file, model_path):
    """
    This function takes a path to an CSV formatted file and the path to a model understood by this program and
    predicts the MOS value of every sentence in the CSV file using the model at the specified model_path.
    The CSV must contain a "sentence" and a "sent_id" column. The predicted MOS scores of every sentence as well as
    the sent_id of every sentence will be printed to stdout and will also be written as a CSV file to a file called
    "output.csv" in the current working directory.
    :param evaluation_file:
    :param model_path:
    :return:
    """
    with open('output.csv', 'w') as out_file:
        out_file.write("sent_id, mos\n")
        sentences = pd.read_csv(evaluation_file).rename(columns=lambda x: x.strip())

        model = keras.models.load_model(model_path, compile=False)
        for index, row in sentences.iterrows():

            sentence = row.sentence
            sent_id = row.sent_id

            # preprocess sentence
            # fastText numerical word vector
            x1 = sentence_to_word_embedding_representation(sentence)
            # numerical features of the sentence
            x2 = sentence_extract_numerical_features(sentence)
            # BERT numerical sentence vector
            x3 = sentence2vec_representation(sentence)

            # adjust shape of the input data
            x1 = x1.reshape((1, len(x1), WORD_VECTOR_SIZE))
            x2 = x2.reshape(1, len(x2))
            x3 = x3.reshape(1, len(x3))

            # predict mos value
            result = model.predict([x1, x2, x3], batch_size=1, steps=1)

            y = result[-1][-1]
            # y must be within [1,7]
            y = np.clip(y, 1.0, 7.0)

            # print to stdout and write to output.csv
            sent_id_without_n = sent_id.replace("\n","")
            print(sent_id_without_n + "," + str(y))
            out_file.write(sent_id_without_n + "," + str(y) + "\n")


def train_new_model(training_file, test_set_path, continue_training_model=False, model_path=None):
    # preprocess training, validation and test data sets
    (training_x1, training_x2, training_x3, training_y), (validation_x1, validation_x2, validation_x3, validation_y) = pre_process_training_data(training_file)
    (data_x1, data_x2, data_x3, data_y) = pre_process_test_data(test_set_path)
    TRAINING_SET_SIZE = training_y.size
    VALIDATION_SET_SIZE = validation_y.size

    if not continue_training_model:
        # train new model

        # FastText submodel
        model1 = keras.Sequential()
        # model.add(layers.Embedding(input_dim=WORD_VECTOR_SIZE, output_dim=1))
        model1.add(layers.LSTM(256, return_sequences=True, input_shape=(None, WORD_VECTOR_SIZE)))
        model1.add(layers.LSTM(128, return_sequences=True))
        model1.add(layers.LSTM(64, return_sequences=False))
        # model.add(layers.TimeDistributed(layers.Dense(32, activation='sigmoid')))

        # numerical features submodel
        model2 = keras.Sequential()
        model2.add(layers.Dense(256, input_shape=(training_x2.shape[1],)))
        # model2.add(layers.Dropout(0.2))
        # model2.add(layers.Dense(128))
        model2.add(layers.Dense(128))
        model2.add(layers.Dense(64))

        # BERT submodel
        model3 = keras.Sequential()
        model3.add(layers.Dense(256, input_shape=(training_x3.shape[1],)))
        # model3.add(layers.Dropout(0.2))
        model3.add(layers.Dense(128))
        model3.add(layers.Dense(64))

        # merged model using the preceding three models as input
        merged = Concatenate()([model1.output, model2.output, model3.output])
        z = layers.Dense(196)(merged)
        # z = layers.Dropout(0.2)(z)
        z = layers.Dense(64)(z)
        z = layers.Dense(32)(z)
        z = layers.Dense(1)(z)

        # final model
        model = Model(inputs=[model1.input, model2.input, model3.input], outputs=z)

        # exponential decreasing learning rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001,
            decay_steps=90,
            decay_rate=0.96,
            staircase=True
        )

        # compile the model
        model.compile(loss=root_mean_squared_error,
                      # optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
                      # optimizer=keras.optimizers.Adam(learning_rate=0.001))
                      optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
    else:
        # continue training preexisting model

        # exponential decreasing learning rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001,
            decay_steps=90,
            decay_rate=0.96,
            staircase=True
        )
        model = keras.models.load_model(model_path, compile=True, custom_objects={
            "lr_schedule": lr_schedule,
            "root_mean_squared_error": root_mean_squared_error
        })

    def train_generator(training_x1, training_x2, training_x3, training_y, repeat=True):
        """
        This generator is used to format the input data in a format that is usable by the model. If repeat is set to
        True, then the generator can be used to indefinitely return data to the model. Every time the whole data was
        returned once, the data is shuffled before returned again.
        :param training_x1: numpy array with numpy arrays of FastText word vectors of all sentences
        :param training_x2: numpy array with numpy arrays containing the numerical features of all sentences
        :param training_x3: numpy array with numpy arrays containing the BERT sentence vectors of all sentences
        :param training_y: numpy array containing the MOS score of all sentences
        :param repeat: True if this generator is used to indefinitely provide the model with data (e.g. during training),
        False if this generator is used for validation purposes.
        :return: input data in a format understandable by the model
        """
        while True:
            tmp = list(zip(training_x1, training_x2, training_x3, training_y))
            tmp = shuffle(tmp)
            training_x1, training_x2, training_x3, training_y = zip(*tmp)

            for x1, x2, x3, y in zip(training_x1, training_x2, training_x3, training_y):
                sequence_length = len(x1)

                x_train1 = np.array(x1).reshape((1, sequence_length, WORD_VECTOR_SIZE))
                x_train2 = np.array(x2).reshape(1, len(x2))
                x_train3 = np.array(x3).reshape(1, len(x3))
                yield [x_train1, x_train2, x_train3], np.array(y).reshape(1)

            if repeat is False:
                return


    callbacks = [
        # stop training once the whole training set was seen 6 times without improvements
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=TRAINING_SET_SIZE/16 * 6 + 2,
            verbose=2,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        ),
        # save every improved model
        tf.keras.callbacks.ModelCheckpoint(
            "models/weights.loss-{val_loss:.5f}--epoch-{epoch:04d}--wordvectorsize-"+str(WORD_VECTOR_SIZE)+".hdf5",
            monitor="val_loss",
            verbose=2,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            options=None,
        )

    ]

    # train the model
    seqModel = model.fit(
        train_generator(training_x1, training_x2, training_x3, training_y),
        validation_data=train_generator(validation_x1, validation_x2, validation_x3, validation_y),
        steps_per_epoch=16, validation_steps=VALIDATION_SET_SIZE, validation_freq=1, epochs=2000, verbose=2, callbacks=callbacks
    )

    # evaluate on the test set one final time
    test_loss = model.evaluate(train_generator(data_x1, data_x2, data_x3, data_y, False))

    # create plot of training process
    train_loss = seqModel.history['loss']
    val_loss = seqModel.history['val_loss']
    xc = range(len(train_loss))
    plt.figure()
    plt.plot(xc, train_loss, color="blue")
    plt.plot(xc, val_loss, color="orange")
    axes = plt.gca()
    axes.set_ylim([0.2, 0.9])
    axes.set_xlim([0, len(train_loss)])
    plt.hlines(y=min(val_loss), xmin=[0], xmax=[len(train_loss)], colors='orange', linestyles='dashed', lw=2, zorder=5)
    plt.hlines(y=[test_loss], xmin=[0], xmax=[len(train_loss)], colors='green', linestyles='dashed', lw=2, zorder=5)
    plt.legend(['training loss', 'validation loss', 'best validation loss', 'test loss'])
    plt.hlines(y=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], xmin=[0, 0, 0, 0, 0, 0, 0, 0], xmax=[len(train_loss)], colors='black', linestyles='dotted', lw=1)
    plt.savefig(f"models/training_plot_validation_loss-{min(val_loss)}_test_loss-{test_loss}.svg")
    plt.show()

    # print model summary
    model.summary()


if __name__ == '__main__':
    # parse input parameters
    parser = ArgumentParser()
    parser.add_argument("--mode", action="store", dest="mode",
                        help="\"train\" to train a new model, \"evaluate\" to predict the MOS score "
                             "of sentences specified in the CSV file at the path provided at --input,"
                             "or \"continue_training\" to continue training the model specified at --model.", default="evaluate")
    parser.add_argument("--model", action="store", dest="modelPath",
                        help="Path to the model used for evaluation if in mode \"evaluate\" or used to "
                             "continue training if in mode \"continue_training\".", default="model.hdf5")
    parser.add_argument("--input", action="store", dest="input",
                        help="Path to the file that contains the sentences that should be used for training or evaluation. "
                             "If in \"train\" or \"continue_training\" mode, then must be an excel file with columns"
                             "\"Sentence\" and \"MOS\" or must be a file in CSV format with columns \"sentence\" and \"sent_id\".",
                        required=True)
    parser.add_argument("--testSet", action="store", dest="testSetPath", default="",
                        help="Path to the file that contains the sentences that should be used for testing the model "
                             "after training finished. The file must be an excel file with columns \"Sentence\" and \""
                             "MOS\". Required if in mode \"train\" or \"continue_training\".",
                        required=False)
    parser.add_argument("--fasttext", action="store", dest="fasttextVectorSize", default=300, type=int,
                        help="Word vector size used for FastText. Defaults to 300. Must be at most 300. " 
                             "If below 300, then the model will be reduced to a lower dimension. This might require a lot of RAM.")

    args = parser.parse_args()
    # --mode continue_training --input training.xlsx --model good_good_cont_weights.loss-0.37711--epoch-0245--wordvectorsize-300.hdf5

    if (args.mode == "train" or args.mode == "continue_training") and args.testSetPath == "":
        print("Selected mode \""+args.mode + "\", but path to test set was not specified. Use --testSet TESTSETPATH"
                                             "to specify the path to your test set.")
        exit(-1)

    # if specified word vector size is below 300, then reduce the FastText model to specified dimension
    WORD_VECTOR_SIZE = args.fasttextVectorSize
    if WORD_VECTOR_SIZE < 300:
        fasttext.util.reduce_model(ft, WORD_VECTOR_SIZE)

    # train new model or evaluate sentences using a specified model
    if args.mode == "train":
        train_new_model(args.input, test_set_path=args.testSetPath)
    if args.mode == "continue_training":
        train_new_model(args.input, continue_training_model=True, model_path=args.modelPath, test_set_path=args.testSetPath)
    if args.mode == "evaluate":
        evaluate_sentences(args.input, args.modelPath)
