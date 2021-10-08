import random
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from operator import itemgetter
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import matplotlib.pyplot as plt
from general_helper_functions import *
from textfile_functions import *
from global_variables import *

'''
classifier_functions.py
This file contains all the classifier functions.
'''

warnings.filterwarnings("ignore")  # for ignore warnings

# Lists with keywords for every dialogue act
# These are used for the keyword matching (baseline 1)
word_list_ack = ["okay", "kay"]
word_list_affirm = ["yes"]
word_list_bye = ["goodbye", "bye"]
word_list_confirm = ["it"]
word_list_deny = ["wrong", "dont", "not", "change"]
word_list_hello = ["hello", "hi", "welcome"]
word_list_inform = ["restaurant", "food", "expensive", "moderately", "moderate", "cheap", "town"]
word_list_negate = ["no"]
word_list_null = ["noise", "sil", "unintelligible", "cough"]
word_list_repeat = ["repeat", "again"]
word_list_reqalts = ["about", "else"]
word_list_reqmore = ["more"]
word_list_request = ["adress", "phone", "number", "location", "post", "range", "price"]
word_list_restart = ["start"]
word_list_thankyou = ["thank"]


# __________________________________________PUBLIC FUNCTIONS__________________________________________ #

def classify_by_keyword_accuracy(file):
    """ Calculate the accuracy of baseline 1 """
    file_utterances, file_acts = split_into_utterances_and_acts(file)

    correct = 0
    for i in range(len(file_utterances)):
        actual = file_acts[i]
        classify = classify_by_keyword(file_utterances[i])
        if len(classify) == 0:
            classify.append(random.choice(TARGET_NAMES))
        for x in classify:
            if x == actual:
                correct += 1

    accuracy_baseline1 = (100 * correct) / len(file_utterances)
    return accuracy_baseline1


def randomly_classify_accuracy(file, distribution):
    """ Calculate the accuracy of baseline 2 """
    file_utterances, file_acts = split_into_utterances_and_acts(file)

    correct = 0
    for i in range(len(file_utterances)):
        randomPercentage = random.uniform(0, 1)
        actual = file_acts[i]

        for j in range(len(distribution)):
            if randomPercentage < distribution[j][1]:
                classify = distribution[j][0]
                if classify == actual:
                    correct += 1
                break

    accuracy_baseline2 = (100 * correct) / len(file_utterances)
    return accuracy_baseline2


def classify_by_keyword(utterance):
    """ Classify an utterance and return a list with all possible dialoge acts """
    dialog_act_list = []
    split_utterance = utterance.split(" ")
    for i in split_utterance:
        if i in word_list_ack:
            dialog_act_list.append("ack")
        elif i in word_list_affirm:
            dialog_act_list.append("affirm")
        elif i in word_list_bye:
            dialog_act_list.append("bye")
        elif i in word_list_confirm:
            dialog_act_list.append("confirm")
        elif i in word_list_deny:
            dialog_act_list.append("deny")
        elif i in word_list_hello:
            dialog_act_list.append("hello")
        elif i in word_list_inform:
            dialog_act_list.append("inform")
        elif i in word_list_negate:
            dialog_act_list.append("negate")
        elif i in word_list_null:
            dialog_act_list.append("null")
        elif i in word_list_repeat:
            dialog_act_list.append("repeat")
        elif i in word_list_reqalts:
            dialog_act_list.append("reqalts")
        elif i in word_list_reqmore:
            dialog_act_list.append("reqmore")
        elif i in word_list_request:
            dialog_act_list.append("request")
        elif i in word_list_restart:
            dialog_act_list.append("restart")
        elif i in word_list_thankyou:
            dialog_act_list.append("thankyou")

    return list(dict.fromkeys(dialog_act_list))


def randomly_classify_with_distribution_file(file_data):
    """ Open the given file and return a list of cumulative probabilities """
    file = open(file_data, "r")

    return get_cumulative_act_probability_list(file)


def randomly_classify_with_distribution(distribution):
    """ Return a random dialogue act from the list of distributions """
    input("Please enter an utterance to classify:  ")
    random_percentage = random.uniform(0, 1)
    for i in range(len(distribution)):
        if random_percentage < distribution[i][1]:
            return distribution[i][0]


def create_list_unique_words(train_file):
    """ Create a list of unique words, given a .txt file """
    with open(train_file) as infile:
        lst = []
        for line in infile:
            words = line.split()
            words.pop(0)
            for word in words:
                if word not in lst:
                    lst.append(word)  # append only this word to the list, not all words on this line
        lst.sort()
    return lst


def create_data_matrix(vocabulary, file):
    """ Convert the train data to a matrix. Row is a utterance vector, features/columns are the unique words. """
    data_matrix = []

    with open(file) as infile:
        for line in infile:
            data_matrix.append(" ".join(line.split()[1:]))

    return CountVectorizer(vocabulary=vocabulary, analyzer="word").transform(data_matrix).toarray()


def create_labels_matrix(file, lb_classes=None):
    """ Get labels list of utterances, then vectorize them to create a matrix. """
    labels = []

    with open(file) as infile:
        for i, line in enumerate(infile):
            labels.append(line.split(" ")[0])

    if lb_classes is not None:
        label_binarizer = LabelBinarizer()
        label_binarizer.classes_ = lb_classes
        labels_matrix = label_binarizer.transform(labels)
    else:
        label_binarizer = LabelBinarizer()
        labels_matrix = label_binarizer.fit_transform(labels)

    return labels_matrix, label_binarizer.classes_


def neural_network(data, labels, vocabulary, test_data, test_labels):
    model = Sequential([
        Dense(len(vocabulary), input_shape=(len(vocabulary),)),
        Activation('relu'),
        Dropout(0.88),
        Dense(15),
        Activation('softmax'),
    ])
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # history = model.fit(data, labels, validation_split=0.25, epochs=5)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    score = model.evaluate(test_data, test_labels)
    print(score)
    return model


# __________________________________________PRIVATE FUNCTIONS__________________________________________ #


def randomly_classify_with_distribution_train(train_data):
    """ Get the utterances content and the associated dialogue acts and make a txt file with all the utterances """
    train_file = open(train_data, "r")
    list_cumulative_act_probability = get_cumulative_act_probability_list(train_file)
    train_file.close()

    return list_cumulative_act_probability


def get_cumulative_act_probability_list(train_file):
    """ Return a list of tuples for the training set. The tuples are of type (act, cumulative_probability) """
    act_dictionary, number_of_acts = get_act_frequencies(train_file)
    act_probability_sorted_list = get_sorted_act_probabilities(act_dictionary, number_of_acts)

    act_probability_sorted_cumulative_list = []
    for index, act in enumerate(act_probability_sorted_list):
        if index == 0:
            act_probability_sorted_cumulative_list.append((act[0], act[1]))
        else:
            act_probability_sorted_cumulative_list.append(
                (act[0], (act[1] + act_probability_sorted_cumulative_list[index - 1][1])))
    return act_probability_sorted_cumulative_list


def get_sorted_act_probabilities(act_dictionary, number_of_acts):
    """ Return a sorted list with tuples of type (act, probability) """
    act_probability_list = []
    for act in act_dictionary:
        probability = act_dictionary[act] / number_of_acts
        act_probability_list.append((act, probability))

    act_probability_sorted_list = sorted(act_probability_list, key=itemgetter(1))
    return act_probability_sorted_list


def get_act_frequencies(train_file):
    """ Return all act frequencies (dict) and the total number of acts """
    act_dictionary = {}
    number_of_acts = 0

    for i, line in enumerate(train_file):
        act = line.split(" ")[0]
        if act in act_dictionary:
            act_dictionary[act] += 1
            number_of_acts += 1
        else:
            act_dictionary[act] = 1
            number_of_acts += 1
    return act_dictionary, number_of_acts


def split_into_utterances_and_acts(file):
    """ Split the file in two lists (1 with all the utterances and 1 with all the dialog_acts """
    utterances, acts = [], []

    used_file = open(file, "r")
    lines_file = used_file.readlines()

    for line in lines_file:
        content = line.split(" ")
        label = [i.replace("\n", "") for i in content[1:]]
        acts.append(content[0])
        utterances.append(" ".join(label))

    return utterances, acts


# --------------ML: Text classification
def preparing_data_set(train_file, list_of_keywords):
    labels, texts = [], []

    text_file = open(train_file, "r")
    lines = text_file.readlines()

    for line in lines:
        content = line.split(" ")
        label = [i.replace("\n", "") for i in content[1:]]
        labels.append(content[0])
        texts.append(" ".join(label))

    # data_preprocessing
    encoder = LabelEncoder()
    target = encoder.fit_transform(labels)

    count_vect = CountVectorizer(vocabulary=list_of_keywords, analyzer='word')
    X_train_counts = count_vect.fit_transform(texts).toarray()

    tfidf_transformer = TfidfTransformer()
    data = tfidf_transformer.fit_transform(X_train_counts).toarray()

    return data, target


def create_train_classifiers(X, y):
    # Classifier Initialization (Random Forest, Secision Tress, LR)
    RF = RandomForestClassifier(n_estimators=100, max_depth=100)
    DT = DecisionTreeClassifier(max_depth=100)
    LR = LogisticRegression()

    # Classifier training
    print("Training classifiers...")
    RF.fit(X, y)
    DT.fit(X, y)
    LR.fit(X, y)

    return RF, DT, LR


def classify_by_ml(method, data, label):
    # Predictions
    prediction = method.predict(data)

    # Accuracy evaluation
    print("------------// classifier: ", str(method))
    # print("Classification Report: \n", classification_report(label, prediction))
    scores = cross_val_score(method, data, label, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std()))


def classify_utterance_from_input(vocabulary, RF, DT, LR):
    reply = str(input("Do you want to classify text from input? (y/n): ")).lower().strip()
    if reply == 'y':
        X_input = input("Please enter the text:  ")
        classify_utterance(X_input, vocabulary, RF, DT, LR)
        return classify_utterance_from_input(vocabulary, RF, DT, LR)
    else:
        return 0


def classify_utterance(X_input, vocabulary, RF, DT, LR):
    content = X_input.split(" ")
    count_vect = CountVectorizer(vocabulary=vocabulary, analyzer='word')
    X_input = count_vect.fit_transform(content)

    pred_DT = DT.predict(X_input)
    pred_RF = RF.predict(X_input)
    pred_LR = LR.predict(X_input)

    print("Decision Tree prediction: ", TARGET_NAMES[pred_DT[0]])
    print("Random Forest prediction: ", TARGET_NAMES[pred_RF[0]])
    print("Logistic Regression prediction: ", TARGET_NAMES[pred_LR[0]])