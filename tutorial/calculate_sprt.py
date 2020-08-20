import sys

sys.path.append("../")

# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader

import sys

sys.path.append("../")
from sklearn.cluster import KMeans

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import copy

from adf_data.census import census_data
from adf_data.credit import credit_data
from adf_data.bank import bank_data
from adf_model.tutorial_models import dnn
from adf_utils.utils_tf import model_prediction, model_argmax
from adf_utils.config import census, credit, bank
from adf_tutorial.utils import cluster, gradient_graph
import itertools
from sklearn.datasets import load_iris
import random
# random.seed(1)
import time
# from SPRT import sprt_calculate
import matplotlib as mpl
import matplotlib.pyplot as plt


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def select_feature(dataset, feature_set, n):
    data = dataset[:]
    dataset = []
    all_feature = list(range(1, n + 1))
    for f in feature_set:
        all_feature.remove(f)
    for i in range(0, len(all_feature)):
        all_feature[i] -= i
    # print("delete feature index:", all_feature)
    for row in data:
        row = list(row)
        # dataset.append(row)
        for index in all_feature:
            del row[index - 1]
        dataset.append(row)
    return dataset


def get_DT_cluster(dataset, cluster_num, feature_set, n):
    # data = {"census": census_data, "credit": credit_data, "bank": bank_data}
    # data_config = {"census": census, "credit": credit, "bank": bank}

    def seed_test_input(clusters, limit, xx):
        i = 0
        rows = []
        max_size = max([len(c[0]) for c in clusters])  # num of params?
        # print("-->max_size:", max_size)
        while i < max_size:
            # if len(rows) >= limit:
            #     break
            for c in clusters:
                if i >= len(c[0]):
                    continue
                row = c[0][i]
                rows.append(row)
                # if len(rows) >= limit:
                #     break
            i += 1
        output_dataset = []
        final_rows = random.sample(rows, limit)
        for index in final_rows:
            data = list(xx[index])
            output_dataset.append(data)
        return output_dataset

    # X, Y, input_shape, nb_classes = data[dataset]()
    # print("-->X:", X)
    xx = select_feature(dataset, feature_set, n)
    xx = np.array(xx)
    # print("-->xx:", xx, type(xx), xx.shape)
    clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(xx)
    clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]

    # select the seed input for testing
    inputs = seed_test_input(clusters, min(5000, len(xx)), xx)  # 1000
    # print("-->inputs:", inputs)
    return inputs


# Load a CSV file
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def accuracy_dif_label(actual, predicted):
    correct = 0
    count_0 = 0
    count_1 = 0
    correct_0 = 0
    correct_1 = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
            if actual[i] == 0:
                correct_0 += 1
                count_0 += 1
            else:
                correct_1 += 1
                count_1 += 1
        else:
            if actual[i] == 0:
                count_0 += 1
            else:
                count_1 += 1
    accuracy_0 = correct_0 / float(count_0) * 100.0
    accuracy_1 = correct_1 / float(count_1) * 100.0
    accuracy = correct / float(len(actual)) * 100.0
    all_accuracy = [accuracy, accuracy_0, accuracy_1]
    return all_accuracy


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    dif_scores = list()
    decision_trees = []
    for fold in folds:
        # print("-->fold", fold)
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        print("-->build tree")
        predicted, tree = algorithm(train_set, test_set, *args)
        decision_trees.append(tree)
        actual = [row[-1] for row in fold]
        # print("-->actual:", actual)
        # print("-->predicted:", predicted)
        accuracy = accuracy_metric(actual, predicted)
        dif_accuracy = accuracy_dif_label(actual, predicted)
        dif_scores.append(dif_accuracy)
        scores.append(accuracy)
    return scores, dif_scores, decision_trees


def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    # print("n_instancesL:", n_instances)
    gini = 0.0
    for group in groups:
        # print("group:", group)
        size = float(len(group))
        # print("size:", size)
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            # print("class:", class_val)
            p = [row[-1] for row in group].count(class_val) / size
            # print([row[-1] for row in group].count(class_val))
            # print("p:", p)
            score += p * p
            # print("score:", score)
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
        # print("gini:", gini)
    return gini


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    # print("-->class_values:", class_values)
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        # print("-->index:", index)
        for row in dataset:
            # print("index:", index, row[index])
            groups = test_split(index, row[index], dataset)
            # print("groups:", groups)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                # print("-->gini", gini)
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    global total_gini
    total_gini += b_score
    global count_gini
    count_gini += 1
    # print("-->get_split:", b_index, b_value, b_score)
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# split = get_split(dataset)
# print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    # return most frequency number
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    # print("left:", left)
    # print("right:", right)
    del (node['groups'])
    # check for a no split
    if not left or not right:
        # print("end")
        # left or right is empty
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        # print("depth end")
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        # print(node['left'], node['right'])
        return
    # process left child
    if len(left) <= min_size:
        # print("end")
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        # print("end")
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    print("-->root:", root)
    split(root, max_depth, min_size, 1)
    return root


# Print a decision tree
def print_tree(node, depth=0):
    # print("node:", node)
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * ' ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


'''prediction'''


# Make a prediction with a decision tree
def predict(node, row, feature_set):
    if row[feature_set[node['index']] - 1] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row, feature_set)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row, feature_set)
        else:
            return node['right']


# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    # print("-->tree")
    # print_tree(tree)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions, tree


def con_fromtree(tree):
    conditions = []

    def predict(node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right']


def get_conditions(tree, result, dir, tmp=list()):
    if tree is None:
        return
    tmp.append([tree['index'], tree['value'], tree['left'], tree['right'], dir])
    tmp1 = copy.deepcopy(tmp)

    if isinstance(tree['left'], dict) == False or isinstance(tree['right'], dict) == False:
        if isinstance(tree['left'], dict) == False and isinstance(tree['right'], dict) == False:
            if tree['left'] != tree['right']:
                # print("-->tmp:", tmp)
                l = [[i[0], i[1], i[-1]] for i in tmp]
                l.append([tree['left'], 0])
                result.append(l)
                l = [[i[0], i[1], i[-1]] for i in tmp]
                l.append([tree['right'], 1])
                result.append(l)
            else:
                # print("-->tmp:", tmp)
                l = [[i[0], i[1], i[-1]] for i in tmp]
                l.append([tree['left']])
                result.append(l)
            return
        elif isinstance(tree['left'], dict) == False:
            l = [[i[0], i[1], i[-1]] for i in tmp]
            l.append([tree['left'], 0])
            result.append(l)
        elif isinstance(tree['right'], dict) == False:
            l = [[i[0], i[1], i[-1]] for i in tmp]
            l.append([tree['right'], 1])
            result.append(l)

    if isinstance(tree['left'], dict):
        get_conditions(tree['left'], result, 0, tmp)
    if isinstance(tree['right'], dict):
        get_conditions(tree['right'], result, 1, tmp1)


# step size of perturbation
perturbation_size = 1


def clip(input, conf):
    """
    Clip the generating instance with each feature to make sure it is valid
    :param input: generating instance
    :param conf: the configuration of dataset
    :return: a valid generating instance
    """
    for i in range(len(input)):
        input[i] = max(input[i], conf.input_bounds[i][0])
        input[i] = min(input[i], conf.input_bounds[i][1])
    return input


def clip_range(input, conf, ranging):
    """
    Clip the generating instance with each feature to make sure it is valid
    :param input: generating instance
    :param conf: the configuration of dataset
    :return: a valid generating instance
    """
    for i in range(len(input)):
        input[i] = max(input[i], conf.input_bounds[i][0])
        input[i] = min(input[i], conf.input_bounds[i][1])
    return input


# def is_DT_condition(data, feature_set, condition):
#     for i in range(0, len(condition)-1):
#         if len(condition[i + 1]) >= 2:
#             if condition[i + 1][-1] == 0:
#                 if data[feature_set[condition[i][0]] - 1] < condition[i][1]:
#                     y = True
#                 else:
#                     y = False
#                     break
#             else:
#                 if data[feature_set[condition[i][0]] - 1] >= condition[i][1]:
#                     y = True
#                 else:
#                     y = False
#                     break
#     return y


def seed_test_input(clusters, limit, basic_label, feature_set, condition, original_dataset):
    def is_DT_condition(data, feature_set, condition):
        for i in range(0, len(condition) - 1):
            if len(condition[i + 1]) >= 2:
                if condition[i + 1][-1] == 0:
                    if data[feature_set[condition[i][0]] - 1] < condition[i][1]:
                        y = True
                    else:
                        y = False
                        break
                else:
                    if data[feature_set[condition[i][0]] - 1] >= condition[i][1]:
                        y = True
                    else:
                        y = False
                        break
        return y

    i = 0
    rows = []
    max_size = max([len(c[0]) for c in clusters])  # num of params?
    print("-->max_size:", max_size)

    # row1 = random.sample(cluster[0][0], limit/4)
    # row2 = random.sample(cluster[1][0], limit / 4)
    # row3 = random.sample(cluster[2][0], limit / 4)
    # row4 = random.sample(cluster[3][0], limit / 4)
    #
    # rows = row1 + row2 + row3 + row4

    while i < max_size:
        if len(rows) == limit:
            break
        for c in clusters:
            if i >= len(c[0]):
                continue
            row = c[0][i]

            # n = X[row]
            n = original_dataset[row][:-1]

            if is_DT_condition(n, feature_set, condition) == True:
                # label = np.argmax(model_prediction(sess, x, preds, np.array([n]))[0])
                label = original_dataset[row][-1]
                if label == basic_label:
                    # print("basic_label, label", basic_label, label)
                    rows.append(row)
                    if len(rows) == limit:
                        break
                # else:
                #     print("label != basic_label,", label, basic_label)

        i += 1
    return np.array(rows)
    # print("-->length of rows:", len(rows))
    # if len(rows) > limit:
    #     random_rows = random.sample(rows, limit)
    # else:
    #     random_rows = rows
    # return np.array(random_rows)


def get_cluster(dataset, cluster_num, feature_set):
    # all_conditions = r
    # condition = all_condition[i]
    data = {"census": census_data, "credit": credit_data, "bank": bank_data}
    data_config = {"census": census, "credit": credit, "bank": bank}

    X, Y, input_shape, nb_classes = data[dataset]()
    xx = []
    index_set = feature_set[:]
    for index_num in range(len(index_set)):
        if index_set[index_num] != -1:
            index_set[index_num] -= index_num + 1

    for n in X:
        n = n.tolist()
        length = len(n)
        for i in feature_set:
            if i > length:
                continue
            elif i < 0:
                continue
            n[i - 1] = 0
        xx.append(n)

    xx = np.array(xx)
    # print("-->xx:", xx, type(xx), xx.shape)

    if [] == xx.tolist():
        return []
    if len(xx) < 4:
        cluster_num = 1

    clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(xx)
    clusters = [np.where(clf.labels_ == i) for i in range(cluster_num)]

    # print("-->clusters:", clusters)
    return clusters

def sprt_detect(original_labels, predict_labels, k_value, threshold):

    def sprt_one_figure(prs, accept_pr, deny_pr, threshold, k_value):
        length = len(prs)
        Y = list(range(0, length))
        title_name = "threshold=" + str(threshold) + " (k=" + str(k_value) +")"
        plt.title(title_name)
        accept_prs = [accept_pr]*length
        deny_prs = [deny_pr]*length
        plt.plot(Y, accept_prs, color='black', linestyle="--", label="accept_bound")
        plt.plot(Y, deny_prs, color='black', linestyle=":", label="deny_bound")
        plt.plot(Y, prs, label="k=" + str(k_value))
        # plt.plot(sub_axix, test_acys, color='red', label='testing accuracy')
        # plt.plot(x_axix, train_pn_dis, color='skyblue', label='PN distance')
        # plt.plot(x_axix, thresholds, color='blue', label='threshold')
        plt.legend()
        plt.xlabel('number of detected samples')
        plt.ylabel('rate')
        plt.show()

    def calculate_sprt_ratio(c, n):
        '''
        :param c: number of model which lead to label changes
        :param n: total number of mutations
        :return: the sprt ratio
        '''
        p1 = threshold + sigma
        p0 = threshold - sigma

        return c * np.log(p1 / p0) + (n - c) * np.log((1 - p1) / (1 - p0))

    # threshold = 0.75
    sigma = 0.05
    beta = 0.05
    alpha = 0.05

    accept_pr = np.log((1 - beta) / alpha)
    deny_pr = np.log(beta / (1 - alpha))

    print("-->accept/deny pr:", accept_pr, deny_pr)
    print("-->p0, p1:", threshold + sigma, threshold - sigma)

    same_count = 0
    total_count = 0

    length = len(original_labels)

    prs = []
    for i in range(0, len(original_labels)):
        pr = calculate_sprt_ratio(same_count, total_count)
        total_count += 1
        o_label = original_labels[i]
        p_label = predict_labels[i]
        # pr = calculate_sprt_ratio(same_count, total_count)
        prs.append(pr)
        if o_label == p_label:
            same_count += 1
        if pr >= accept_pr:
            print("-->last pr:", pr)
            print("-->accept_pr", accept_pr)
            # sprt_one_figure(prs, accept_pr, deny_pr, k_value, threshold)
            prs[-1] = accept_pr
            return True, same_count, total_count, prs, accept_pr, deny_pr
        if pr <= deny_pr:
            print("-->last pr:", pr)
            print("-->deny_pr", deny_pr)
            prs[-1] = deny_pr
            # sprt_one_figure(prs, accept_pr, deny_pr, k_value, threshold)
            return False, same_count, total_count, prs, accept_pr, deny_pr
        if total_count >= len(original_labels):
            return 0, same_count, total_count, prs, accept_pr, deny_pr

    # random_list = random.sample([i for i in range(length)], 5000)
    # prs = []
    # for num in random_list:
    #     total_count += 1
    #     # print("-->total count", total_count)
    #     origi_label = original_labels[num]
    #     predict_label = predict_labels[num]
    #     pr = calculate_sprt_ratio(same_count, total_count)
    #     prs.append(pr)
    #     num += 1
    #     if origi_label == predict_label:
    #         same_count += 1
    #         if pr >= accept_pr:
    #             sprt_one_figure(prs, accept_pr, deny_pr, k_value, threshold)
    #             return True, same_count, total_count, prs, accept_pr, deny_pr
    #         if pr <= deny_pr:
    #             sprt_one_figure(prs, accept_pr, deny_pr, k_value, threshold)
    #             return False, same_count, total_count, prs, accept_pr, deny_pr
    #     if total_count >= len(random_list):
    #         return 0, same_count, total_count, prs, accept_pr, deny_pr

def sprt_detect_multiplethre(original_labels, predict_labels, k_value, thresholds):

    def calculate_sprt_ratio(c, n, threshold):
        '''
        :param c: number of model which lead to label changes
        :param n: total number of mutations
        :return: the sprt ratio
        '''
        p1 = threshold + sigma
        p0 = threshold - sigma

        return c * np.log(p1 / p0) + (n - c) * np.log((1 - p1) / (1 - p0))

    # threshold = 0.75
    sigma = 0.05
    beta = 0.05
    alpha = 0.05

    accept_pr = np.log((1 - beta) / alpha)
    deny_pr = np.log(beta / (1 - alpha))

    def calculate_prs(threshold):

        print("-->accept/deny pr:", accept_pr, deny_pr)
        print("-->p0, p1:", threshold + sigma, threshold - sigma)

        same_count = 0
        total_count = 0

        length = len(original_labels)

        prs = []
        for i in range(0, len(original_labels)):
            pr = calculate_sprt_ratio(same_count, total_count, threshold)
            total_count += 1
            o_label = original_labels[i]
            p_label = predict_labels[i]
            # pr = calculate_sprt_ratio(same_count, total_count)
            prs.append(pr)
            if o_label == p_label:
                same_count += 1
            if pr >= accept_pr:
                # sprt_one_figure(prs, accept_pr, deny_pr, k_value, threshold)
                prs[-1] = accept_pr
                return True, prs
                # return True, same_count, total_count, prs, accept_pr, deny_pr
            if pr <= deny_pr:
                prs[-1] = deny_pr
                # sprt_one_figure(prs, accept_pr, deny_pr, k_value, threshold)
                return False, prs
                # return False, same_count, total_count, prs, accept_pr, deny_pr
            if total_count >= len(original_labels):
                return 0, prs
                # return 0, same_count, total_count, prs, accept_pr, deny_pr

    def multi_thr_figure(all_prs, accept_pr, deny_pr):
        length = max(len(p) for p in all_prs)
        Y = list(range(0, length))
        title_name = " (k=" + str(k_value) + ") multiple threshold:" + str(thresholds)
        plt.title(title_name)
        accept_prs = [accept_pr] * length
        deny_prs = [deny_pr] * length
        plt.plot(Y, accept_prs, color='black', linestyle="--", label="accept_bound")
        plt.plot(Y, deny_prs, color='black', linestyle=":", label="deny_bound")
        i = 0
        for i in range(0, len(all_prs)):
            prs = all_prs[i]
            plt.plot(list(range(0, len(prs))), prs, label=str(thresholds[i]))
            i += 1
        # plt.legend()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
        plt.xlabel('number of detected samples')
        plt.ylabel('rate')
        plt.show()


    all_prs = []
    all_result = []
    for threshold in thresholds:
        print("-->threshold:", threshold)
        result, prs = calculate_prs(threshold)
        all_prs.append(prs)
        all_result.append(result)
    multi_thr_figure(all_prs, accept_pr, deny_pr)
    return all_result, all_prs

def same_ratio(list1, list2):
    same_count = 0
    total_count = 0
    for i in range(0, len(list1)):
        total_count += 1
        if list1[i] == list2[i]:
            same_count += 1
    print("-->same ratio", float(same_count)/total_count)

def generate_random_data(max_num, conf):
    # conf = data_config[dataset]
    params = conf.params
    all_data = []
    while len(all_data) < max_num:
        data = []
        for i in range(params):
            d = random.randint(conf.input_bounds[i][0], conf.input_bounds[i][1])
            data.append(d)
        all_data.append(data)
    return all_data


def interpretability(filename, dataset, directorys, k_values, thresholds):
    data = {"census": census_data, "credit": credit_data, "bank": bank_data}
    data_config = {"census": census, "credit": credit, "bank": bank}

    params = data_config[dataset].params
    X, Y, input_shape, nb_classes = data[dataset]()
    config = tf.ConfigProto()
    conf = data_config[dataset]
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)
    # print("-->preds ", preds)
    saver = tf.train.Saver()
    # model_file = "../retrained_models/" + dataset + "_df/999/test.model"
    model_file = "../models/" + dataset + "/test.model"
    saver.restore(sess, model_file)
    grad_0 = gradient_graph(x, preds)
    tfops = tf.sign(grad_0)

    # process dataset
    dataset_list = load_csv(filename)
    del dataset_list[0]
    for i in range(len(dataset_list[0])):
        str_column_to_float(dataset_list, i)
    print("-->dataset:", np.array(dataset_list))
    print(np.array(dataset_list).shape)
    model_dataset = []
    row_data = []
    for d in dataset_list:
        del (d[-1])
        row_data.append(d)
        probs = model_prediction(sess, x, preds, np.array([d]))[0]  # n_probs: prediction vector
        label = np.argmax(probs)  # GET index of max value in n_probs
        d.append(label)
        model_dataset.append(d)
    print("-->dataset:", np.array(model_dataset))
    print(np.array(model_dataset).shape)
    original_dataset = model_dataset

    ######
    # use DT with highest accuracy
    def get_dt(directory, k_value):
        tree_file = directory + "DT_trees"
        all_DT_trees = []
        # print("-->tree_file", tree_file)
        with open(tree_file, 'r') as f:
            for line in f:
                all_DT_trees.append(line.split("\n")[0])
        accuracy_file = directory + "accuracy"
        accuracy = []
        max_accuracy_feature = 0
        max_accuracy = 0
        with open(accuracy_file, 'r') as f:  # 1
            lines = f.readlines()  # 2
            i = 0
            for line in lines:  # 3
                value = [float(s) for s in line.split()]  # 2
                accuracy.append(value[0])  # 5
                if value[0] > max_accuracy:
                    max_accuracy = value[0]
                    max_accuracy_feature = i
                i += 1
        all_feature_set = list(range(1, params + 1))
        feature_sets = list(itertools.combinations(all_feature_set, k_value))
        print("-->selected feature_set:", feature_sets[max_accuracy_feature], max_accuracy_feature, max_accuracy)
        # if k_value == 3:
        #     max_accuracy_feature = 170
        #     print(feature_sets[max_accuracy_feature])
        # if k_value == 2:
        #     max_accuracy_feature = 50  # 72
        #     print(feature_sets[max_accuracy_feature])
        feature_sets = [feature_sets[max_accuracy_feature]]
        return feature_sets, all_DT_trees[max_accuracy_feature]

    def get_labels1(tree):
        original_labels = [i[-1] for i in original_dataset]
        predict_labels = []
        for d in row_data:
            d = map(int, d)
            label = predict(tree, d)
            predict_labels.append(label)
        return original_labels, predict_labels

    def get_labels(tree, test_datas):
        original_labels = []
        predict_labels = []
        for d in test_datas:
            probs = model_prediction(sess, x, preds, np.array([d]))[0]  # n_probs: prediction vector
            model_label = np.argmax(probs)  # GET index of max value in n_probs
            original_labels.append(model_label)
            tree_label = predict(tree, d, feature_sets[0])
            predict_labels.append(tree_label)
        return original_labels, predict_labels

    def one_num(list):
        num = 0
        for l in list:
            if l == 1:
                num += 1
        return num


    def sprt_three_figure(all_prs, accept_pr, deny_pr, threshold, k_values):
        prs1 = all_prs[0]
        prs2 = all_prs[1]
        prs3 = all_prs[2]
        k_value1 = k_values[0]
        k_value2 = k_values[1]
        k_value3 = k_values[2]

        length = max(len(prs1), len(prs2), len(prs3))
        Y = list(range(0, length))
        title_name = "threshold=" + str(threshold)
        plt.title(title_name)
        accept_prs = [accept_pr]*length
        deny_prs = [deny_pr]*length
        plt.plot(Y, accept_prs, color='black', linestyle="--", label="accept_bound")
        plt.plot(Y, deny_prs, color='black', linestyle=":", label="deny_bound")

        plt.plot(list(range(0, len(prs1))), prs1, color='red', label="k=" + str(k_value1))
        plt.plot(list(range(0, len(prs2))), prs2, color='blue', label="k=" + str(k_value2))
        plt.plot(list(range(0, len(prs3))), prs3, color='green', label="k=" + str(k_value3))
        # plt.legend()
        # plt.legend(loc=[0, 1])
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
        plt.xlabel('number of detected samples')
        plt.ylabel('rate')
        plt.show()

    def sprt_four_figure(all_prs, accept_pr, deny_pr, threshold, k_values):
        prs1 = all_prs[0]
        prs2 = all_prs[1]
        prs3 = all_prs[2]
        prs4 = all_prs[3]
        k_value1 = k_values[0]
        k_value2 = k_values[1]
        k_value3 = k_values[2]
        k_value4 = k_values[3]

        length = max(len(prs1), len(prs2), len(prs3), len(prs4))
        Y = list(range(0, length))
        title_name = "threshold=" + str(threshold)
        plt.title(title_name)
        accept_prs = [accept_pr]*length
        deny_prs = [deny_pr]*length
        plt.plot(Y, accept_prs, color='black', linestyle="--", label="accept_bound")
        plt.plot(Y, deny_prs, color='black', linestyle=":", label="deny_bound")

        plt.plot(list(range(0, len(prs1))), prs1, color='red', label="k=" + str(k_value1))
        plt.plot(list(range(0, len(prs2))), prs2, color='blue', label="k=" + str(k_value2))
        plt.plot(list(range(0, len(prs3))), prs3, color='green', label="k=" + str(k_value3))
        plt.plot(list(range(0, len(prs4))), prs4, color='purple', label="k=" + str(k_value4))
        # plt.legend()
        # plt.legend(loc=[0, 1])
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
        plt.xlabel('number of detected samples')
        plt.ylabel('rate')
        plt.show()

    def sprt_one_figure(prs, accept_pr, deny_pr, threshold, k_value):
        length = len(prs)
        Y = list(range(0, length))
        title_name = "threshold=" + str(threshold) + " (k=" + str(k_value) +")"
        plt.title(title_name)
        accept_prs = [accept_pr]*length
        deny_prs = [deny_pr]*length
        plt.plot(Y, accept_prs, color='black', linestyle="--", label="accept_bound")
        plt.plot(Y, deny_prs, color='black', linestyle=":", label="deny_bound")
        plt.plot(Y, prs, label="k=" + str(k_value))
        # plt.plot(sub_axix, test_acys, color='red', label='testing accuracy')
        # plt.plot(x_axix, train_pn_dis, color='skyblue', label='PN distance')
        # plt.plot(x_axix, thresholds, color='blue', label='threshold')
        plt.legend()
        plt.xlabel('number of detected samples')
        plt.ylabel('rate')
        plt.show()

    all_prs = []
    random_test_data = generate_random_data(1000, conf)
    print("-->random_test_data:", random_test_data)
    for i in range(0, len(directorys)):
        directory = directorys[i]
        k_value = k_values[i]

        print("-->dir, k", directory, k_value)

        feature_sets, tree = get_dt(directory, k_value)
        print("-->feature_set", feature_sets[0])

        print("-->tree", tree)

        tree = dict(eval(tree))
        original_labels, predict_labels = get_labels(tree, random_test_data)
        same_ratio(original_labels, predict_labels)
        print("-->one num", one_num(original_labels), one_num(predict_labels))

        print("-->original labels", original_labels)
        print("-->predict labels", predict_labels)
        # print("-->sprt result:", sprt_detect(original_labels, predict_labels, threshold, k_value))

        if_accept, same_count, total_count, prs, accept_pr, deny_pr = sprt_detect(original_labels, predict_labels, k_value, threshold)
        print("-->sprt result:", if_accept, same_count, total_count)
        all_prs.append(prs)

        sprt_detect_multiplethre(original_labels, predict_labels, k_value, thresholds)

    # test
    # sprt_three_figure(all_prs, accept_pr, deny_pr, threshold, k_values)
    # sprt_four_figure(all_prs, accept_pr, deny_pr, threshold, k_values)
    # sprt_one_figure(prs, accept_pr, deny_pr, k_value, threshold)

    return


total_gini = 0
count_gini = 0
filename = '../datasets/census'
dataset = "census"
directory1 = "../retrained_result/credit/k=2(all)/"
directory2 = "../retrained_result/credit/k=3(all)/"
directory3 = "../retrained_result/credit/k=4(all)/"
directory4 = "../retrained_result/credit/k=5(all)/"

directorys = [directory1, directory2, directory3, directory4]
k_values = [2, 3, 4]

# directory4 = "../result/bank/k=5/"
# directorys = [directory1, directory2, directory3, directory4]
# k_values = [2, 3, 4, 5]

threshold = 0.85

# test
# directorys = [directory2]
# k_values = [3]
# end test

directory1 = "../allinput10000/census/k=2(all)/"
# directory1 = "../retrained_result/credit/k=2(all)/"
#
directorys = [directory1]
k_values = [2]
# thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.94, 0.95]
thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.94]
start = time.clock()
prs1 = interpretability(filename, dataset, directorys, k_values, thresholds)
end = time.clock()

print("-->time", end-start)

# interpretability(filename, dataset, directorys, k_values, thresholds)


# tree = {'index': 1, 'right': {'index': 0, 'right': {'index': 2, 'right': 0.0, 'value': 6.0, 'left': 0.0}, 'value': 4.0, 'left': {'index': 0, 'right': 0.0, 'value': 3.0, 'left': 0.0}}, 'value': 1.0, 'left': {'index': 2, 'right': {'index': 2, 'right': 0.0, 'value': 8.0, 'left': 0.0}, 'value': 6.0, 'left': {'index': 2, 'right': 1.0, 'value': 4.0, 'left': 0.0}}}
# print_tree(tree)







