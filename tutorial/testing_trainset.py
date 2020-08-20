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
import time


# random = np.random.RandomState(0)

# def seed_test_input(clusters, limit):
#     """
#     Select the seed inputs for fairness testing
#     :param clusters: the results of K-means clustering
#     :param limit: the size of seed inputs wanted
#     :return: a sequence of seed inputs
#     """
#     i = 0
#     rows = []
#     max_size = max([len(c[0]) for c in clusters]) #num of params?
#     while i < max_size:
#         if len(rows) == limit:
#             break
#         for c in clusters:
#             if i >= len(c[0]):
#                 continue
#             row = c[0][i]
#             rows.append(row)
#             if len(rows) == limit:
#                 break
#         i += 1
#     return np.array(rows)

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
    inputs = seed_test_input(clusters, min(1000, len(xx)), xx)  # 1000
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


def interpretability(filename, dataset, max_iter, k, n_folds, f_gini, f_accuracy, f_time, f_ci, f_iteration, f_trees):
    data = {"census": census_data, "credit": credit_data, "bank": bank_data}
    data_config = {"census": census, "credit": credit, "bank": bank}

    X, Y, input_shape, nb_classes = data[dataset]()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)
    # print("-->preds ", preds)
    saver = tf.train.Saver()
    saver.restore(sess, "../models/bank/test.model")
    grad_0 = gradient_graph(x, preds)
    tfops = tf.sign(grad_0)

    dataset_list = load_csv(filename)
    del dataset_list[0]
    for i in range(len(dataset_list[0])):
        str_column_to_float(dataset_list, i)
    # print("-->dataset:", np.array(dataset_list))
    # print(np.array(dataset_list).shape)

    new_dataset = []
    for d in dataset_list:
        del (d[-1])
        # d_plus = clip(np.array([d]), data_config[dataset_name]).astype("int")
        # d = d_plus[0]

        #clip d in dataset_list
        # d = clip(d, data_config[dataset])
        # d = list(np.array(d).astype("int"))

        # print(d, type(d), type(d[0]))
        # d = np.array([d])
        probs = model_prediction(sess, x, preds, np.array([d]))[0]  # n_probs: prediction vector
        label = np.argmax(probs)  # GET index of max value in n_probs
        prob = probs[label]
        # d = np.array(d, label)
        d.append(label)
        # print(d)
        new_dataset.append(d)

    # print("-->dataset:", np.array(new_dataset))
    # print(np.array(new_dataset).shape)
    original_dataset = new_dataset

    def decision_tree_accuracy(feature_set):
        seed(1)
        original_data = get_DT_cluster(original_dataset, cluster_num, feature_set, params)
        print(len(original_data))
        scores, dif_scores, trees = evaluate_algorithm(original_data, decision_tree, n_folds, max_depth, min_size)
        # print("-->scores, dif_scores:", scores, dif_scores)
        all_scores = []
        all_scores.append(scores)
        all_scores.append(sum([s[1] for s in dif_scores]) / float(len(dif_scores)))
        all_scores.append(sum([s[2] for s in dif_scores]) / float(len(dif_scores)))
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
        # print("-->dif_scores:", dif_scores)
        print('0 Mean Accuracy: %.3f%%' % (sum([s[1] for s in dif_scores]) / float(len(dif_scores))))
        print('1 Mean Accuracy: %.3f%%' % (sum([s[2] for s in dif_scores]) / float(len(dif_scores))))

        f_accuracy.write(str(sum(scores) / float(len(scores))) + " ")
        f_accuracy.write(str(sum([s[1] for s in dif_scores]) / float(len(dif_scores))) + " ")
        f_accuracy.write(str(sum([s[2] for s in dif_scores]) / float(len(dif_scores))) + "\n")

        max_index = scores.index(max(scores))

        return all_scores, trees[max_index]

    def perturbation(sess, preds, x, feature_set, condition, clusters, limit, original_dataset):
        # grad_0 = gradient_graph(x, preds)
        # print("-->feature_set1:", feature_set)

        # inputs = get_cluster(sess, x, preds, dataset, cluster_num, feature_set, condition)
        basic_label = condition[-1][0]
        inputs = seed_test_input(clusters, limit, basic_label, feature_set, condition, original_dataset)
        # print("-->inputs:", inputs)

        length = len(inputs)
        print("-->length1", length)

        seed_num = 0
        ci_num = 0
        r = False
        itr_num = 0
        get_CI = False
        final_itr_num = 0
        zero_gradient_itr = 0

        # print("-->inputs", inputs)

        for num in range(len(inputs)):
            # print("-->seed iteration: ", num)
            seed_num += 1

            index = inputs[num]
            sample = original_dataset[index][:-1]
            sample = np.array([sample])
            # sample = X[index:index + 1]
            # sample = X[index]
            # print("-->sample:", sample)
            # probs = model_prediction(sess, x, preds, sample)[0]
            # label = np.argmax(probs)  # index of maximum probability in prediction
            # label1 = original_dataset[index][-1]
            # if label != label1:
            #     print("label != label1")
            # if label != basic_label:
            #     print("label != basic_label")
            # print("-->basic_label:", label)

            for iter in range(max_iter + 1):  # 10
                # print("--> global iteration:", iter)
                itr_num += 1
                # print("--> sample:", sample)

                s_grad = sess.run(tfops, feed_dict={x: sample})
                g_diff = s_grad[0]
                # print("-->g_diff", g_diff)

                # features in feature_set unchange
                # print("-->index in feature set:", feature_set)
                for index in feature_set:
                    g_diff[index - 1] = 0
                # print("-->g_diff", g_diff)

                if np.zeros(input_shape[1]).tolist() == g_diff.tolist():
                    # print("-->0 gradient")
                    # zero_gradient_itr += 1
                    # index = np.random.randint(len(g_diff) - 1)
                    # g_diff[index] = 1.0*
                    break

                # sample[0] = clip(sample[0] + perturbation_size * g_diff, data_config[dataset]).astype("int")
                # n_sample = sample.copy()
                # print("1-->n_sample:", n_sample)

                n_sample = []
                new_sample = clip(sample[0] + perturbation_size * g_diff, data_config[dataset])
                n_sample.append(new_sample)
                n_sample = np.array(n_sample)
                # print("2-->n_sample:", n_sample)

                n_probs = model_prediction(sess, x, preds, n_sample)[0]  # n_probs: prediction vector
                n_label = np.argmax(n_probs)  # GET index of max value in n_probs
                # print("-", n_label)

                if n_label != basic_label:
                    # print("-->label != n_label")
                    # print("final label:", label, n_label)
                    # print("-->n_sample:", n_sample)
                    ci_num += 1
                    if get_CI == False:
                        final_itr_num = itr_num
                    get_CI = True
                    r = True
                    break
                    # return True
        # return False
        print(r, ci_num, seed_num, final_itr_num)
        return r, ci_num, seed_num, final_itr_num

    all_feature_set = list(range(1, data_config[dataset].params + 1))
    cluster_num = 4
    params = data_config[dataset].params
    max_depth = 2
    min_size = 10
    feature_sets = list(itertools.combinations(all_feature_set, k))
    print(feature_sets)

    DT_file_index = 0
    scores = []
    feature_sets = [(12, 16)]
    for feature_set in feature_sets:
        print("-->feature_set", feature_set)
        # decision tree
        # tree = all_DT_trees[DT_file_index]
        # tree = dict(eval(tree))

        DT_file_index += 1

        start1 = time.clock()
        score, tree = decision_tree_accuracy(feature_set)
        end1 = time.clock()
        f_trees.write(str(tree) + "\n")
        f_time.write(str(end1 - start1) + " ")

        # perturbation
        print("-->tree:", tree)
        tree_conditions = []
        get_conditions(tree, result=tree_conditions, dir=-1, tmp=[])
        print("-->tree_condition:", tree_conditions)
        # print(tree_conditions[15])
        all_result = []
        all_general_result = []
        results = []
        number = 1
        feature_set = list(feature_set)
        all_ci_num = 0
        all_seed_num = 0
        all_itr_num = 0

        limit = 1000
        clusters = get_cluster(dataset, cluster_num, feature_set)

        tree_brench = len(tree_conditions)

        # set tree conditions
        # tree_conditions = [tree_conditions[6]]

        start2 = time.clock()
        for condition in tree_conditions:
            print("sequence:", number, condition)
            result, ci_num, seed_num, itr_num = perturbation(sess, preds, x, feature_set, condition, clusters, limit,
                                                             original_dataset)
            # sess, preds, x, feature_set, condition, clusters, limit
            all_ci_num += ci_num
            all_seed_num += seed_num
            results.append(result)
            print("-->result:", result)
            if result == True:
                all_itr_num += itr_num
            number += 1
        all_result.append(results)
        true_num = results.count(True)
        print("-->results:", results)
        print("-->counter instance:", all_ci_num, all_seed_num, all_ci_num / float(all_seed_num))
        print("-->iteration num:", all_itr_num / float(true_num))

        # file 2 counter instance
        f_ci.write(str(all_ci_num) + " " + str(all_seed_num) + " " + str(all_ci_num / float(all_seed_num)) + "\n")

        # file 3 iteration num
        f_iteration.write(str(all_itr_num / float(true_num)) + " " + str(true_num / float(tree_brench)) + "\n")

        if len(results) == len(tree_conditions):
            if not any(results):
                print("-->used features:", feature_set)
                print("-->all_results:", all_result)
                print("-->interpretable!")
                break

        end2 = time.clock()
        f_time.write(str(end2 - start2) + "\n")

    return


total_gini = 0
count_gini = 0
filename = '../datasets/bank'

root_filename = "../result_file/kk/"

gini_input = root_filename + 'gini'
accuracy_input = root_filename + 'accuracy'
time_consume = root_filename + 'time'
counter_instance = root_filename + 'CI_num'
iteration_num = root_filename + 'iteration_num'
DT_trees = root_filename + 'DT_trees'

f_gini = open(gini_input, "w")
f_accuracy = open(accuracy_input, "w")
f_time = open(time_consume, "w")
f_ci = open(counter_instance, "w")
f_iteration = open(iteration_num, "w")
f_trees = open(DT_trees, "w")
f_gini.truncate()
f_accuracy.truncate()
f_time.truncate()
f_ci.truncate()
f_iteration.truncate()
f_trees.truncate()

dataset = "bank"

interpretability(filename, dataset, 10, 2, 3, f_gini, f_accuracy, f_time, f_ci, f_iteration, f_trees)
# dataset, max_iter, k, original_dataset, n_folds, x, preds



