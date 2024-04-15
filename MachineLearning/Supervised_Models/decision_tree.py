import numpy as np
import pandas as pd
import math
import sys

from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

class DecisionNode:
    """
    Class for a parent/leaf node in the decision tree.
    A Node with node information about it's left and right nodes if any. it has the impurity info also.
    """
    def __init__(self, impurity=None, question=None, feature_index=None, threshold=None,
                 true_subtree=None, false_subtree=None):
        """
        :param
        """
        self.impurity = impurity
        # Which question to ask , to split the dataset.
        self.question = question 
        # Index of the feature which make the best fit for this node.
        self.feature_index = feature_index
        # The threshold value for that feature to make the split.
        self.threshold = threshold
        # DecisionNode Object of the left subtree.
        self.true_left_subtree = true_subtree
        # DecisionNode Object of the right subtree.
        self.false_right_subtree = false_subtree

class LeafNode:
    """ Leaf Node of the decision tree."""
    def __init__(self, value):
        self.prediction_value = value


class DecisionTree:
    """Common class for making decision tree for classification and regression tasks."""
    def __init__(self, min_sample_split=3, min_impurity=1e-7, max_depth=float('inf'),
                 impurity_function=None, leaf_node_calculation=None):
        """
        """
        self.root = None

        self.min_sample_split = min_sample_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.impurity_function = impurity_function
        self.leaf_node_calculation = leaf_node_calculation

    def _partition_dataset(self, Xy, feature_index, threshold):
        """Split the dataset based on the given feature and threshold.
        
        """
        split_func = None
        if isinstance(threshold, int) or isinstance(threshold, float):
            split_func = lambda sample: sample[feature_index] >= threshold
        else:
            split_func = lambda sample: sample[feature_index] == threshold

        X_1 = np.array([sample for sample in Xy if split_func(sample)])
        X_2 = np.array([sample for sample in Xy if not split_func(sample)])

        return X_1, X_2

    def _find_best_split(self, Xy):
        """ Find the best question/best feature threshold which splits the data well.
        
        """
        best_question = tuple() # this will containe the feature and its value which make the best split(higest gain).
        best_datasplit = {} # best data split.
        largest_impurity = 0
        n_features = (Xy.shape[1] - 1)
        # iterate over all the features.
        for feature_index in range(n_features):
            # find the unique values in that feature.
            unique_value = set(s for s in Xy[:,feature_index])
            # iterate over all the unique values to find the impurity.
            for threshold in unique_value:
                # split the dataset based on the feature value.
                true_xy, false_xy = self._partition_dataset(Xy, feature_index, threshold)
                # skip the node which has any on type 0. because this means it is already pure.
                if len(true_xy) > 0 and len(false_xy) > 0:
                    

                    # find the y values.
                    y = Xy[:, -1]
                    true_y = true_xy[:, -1]
                    false_y = false_xy[:, -1]

                    # calculate the impurity function.
                    impurity = self.impurity_function(y, true_y, false_y)

                    # if the calculated impurity is larger than save this value for comaparition.
                    if impurity > largest_impurity:
                        largest_impurity = impurity
                        best_question = (feature_index, threshold)
                        best_datasplit = {
                                    "leftX": true_xy[:, :n_features],   # X of left subtree
                                    "lefty": true_xy[:, n_features:],   # y of left subtree
                                    "rightX": false_xy[:, :n_features],  # X of right subtree
                                    "righty": false_xy[:, n_features:]   # y of right subtree
                        }
                    
        return largest_impurity, best_question, best_datasplit

    def _build_tree(self, X, y, current_depth=0):
        """
        This is a recursive method to build the decision tree.
        """
        n_samples , n_features = X.shape
        # Add y as last column of X
        Xy = np.concatenate((X, y), axis=1)
        # find the Information gain on each feature each values and return the question which splits the data very well
        # based on the impurity function. (classfication - Information gain, regression - variance reduction).
        if (n_samples >= self.min_sample_split) and (current_depth <= self.max_depth):
            # find the best split/ which question split the data well.
            impurity, quesion, best_datasplit = self._find_best_split(Xy)
            if impurity > self.min_impurity:
            # Build subtrees for the right and left branch.
                true_branch = self._build_tree(best_datasplit["leftX"], best_datasplit["lefty"], current_depth + 1)
                false_branch = self._build_tree(best_datasplit["rightX"], best_datasplit["righty"], current_depth + 1)
                return DecisionNode( impurity=impurity, question=quesion, feature_index=quesion[0], threshold=quesion[1],
                                    true_subtree=true_branch, false_subtree=false_branch)

        leaf_value = self._leaf_value_calculation(y)
        return LeafNode(value=leaf_value)


    def train(self, X, y):
        """
        Build the decision tree.

        :param X: Train features/dependant values.
        :param y: train target/independant value.
        """
        self.root = self._build_tree(X, y, current_depth=0)

    def predict_sample(self, x, tree=None):
        """move form the top to bottom of the tree make a prediction of the sample by the
            value in the leaf node """
        if tree is None:
            tree = self.root
        # if it a leaf node the return the prediction.
        if isinstance(tree , LeafNode):

            return tree.prediction_value
        feature_value = x[tree.feature_index]

        branch = tree.false_right_subtree

        if isinstance(feature_value, int) or isinstance(feature_value, float):
            
            if feature_value >= tree.threshold:

                branch = tree.true_left_subtree
        elif feature_value == tree.threshold:
            branch = tree.true_left_subtree

        return self.predict_sample(x, branch)

    def predict(self, test_X):
        """ predict the unknow feature."""
        x = np.array(test_X)
        y_pred = [self.predict_sample(sample) for sample in x]
        # y_pred = np.array(y_pred)
        # y_pred = np.expand_dims(y_pred, axis = 1)
        return y_pred
    
    def draw_tree(self, tree = None, indentation = " "):
        """print the whole decitions of the tree from top to bottom."""
        if tree is None:
            tree = self.root

        def print_question(question, indention):
            """
            :param question: tuple of feature_index and threshold.
            """
            feature_index = question[0]
            threshold = question[1]

            condition = "=="
            if isinstance(threshold, int) or isinstance(threshold, float):
                condition = ">="
            print(indention,"Is {col}{condition}{value}?".format(col=feature_index, condition=condition, value=threshold))

        if isinstance(tree , LeafNode):
            print(indentation,"The predicted value -->", tree.prediction_value)
            return
        
        else:
            # print the question.
            print_question(tree.question,indentation)
            if tree.true_left_subtree is not None:
                # travers to the true left branch.
                print (indentation + '----- True branch :)')
                self.draw_tree(tree.true_left_subtree, indentation + "  ")
            if tree.false_right_subtree is not None:
                # travers to the false right-side branch.
                print (indentation + '----- False branch :)')
                self.draw_tree(tree.false_right_subtree, indentation + "  ")


class DecisionTreeClassifier(DecisionTree):
    """ Decision Tree for the classification problem."""
    def __init__(self, min_sample_split=3, min_impurity=1e-7, max_depth=float('inf'),
                 ):
        """
        :param min_sample_split: min value a leaf node must have.
        :param min_impurity: minimum impurity.
        :param max_depth: maximum depth of the tree.
        """
        self._impurity_function = self._claculate_information_gain
        self._leaf_value_calculation = self._calculate_majarity_class
        super(DecisionTreeClassifier, self).__init__(min_sample_split=min_sample_split, min_impurity=min_impurity, max_depth=max_depth,
                         impurity_function=self._impurity_function, leaf_node_calculation=self._leaf_value_calculation)
    
    def _entropy(self, y):
        """ Find the entropy for the given data"""
        entropy = 0
        unique_value = np.unique(y)
        for val in unique_value:
            # probability of that class.
            p = len(y[y==val]) / len(y)
            entropy += -p * (math.log(p) / math.log(2))
        return entropy


    def _claculate_information_gain(self, y, y1, y2):
        """
        Calculate the information gain.

        :param y: target value.
        :param y1: target value for dataset in the true split/right branch.
        :param y2: target value for dataset in the false split/left branch.
        """
        # propobility of true values.
        p = len(y1) / len(y)
        entropy = self._entropy(y)
        info_gain = entropy - p * self._entropy(y1) - (1 - p) * self._entropy(y2)
        return info_gain       

    def _calculate_majarity_class(self, y):
        """
        calculate the prediction value for that leaf node.
        
        :param y: leaf node target array.
        """
        most_frequent_label = None
        max_count = 0
        unique_labels = np.unique(y)
        # iterate over all the unique values and find their frequentcy count.
        for label in unique_labels:
            count = len( y[y == label])
            if count > max_count:
                most_frequent_label = label
                max_count = count
        return most_frequent_label

    def train(self, X, y):
        """
        Build the tree.

        :param X: Feature array/depentant values.
        :parma y: target array/indepentant values.
        """
        # train the model.
        super(DecisionTreeClassifier, self).train(X, y)
    
    def predict(self, test_X):
        """ predict the unknow feature."""
        y_pred = super(DecisionTreeClassifier, self).predict(test_X)
        y_pred = np.array(y_pred)
        y_pred = np.expand_dims(y_pred, axis = 1)
        return y_pred
    
class DecisionTreeRegression(DecisionTree):
    """ Decision Tree for the classification problem."""
    def __init__(self, min_sample_split=3, min_impurity=1e-7, max_depth=float('inf'),
                 ):
        """
        :param min_sample_split: min value a leaf node must have.
        :param min_impurity: minimum impurity.
        :param max_depth: maximum depth of the tree.
        """
        self._impurity_function = self._claculate_variance_reduction
        self._leaf_value_calculation = self._calculate_colum_mean
        super(DecisionTreeRegression, self).__init__(min_sample_split=min_sample_split, min_impurity=min_impurity, max_depth=max_depth,
                         impurity_function=self._impurity_function, leaf_node_calculation=self._leaf_value_calculation)
    

    def _claculate_variance_reduction(self, y, y1, y2):
        """
        Calculate the Variance reduction.

        :param y: target value.
        :param y1: target value for dataset in the true split/right branch.
        :param y2: target value for dataset in the false split/left branch.
        """
        # propobility of true values.
        variance = np.var(y)
        variance_y1 = np.var(y1)
        variance_y2 = np.var(y2)

        y_len = len(y)
        fraction_1 = len(y1) / y_len 
        fraction_2 = len(y2) / y_len
        variance_reduction = variance - (fraction_1 * variance_y1 + fraction_2 * variance_y2)
        return  variance_reduction

    def _calculate_colum_mean(self, y):
        """
        calculate the prediction value for that leaf node using mean.
        
        :param y: leaf node target array.
        """
        mean = np.mean(y, axis=0)
        return mean

    def train(self, X, y):
        """
        Build the tree.

        :param X: Feature array/depentant values.
        :parma y: target array/indepentant values.
        """
        # train the model.
        super(DecisionTreeRegression, self).train(X, y)
    
    