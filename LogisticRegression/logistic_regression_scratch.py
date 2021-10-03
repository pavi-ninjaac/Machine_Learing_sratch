import numpy as np
from scipy.sparse import csc
from sklearn.datasets import make_blobs
"""
1. i need LogisticRegression class
2. I need __init__ function with learning rate and #iteration parameter
3. I need train function with X,y parameter
4. I need predict function
5. I need sigmoid function for gradiant decent and hypothesis
"""

class LogisticRegression:
    def __init__(self, learning_rate = 0.1, iteration = 10000):
        """
        :param learning_rate: A samll value needed for gradient decent, default value id 0.1.
        :param iteration: Number of training iteration, default value is 10,000.
        """
        self.lr = learning_rate
        self.it = iteration
    
    def cost_function(self, y, y_pred):
        """
        :param y: Original target value.
        :param y_pred: predicted target value.
        """
        return -1 / self.m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # hypothesis function.   
    def sigmoid(self, z):
        """
        :param z: Value to calculate sigmoid.
        """
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        """
        :param X: training data feature values ---> N Dimentional vector.
        :param y: training data target value -----> 1 Dimentional array.
        """
        # Target value should be in the shape of (n, 1) not (n, ).
        # So, this will check that and change the shape to (n, 1), if not.
        try:
            y.shape[1]
        except IndexError as e:
            # we need to change it to the 1 D array, not a list.
            print("ERROR: Target array should be a one dimentional array not a list"
                  "----> here the target value not in the shape of (n,1). \nShape ({shape_y_0},1) and {shape_y} not match"
                  .format(shape_y_0 = y.shape[0] , shape_y = y.shape))
            return 
            
        print(y.shape)

        # m is number of training samples.
        self.m  = X.shape[0]
        # n is number of features/columns/dependant variables.
        self.n = X.shape[1]

        # Set the initial weight.
        self.w = np.zeros((self.n , 1))
        print("self.w",self.w)
        # bias.
        self.b = 0

        for it in range(1, self.it+1):
            # 1. Find the predicted value.
            # 2. Find the Cost function.
            # 3. Find the derivation of weights and bias.
            # 4. Apply Gradient Decent.

            y_pred = self.sigmoid(np.dot(X, self.w) + self.b)
            print("y_pred",y_pred)

            cost = self.cost_function(y, y_pred)
            print("Cost",cost)

            # Derivation of w and b.
            print("self.m",self.m)
            dw = 1 / self.m * np.dot(X.T, (y_pred - y))
            print("dw",dw)
            db = 1 / self.m * np.sum(y_pred - y)
            print("db",db)

            # Chnage the parameter value/ apply Gradient decent.
            self.w -= self.lr * dw
            print("self.w after update",self.w)
            self.b -= self.lr * db
            print("self.b after update",self.b)

            if it % 1000 == 0:
                print("The Cost function for the iteration {}----->{} :)".format(it, cost))
    
    def predict(self, test_X):
        """
        :param: test_X: Values need to be predicted.
        """
        y_pred = self.sigmoid(np.dot(test_X, self.w) + self.b)
        # output of the sigmoid function is between [0 - 1], then need to convert it to class values either 0 or 1.
        y_pred_class = y_pred >= 0.5
        return y_pred_class
        

# Define the traning data.
X, y = make_blobs(n_samples=10, centers=2)
#print("y.shape",y.shape[1])
y = y[:, np.newaxis]
print(y.shape)
print("="*100)
print("Number of training data samples-----> {}".format(X.shape[0]))
print("Number of training features --------> {}".format(X.shape[1]))

#define the parameters
param = {
    "learning_rate" : 0.1,
    "iteration" : 1
}
print("="*100)
log_reg = LogisticRegression(**param)
log_reg.train(X, y)
y_pred = log_reg.predict(X)
print(y_pred)
print(y)
print(y==y_pred)
acc = (np.sum(y==y_pred)/X.shape[0]) 
print("="*100)
print("Accuracy of the prediction is {}".format(acc))


