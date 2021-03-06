import numpy as np
class ELM:
    def __init__(self, features, hidden_nodes, activation):
        self.features = features
        self.hidden_nodes = hidden_nodes
        self.activation = activation

        self.input_weights = None
        self.output_weights = None  # beta
        self.bias = None

    def relu(self, h):
        return np.maximum(h, 0, h)

    def sigmoid(self, h):
        return 1 / (1 + np.exp(-h))

    def hidden_layer(self, x, win, b):
        h = np.dot(x, win) + b
        if self.activation != None:
            h = self.activation(h)
        return h

    def fit(self, x_train, y_train):
        self.input_weights = np.random.normal(size=(x_train.shape[1], self.hidden_nodes))
        bias = np.random.normal(size=(x_train.shape[0], self.hidden_nodes))
        H = self.hidden_layer(x_train, self.input_weights, bias)
        self.output_weights = np.dot(np.linalg.pinv(H), y_train)

        pred = self.predict(x_train)
        acc = self.performance(y_train, pred)
        return pred, acc

    def predict(self, x):
        Win = np.random.normal(size=(x.shape[1], self.hidden_nodes))
        bias = np.random.normal(size=(x.shape[0], self.hidden_nodes))
        H = self.hidden_layer(x, self.input_weights, bias)
        predictions = np.dot(H, self.output_weights)
        return predictions

    def evaluate(self, x_test, y_test):
        pred = self.predict(x_test)
        acc = self.performance(y_test, pred)
        return pred, acc

    def performance(self, y_actual, y_predicted):
        y_actual = np.argmax(y_actual, axis=-1)
        y_predicted = np.argmax(y_predicted, axis=-1)

        correct = np.sum(y_actual == y_predicted)
        accuracy = (correct / y_actual.shape[0]) * 100
        accuracy = format(accuracy, '.2f')
        return accuracy


class IELM(ELM):
    def __init__(self, features, hidden_nodes, activation):
        ELM.__init__(self, features, hidden_nodes, activation)
        self.features = features
        self.l_max = hidden_nodes
        self.activation = activation

        self.input_weights = None
        self.output_weights = None  # beta
        self.bias = None

    def fit(self, x_train, y_train):
        l = 0
        E = y_train
        bias = np.random.normal(size=(self.l_max, 1))

        while l < self.l_max:
            Win = np.random.normal(size=(x_train.shape[1], 1))
            H = self.hidden_layer(x_train, Win, bias[l])
            beta = np.dot(np.linalg.pinv(H), E)
            pred = np.dot(H, beta)

            E -= pred

            self.input_weights = np.concatenate((self.input_weights, Win), axis=1) if l > 0 else Win
            self.output_weights = np.concatenate((self.output_weights, beta), axis=0) if l > 0 else beta
            self.bias = np.concatenate((self.bias, bias[0]), axis=0) if l > 0 else bias[l]

            l += 1

        pred = self.predict(x_train)
        acc = self.performance(y_train, pred)
        return pred, acc
    
class OSELM:
    def __init__(self, hidden_nodes, activation):
        self.features = None
        self.hidden_nodes = hidden_nodes
        self.activation = activation
 
        self.input_weights = None
        self.feedback = None # p
        self.output_weights = None  # beta
        self.bias = None
 
 
    def hidden_layer(self, x):
        bias = np.array([self.bias,]* x.shape[0])
        
        h = np.dot(x, self.input_weights) + bias
        
        if self.activation == 'relu':
            h = np.maximum(h, 0, h)
        elif self.activation == 'sigmoid':
            h = 1 / (1 + np.exp(-h))
        elif self.activation == 'sine':
            h = np.sin(h)
        elif self.activation == 'swish':
            sigmoid = 1/(1 + np.exp(-h)) 
            h = h * sigmoid
        elif self.activation == 'leakyrelu':
            h = np.where(h > 0, h, h * 0.01)
        elif self.activation == 'tanh':
            h = np.tanh(h)
        return h
 
    def fit(self, x_train, y_train):
        samples = x_train.shape[0]
        self.features = x_train.shape[1]
        
        if self.input_weights is None:
            self.input_weights = np.random.normal(size=(self.features, self.hidden_nodes))
            self.bias = np.random.random((1, self.hidden_nodes))[0]
        
        h = self.hidden_layer(x_train)
        
        if self.output_weights is None:
            self.feedback = np.linalg.pinv(np.dot(h.T, h))
            self.output_weights = self.feedback.dot(h.T).dot(y_train)
        else:
            part1 = self.feedback.dot(h.T)
            part2 = np.identity(samples) + h.dot(self.feedback).dot(h.T)
            part2 = np.linalg.pinv(part2)
            part3 = h.dot(self.feedback)
            self.feedback -= part1.dot(part2.dot(part3))
            
            
            p1 = self.feedback.dot(h.T)
            p2 = y_train - h.dot(self.output_weights)
            self.output_weights += p1.dot(p2)
        
    def predict(self, x):
        H = self.hidden_layer(x)
        predictions = np.dot(H, self.output_weights)
        return predictions
 
    def evaluate(self, x_test, y_test):
        pred = self.predict(x_test)
        acc = self.performance(y_test, pred)
        return pred, acc
 
    def performance(self, y_actual, y_predicted):
       
        y_actual = np.argmax(y_actual, axis=-1)
        y_predicted = np.argmax(y_predicted, axis=-1)
        
        correct = np.sum(y_actual == y_predicted)
        accuracy = (correct / y_actual.shape[0]) * 100
#         accuracy = format(accuracy, '.2f')
        return accuracy
