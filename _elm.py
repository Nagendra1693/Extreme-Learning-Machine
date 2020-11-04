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