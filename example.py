import utils
import _elm

(x_train, y_train), (x_test, y_test) = utils.get_data('mnist')
features = x_train[0]
hidden_nodes = 100
activation = None
elm = _elm.ELM(features, hidden_nodes, activation)

_, training_accuracy = elm.fit(x_train, y_train)
_, testing_accuracy = elm.evaluate(x_test, y_test)

print('ELM training accuracy : ', training_accuracy)
print('ELM testing accuracy : ', testing_accuracy)

ielm = _elm.IELM(features, hidden_nodes, activation)

_, ielm_training_accuracy = ielm.fit(x_train, y_train)
_, ielm_testing_accuracy = ielm.evaluate(x_test, y_test)

print('I-ELM training accuracy : ', ielm_training_accuracy)
print('I-ELM testing accuracy : ', ielm_testing_accuracy)