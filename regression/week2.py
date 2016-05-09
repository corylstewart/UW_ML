import graphlab as gl
import numpy as np
from math import sqrt

def get_numpy_data(frame_data, features, output):
    frame_data['constant'] = 1
    features = ['constant'] + features
    feature_sframe = gl.SFrame()
    for feature in features:
        feature_sframe[feature] = frame_data[feature]
    feature_matrix = feature_sframe.to_numpy()
    output_sarray = gl.SFrame()
    output_sarray[output] = frame_data[output]
    output_array = output_sarray.to_numpy()
    return (feature_matrix, output_array.flatten())

def predict_output(feature_matrix, weights):
    return np.dot(feature_matrix, weights)

def feature_derivative(errors, feature):
    derivative = np.dot(errors, feature) * 2
    return(derivative)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    len_weights = len(weights)
    while not converged:
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output
        gradient_sum_squares = 0
        for i in range(len_weights):
            derivative = feature_derivative(errors, feature_matrix[:, i])
            gradient_sum_squares += derivative*derivative
            weights[i] -= derivative*step_size
        if sqrt(gradient_sum_squares) < tolerance:
            converged = True
    return weights

filename = 'kc_house_data.gl/'
sales = gl.SFrame(filename)

train_data,test_data = sales.random_split(.8,seed=0)
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7
weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)
print 'Question 1: ', weights[1]

(simple_feature_matrix_test, output_test) = get_numpy_data(test_data, simple_features, my_output)
predictions = predict_output(simple_feature_matrix_test, weights)
print 'Question 2: ', predictions[0]

simple_rss = sum((predictions-output_test)**2)
diff1 = abs(predictions[0]-output_test[0])

model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9
weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)

(feature_matrix_test, output_test) = get_numpy_data(test_data, model_features, my_output)
predictions = predict_output(feature_matrix_test, weights)
print 'Question 3: ', predictions[0]
mulitple_rss = sum((predictions-output_test)**2)
diff2 = abs(predictions[0]-output_test[0])

print 'Question 4: ', min(((diff1, 'Simple Model'),(diff2, 'Multiple Model')))
print 'Question 5: ', min(((simple_rss, 'Simple RSS'), (mulitple_rss, 'Multiple RSS')))