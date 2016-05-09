import graphlab as gl
import numpy as np
import matplotlib.pyplot as plt

sales = gl.SFrame('kc_house_data.gl/')

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

def get_residual_sum_of_squares(feature_matrix, output, weights):
    return get_rss(predict_output(feature_matrix, weights), output)

def get_rss(prediction, output):
    error = prediction - output
    return (error*error).sum()

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    derivative = 2 * np.dot(errors, feature)
    if not feature_is_constant:
        derivative += 2 * l2_penalty * weight
    return derivative

def ridge_regression_gradient_descent(feature_matrix, output, 
                      initial_weights, step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights)
    count = 0
    len_w = len(weights)
    while count < max_iterations:
        errors = predict_output(feature_matrix, weights) - output
        for i in xrange(len_w):
            weights[i] -= step_size * feature_derivative_ridge(errors, 
                                feature_matrix[:,i], weights[i], l2_penalty, i==0)
        count += 1
    return weights

features = ['sqft_living']
output = 'price'


train_data,test_data = sales.random_split(.8,seed=0)
simple_feature_matrix, train_output = get_numpy_data(train_data, features, output)
simple_test_feature_matrix, test_output = get_numpy_data(test_data, features, output)
initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000

simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, 
                           train_output, initial_weights, step_size, 0.0, max_iterations)

simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, 
                           train_output, initial_weights, step_size, 1e11, max_iterations)
'''
plt.plot(simple_feature_matrix,output,'k.',
         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')
'''

zeros = get_residual_sum_of_squares(simple_test_feature_matrix, test_output, initial_weights)
low_weight = get_residual_sum_of_squares(simple_test_feature_matrix, test_output, simple_weights_0_penalty)
high_weight = get_residual_sum_of_squares(simple_test_feature_matrix, test_output, simple_weights_high_penalty)

print 'Question 1: No regulatization ', round(simple_weights_0_penalty[1],1)
print 'Question 2: High regulatization ', round(simple_weights_high_penalty[1],1)
print 'Question 4: ', low_weight

complex_features = ['sqft_living', 'sqft_living15']

complex_feature_matrix, train_output = get_numpy_data(train_data, complex_features, output)
complex_test_feature_matrix, complex_test_output = get_numpy_data(test_data, complex_features, output)
complex_initial_weights = np.array([0., 0., 0.])
step_size = 1e-12
max_iterations=1000

complex_weights_0_penalty = ridge_regression_gradient_descent(complex_feature_matrix, 
                           train_output, complex_initial_weights, step_size, 0.0, max_iterations)
complex_weights_high_penalty = ridge_regression_gradient_descent(complex_feature_matrix, 
                           train_output, complex_initial_weights, step_size, 1e11, max_iterations)

print 'Question 5: ', round(complex_weights_0_penalty[1], 1)
print 'Question 6: ', round(complex_weights_high_penalty[1], 1)

complex_zeros = get_residual_sum_of_squares(complex_test_feature_matrix, complex_test_output, complex_initial_weights)
print 'Question 7: ', complex_zeros
no_r = predict_output(complex_test_feature_matrix, complex_weights_0_penalty)
high_r = predict_output(complex_test_feature_matrix, complex_weights_high_penalty)
print 'Question 8: ', abs(no_r[0]-complex_test_output[0]), abs(high_r[0]-complex_test_output[0])