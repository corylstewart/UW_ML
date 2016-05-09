import graphlab as gl
import numpy as np
import math

sales = gl.SFrame('kc_house_data.gl')
sales['floors'] = sales['floors'].astype(int)

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

def normalize_features(feature_matrix):
    f_m = np.array(feature_matrix)
    norms = np.linalg.norm(f_m, axis=0)
    return f_m/norms, norms

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
simple_feature_matrix, norms = normalize_features(simple_feature_matrix)
weights = np.array([1., 4., 1.])
prediction = predict_output(simple_feature_matrix, weights)

def get_ro(feature_matrix, output, prediction, weights, i):
    error = output - prediction
    return (feature_matrix[:,i] * (error + (weights[i] * feature_matrix[:,i]))).sum()

print 'Question 1 and 2'
print 'ro_1', '%.4g' % (get_ro(simple_feature_matrix, output, prediction, weights, 1)*2)
print 'ro_2', '%.4g' % (get_ro(simple_feature_matrix, output, prediction, weights, 2)*2)

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = get_ro(feature_matrix, output, prediction, weights, i)

    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i 
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2.
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2.
    else:
        new_weight_i = 0.
    
    return new_weight_i

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    while True:
        worst = float('-inf')
        for i in range(len(initial_weights)):
            old_weights_i = initial_weights[i]
            initial_weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, initial_weights, l1_penalty)
            this_diff = abs(old_weights_i-initial_weights[i])
            if this_diff > worst:
                worst = this_diff
        if worst <= tolerance:
            break
    return initial_weights

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix) # normalize features

weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)

predictions = predict_output(normalized_simple_feature_matrix, weights)
rss = ((output-predictions)*(output-predictions)).sum()
print 'Qeestion 3: ', rss

def find_zero_weights(features, weights):
    for i in range(len(weights)):
        if weights[i] == 0:
            if i == 0:
                print 'Constant'
            else:
                print features[i-1]

def find_non_zero_weights(features, weights):
    for i in range(len(weights)):
        if weights[i] != 0:
            if i == 0:
                print 'Constant'
            else:
                print features[i-1]

print 'Question 4: '
find_zero_weights(simple_features, weights)

train_data,test_data = sales.random_split(.8,seed=0)

all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']

(all_feature_matrix, all_feature_output) = get_numpy_data(train_data, all_features, 'price')
(normalized_all_feature_matrix, all_feature_norms) = normalize_features(all_feature_matrix)

initial_weights = np.zeros(len(all_features)+1)
l1_penalty = 1e7
tolerance = 1.

weights1e7 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, all_feature_output, 
                                               initial_weights, l1_penalty, tolerance)
print 'Question 5: '
find_non_zero_weights(all_features, weights1e7)

initial_weights = np.zeros(len(all_features)+1)
l1_penalty = 1e8

weights1e8 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, all_feature_output, 
                                               initial_weights, l1_penalty, tolerance)

print 'Question 6: '
find_non_zero_weights(all_features, weights1e8)

initial_weights = np.zeros(len(all_features)+1)
l1_penalty = 1e4
tolerance = 5e5

weights1e4 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, all_feature_output, 
                                               initial_weights, l1_penalty, tolerance)

print 'Question 7: '
find_non_zero_weights(all_features, weights1e4)