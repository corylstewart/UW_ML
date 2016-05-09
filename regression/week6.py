import graphlab as gl
import numpy as np
import matplotlib.pyplot as plt

sales = gl.SFrame('kc_house_data_small.gl')

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

def normalize_features(feature_matrix):
    f_m = np.array(feature_matrix)
    norms = np.linalg.norm(f_m, axis=0)
    return f_m/norms, norms

(train_and_validation, test) = sales.random_split(.8, seed=1)
(train, validation) = train_and_validation.random_split(.8, seed=1)

feature_list = ['bedrooms',  
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
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']
features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')

features_train, norms = normalize_features(features_train)
features_test = features_test / norms
features_valid = features_valid / norms

def euclidean_distance(a, b):
    return np.sqrt(np.power(a - b,2).sum())

dist = euclidean_distance(features_test[0], features_train[9])
print 'Question 1: ', '%.3f' % dist

def euclidean_distance_vectorized(neighborhood, house):
    return np.sqrt(np.sum(np.power(neighborhood-house,2), axis=1))

def find_k_nearest_neighbor_vectorized(neighborhood, input_features, k):
    return euclidean_distance_vectorized(neighborhood, 
                                         input_features).argsort()[:k]

def make_average_prediction(houses, output_train):
    return sum([output_train[x] for x in houses])/float(len(houses))

def get_rss(predictions, actuals):
    error = predictions-actuals
    return sum(error*error)

q2 = find_k_nearest_neighbor_vectorized(features_train[:10], features_test[0], 1)
print 'Question 2: ', q2
q3 = find_k_nearest_neighbor_vectorized(features_train, features_test[2], 1)
print 'Question 3: ', q3
print 'Question 4: ', output_train[q3][0]
q5 = find_k_nearest_neighbor_vectorized(features_train, features_test[2], 4)
print 'Question 5: ', q5
print 'Question 6: ', make_average_prediction(q5, output_train)

ten = []
for i in range(10):
    houses = find_k_nearest_neighbor_vectorized(features_train, features_test[i], 10)
    ten.append(make_average_prediction(houses, output_train))
print 'Question 7: ', min(ten)

def find_rss_for_each_k(features_train, output_train, 
                        features_test, output_test, high):
    k_rss = []
    k_s = range(1, high+1)
    houses = []
    for i in range(len(features_test)):
        houses.append(find_k_nearest_neighbor_vectorized(features_train, features_test[i], high))
    for i in range(1, high+1):
        predictions = []
        for j in range(len(houses)):
            predictions.append(make_average_prediction(houses[j][:i], output_train))
        this_rss = get_rss(predictions, output_test)
        k_rss.append(this_rss)
    return k_rss, k_s

index = len(output_test)
k_rss, k_s = find_rss_for_each_k(features_train, output_train, features_test[:index], output_test[:index], 15)
plt.plot(k_s, k_rss, 'bo-')
print 'Question 8: %.4g' % min(k_rss)