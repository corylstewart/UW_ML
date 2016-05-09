import graphlab as gl
import matplotlib.pyplot as plt
import numpy as np

def polynomial_sframe(feature, degree):
    poly_sframe = gl.SFrame()
    poly_sframe['power_1'] = feature
    for i in range(2, degree+1):
        poly_sframe['power_' + str(i)] = feature**i
    return poly_sframe

def rss(prediction, actual):
    diff = prediction - actual
    return (diff*diff).sum()

sales = gl.SFrame('kc_house_data.gl/')
sales = sales.sort(['sqft_living','price'])
l2_small_penalty = 1e-5

train_data = polynomial_sframe(sales['sqft_living'], 15)
train_data['price'] = sales['price']

model = gl.linear_regression.create(train_data, target='price', 
           l2_penalty=l2_small_penalty, l1_penalty=0, 
           validation_set=None, verbose=False)

print 'Question 1: ', model.coefficients[1]['value']


(semi_split1, semi_split2) = train_data.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)

print 'Small l2'
model1 = gl.linear_regression.create(set_1, target='price', 
           l2_penalty=l2_small_penalty, l1_penalty=0, 
           validation_set=None, verbose=False)

model2 = gl.linear_regression.create(set_2, target='price', 
           l2_penalty=l2_small_penalty, l1_penalty=0, 
           validation_set=None, verbose=False)

model3 = gl.linear_regression.create(set_3, target='price', 
           l2_penalty=l2_small_penalty, l1_penalty=0, 
           validation_set=None, verbose=False)

model4 = gl.linear_regression.create(set_4, target='price', 
           l2_penalty=l2_small_penalty, l1_penalty=0, 
           validation_set=None, verbose=False)

print 'Model 1: ', model1.coefficients[1]['value']
print 'Model 2: ', model2.coefficients[1]['value']
print 'Model 3: ', model3.coefficients[1]['value']
print 'Model 4: ', model4.coefficients[1]['value']


print 'Large l2'
l2_small_penalty = 1e5
model1 = gl.linear_regression.create(set_1, target='price', 
           l2_penalty=l2_small_penalty, l1_penalty=0, 
           validation_set=None, verbose=False)

model2 = gl.linear_regression.create(set_2, target='price', 
           l2_penalty=l2_small_penalty, l1_penalty=0, 
           validation_set=None, verbose=False)

model3 = gl.linear_regression.create(set_3, target='price', 
           l2_penalty=l2_small_penalty, l1_penalty=0, 
           validation_set=None, verbose=False)

model4 = gl.linear_regression.create(set_4, target='price', 
           l2_penalty=l2_small_penalty, l1_penalty=0, 
           validation_set=None, verbose=False)

print 'Model 1: ', model1.coefficients[1]['value']
print 'Model 2: ', model2.coefficients[1]['value']
print 'Model 3: ', model3.coefficients[1]['value']
print 'Model 4: ', model4.coefficients[1]['value']

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = gl.toolkits.cross_validation.shuffle(train_valid, random_seed=1)


def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    n = len(data)
    total_error = 0
    for i in range(k):
        start, end = (n*i)/k, (n*(i+1))/k-1
        train = data[0:start]
        train = train.append(data[end+1:])
        validation = data[start:end+1]
        model = gl.linear_regression.create(train, target='price', 
                        l2_penalty=l2_penalty, l1_penalty=0, 
                        validation_set=None, verbose=False)
        prediction = model.predict(validation)
        r = rss(prediction, validation[output_name])
        total_error += r
    return total_error


k = 10
l2_penaltys = np.logspace(1, 7, num=13)
output_name = 'price'
data = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
features_list = data.column_names()
data['price'] = train_valid_shuffled['price']
best = None
best_total = float('inf')
for l2 in l2_penaltys:
    this_one = k_fold_cross_validation(k, l2, data, output_name, features_list)
    if this_one < best_total:
        best_total = this_one
        best = l2

print best, best_total

t = polynomial_sframe(test['sqft_living'], 15)
t['price'] = test['price']
p = model.predict(t)
print rss(p, t['price'])