import graphlab as gl
import matplotlib.pyplot as plt


def polynomial_sframe(feature, degree):
    poly_sframe = gl.SFrame()
    poly_sframe['power_1'] = feature
    for i in range(2, degree+1):
        poly_sframe['power_' + str(i)] = feature**i
    return poly_sframe

def rms(expected, predicted):
    diff = expected-predicted
    z = diff*diff
    return z.sum()

sales = gl.SFrame('kc_house_data.gl/')
sales = sales.sort(['sqft_living', 'price'])
poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']

t_v, testing = sales.random_split(.9, seed=1)
training, validation = t_v.random_split(.5, seed=1)

train_data = polynomial_sframe(training['sqft_living'], 15)
train_data['price'] = training['price']

validation_data = polynomial_sframe(validation['sqft_living'], 15)
validation_data['price'] = validation['price']

test_data = polynomial_sframe(testing['sqft_living'], 15)
test_data['price'] = testing['price']

prefix = 'power_'
best = None
tot = float('inf')
for i in range(1,16):
    feature = [prefix+str(i)]
    model = gl.linear_regression.create(train_data, target='price', 
                    features=feature, validation_set=None)
    prediction = model.predict(validation_data[feature])
    rss = pow((validation_data['price']-prediction),2).sum()
    if rss < tot:
        tot = rss
        best = feature
        best_model = model
print best, tot
prediction = model.predict(test_data[best])
rss = pow((test_data['price']-prediction),2).sum()
print rss


'''
model1 = gl.linear_regression.create(poly1_data, target='price', features=['power_1'], validation_set=None)
model1.get('coefficients')

plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')


poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = gl.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)
model2.get("coefficients")
plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model15 = gl.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None)
model15.get("coefficients")
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
        poly15_data['power_1'], model15.predict(poly15_data),'-')
'''

'''
a, b = poly1_data.random_split(.5, seed=0)
set_1, set_2 = a.random_split(.5, seed=0)
set_3, set_4 = b.random_split(.5, seed=0)

set_1_data = polynomial_sframe(set_1['power_1'], 15)
set_2_data = polynomial_sframe(set_2['power_1'], 15)
set_3_data = polynomial_sframe(set_3['power_1'], 15)
set_4_data = polynomial_sframe(set_4['power_1'], 15)

my_features = set_1_data.column_names()

set_1_data['price'] = set_1['price']
set_2_data['price'] = set_2['price']
set_3_data['price'] = set_3['price']
set_4_data['price'] = set_4['price']

model_set_1 = gl.linear_regression.create(set_1_data, target = 'price', features = my_features, validation_set = None)
model_set_2 = gl.linear_regression.create(set_2_data, target = 'price', features = my_features, validation_set = None)
model_set_3 = gl.linear_regression.create(set_3_data, target = 'price', features = my_features, validation_set = None)
model_set_4 = gl.linear_regression.create(set_4_data, target = 'price', features = my_features, validation_set = None)

print 'Model 1: ', model_set_1.coefficients[-1]['name'], model_set_1.coefficients[-1]['value']
print 'Model 2: ', model_set_2.coefficients[-1]['name'], model_set_2.coefficients[-1]['value']
print 'Model 3: ', model_set_3.coefficients[-1]['name'], model_set_3.coefficients[-1]['value']
print 'Model 4: ', model_set_4.coefficients[-1]['name'], model_set_4.coefficients[-1]['value']


plt.plot(set_1_data['power_1'], model_set_1.predict(set_1_data),'-')

plt.plot(set_2_data['power_1'], model_set_2.predict(set_2_data),'o')

plt.plot(set_3_data['power_1'], model_set_3.predict(set_3_data),'*')

plt.plot(set_4_data['power_1'], model_set_4.predict(set_4_data),'.')
'''

print 'done'