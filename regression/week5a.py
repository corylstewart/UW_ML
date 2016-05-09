import graphlab as gl
from math import log, sqrt
import numpy as np

sales = gl.SFrame('kc_house_data.gl')

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors'] = sales['floors'].astype(float) 
sales['floors_square'] = sales['floors']*sales['floors']

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = gl.linear_regression.create(sales, target='price', features=all_features,
                                        validation_set=None, 
                                        l2_penalty=0., l1_penalty=1e10, verbose=False)

print 'Question 1'
print model_all.get('coefficients').print_rows(num_rows=18, num_columns=3)

(training_and_validation, testing) = sales.random_split(.9,seed=1)
(training, validation) = training_and_validation.random_split(0.5, seed=1)

def get_rss(model, input, output):
    true_value = input[output]
    prediction = model.predict(input)
    error = prediction - true_value
    return (error*error).sum()

best_rss = float('inf')
best_model = None
best_l1 = None

for l1 in np.logspace(1, 7, num=13):
    model = gl.linear_regression.create(training, target='price', features=all_features,
                                        validation_set=None, 
                                        l2_penalty=0., l1_penalty=l1, verbose=False)
    rss = get_rss(model, validation, 'price')
    if rss < best_rss:
        best_model = model
        best_l1 = l1
        best_rss = rss
print 'Question 2: ', best_l1
print 'Question 3: ', best_model.get('coefficients').print_rows(num_rows=18, num_columns=3)
#print 'Question 4: ', 
#print get_rss(best_model, testing, 'price')

max_nonzeros = 7
l1_penalty_values = np.logspace(8, 10, num=20)

def find_best_l1(training, validation, testing, l1_penalty_values, max_nonzeros):
    best_rss = float('inf')
    best_model = None
    non_z = []
    for l1 in l1_penalty_values:
        model = gl.linear_regression.create(training, target='price', features=all_features,
                                            validation_set=None,
                                            l2_penalty=0., l1_penalty=l1, verbose=False)
        rss = get_rss(model, validation, 'price')
        non_z.append(model['coefficients']['value'].nnz())
        if rss < best_rss:
            best_model = model
            best_rss = rss
    return best_model, non_z

best_model_7, non_z = find_best_l1(training, validation, testing, l1_penalty_values, max_nonzeros)

for i in range(len(non_z)):
    if non_z[i] <= max_nonzeros:
        print 'Largest L1 penaly with more than 7 non zero features: ', round(l1_penalty_values[i-1])
        print 'Smallest L1 penalty with 7 or less non zero features: ', round(l1_penalty_values[i])
        break

model = gl.linear_regression.create(training, target='price', features=all_features,
                                            validation_set=None,
                                            l2_penalty=0., l1_penalty=l1_penalty_values[i], verbose=False)
model.get('coefficients').print_rows(num_rows=18, num_columns=3)

print model.get('coefficients').print_rows(num_rows=18, num_columns=3)