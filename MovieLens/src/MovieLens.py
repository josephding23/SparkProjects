from pyspark import SparkContext
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as sp
from pyspark.mllib.feature import Normalizer

sc = SparkContext(master='local[2]', appName='MovieLens')


user_data = sc.textFile('../data/ml-100k/u.user')
print(user_data.first())
user_fields = user_data.map(lambda line: line.split('|'))
num_users = user_fields.map(lambda fields: fields[0]).count()
num_genders = user_fields.map(lambda fields: fields[2]).distinct().count()
num_occupations = user_fields.map(lambda fields: fields[3]).distinct().count()
num_zipcodes = user_fields.map(lambda fields: fields[4]).distinct().count()
print("Users: %d, genders: %d, occupations: %d, ZIP codes: %d" % (num_users, num_genders, num_occupations,num_zipcodes))

ages = user_fields.map(lambda _x: int(_x[1])).collect()
plt.hist(ages, bins=20, color='lightblue', density=True)
fig = plt.gcf()
fig.set_size_inches(16, 10)
fig.show()

count_by_occupation = user_fields.map(lambda fields: (fields[3], 1)).reduceByKey(lambda x, y: x+y).collect()
# add when occupation is the same
x_axis1 = np.array([c[0] for c in count_by_occupation])
y_axis1 = np.array([c[1] for c in count_by_occupation])

x_axis = x_axis1[np.argsort(y_axis1)]
y_axis = y_axis1[np.argsort(y_axis1)]

pos = np.arange(len(x_axis))   # (0, 1, 2, 3, 4.....)
width = 1.0

ax = plt.axes()
ax.set_xticks(pos + (width/2))     # set x-ticks' numbers
ax.set_xticklabels(x_axis)         # set x-ticks' labels
plt.bar(pos, y_axis, width, color='lightblue')
plt.xticks(rotation=30)
fig = plt.gcf()
fig.set_size_inches(16, 10)
fig.show()

count_by_occupation2 = user_fields.map(lambda fields: fields[3]).countByValue()
print('Map-reduce approach:')
print(dict(count_by_occupation))
print('')
print('countByValue approach:')
print(dict(count_by_occupation2))

movie_data = sc.textFile('../data/ml-100k/u.item')
# print(movie_data.first())
num_movies = movie_data.count()
print('Movies: %d' % num_movies)


def convert_year(_x):
    try:
        return int(_x[-4:])
    except:
        return 1900


movie_fields = movie_data.map(lambda lines: lines.split('|'))
years = movie_fields.map(lambda fields: fields[2]).map(lambda _x: convert_year(_x))
years_filtered = years.filter(lambda fields: fields != 1900)


movie_ages = years_filtered.map(lambda yr: 1998-yr).countByValue()
values = movie_ages.values()
bins = movie_ages.keys()
plt.hist(values, bins=bins, color='lightblue', density=True)
fig = plt.gcf()
fig.set_size_inches(16, 10)
fig.show()

rating_data_raw = sc.textFile('../data/ml-100k/u.data')
print(rating_data_raw.first())
num_ratings = rating_data_raw.count()
print('Ratings: %d' % num_ratings)

rating_data = rating_data_raw.map(lambda line: line.split('\t'))
ratings = rating_data.map(lambda fields: int(fields[2]))
max_rating = ratings.reduce(lambda _x, y: max(_x, y))
min_rating = ratings.reduce(lambda _x, y: min(_x, y))
mean_rating = ratings.reduce(lambda _x, y: _x + y) / num_ratings
median_rating = np.median(ratings.collect())
ratings_per_user = num_ratings / num_users
ratings_per_movie = num_movies / num_users
print('Min Rating: %d' % min_rating)
print('Max Rating: %d' % max_rating)
print('Average Rating: %2.2f' % mean_rating)
print('Median Rating: %d' % median_rating)
print('Average Ratings for user: %2.2f' % ratings_per_user)
print('Average Ratings for movie: %2.2f' % ratings_per_movie)
ratings.stats()

count_by_rating = ratings.countByValue()
x_axis = np.array(count_by_rating.keys())
y_axis = np.array([float(c) for c in count_by_rating.values()])
y_axis_normed = y_axis / y_axis.sum()
pos = np.arange(len(x_axis))
width = 1.0

ax = plt.axes()
ax.set_xticks(pos + (width/2))
ax.set_xticklabels(x_axis)
plt.bar(pos, y_axis_normed, width, color='lightblue')
plt.xticks(rotation=30)
fig = plt.gcf()
fig.set_size_inches(16, 10)
fig.show()

user_ratings_grouped = user_data.map(lambda fields: (int(fields[0]), int(fields[2]))).groupByKey()
user_ratings_byuser = user_ratings_grouped.map(lambda k_v: (k_v[0], len(k_v[1])))
user_ratings_byuser_local = user_ratings_byuser.map(lambda k_v: k_v[1]).collect()
plt.hist(user_ratings_byuser_local, bins=200, color='lightblue', density=True)
fig = plt.gcf()
fig.set_size_inches(16, 10)
fig.show()

years_pre_processed = movie_fields.map(lambda fields: fields[2]).map(lambda _x: convert_year(_x)).collect()
years_pre_processed_array = np.array(years_pre_processed)
mean_year = np.mean(years_pre_processed_array[years_pre_processed_array != 1900])
median_year = np.median(years_pre_processed_array[years_pre_processed_array != 1900])
index_bad_data = np.where(years_pre_processed_array[years_pre_processed_array != 1900])
years_pre_processed_array[index_bad_data] = median_year
print("Mean year of release: %d" % mean_year)
print("Median year of release: %d" % median_year)

all_occupations = user_fields.map(lambda fields: fields[3]).distinct().collect()
all_occupations.sort()
idx = 0
all_occupations_dict = {}
for o in all_occupations:
    all_occupations_dict[o] = idx
    idx += 1

print("Encoding of '%s': %d" % (occ.key(), occ.value()) for occ in all_occupations_dict)
K = len(all_occupations_dict)
binary_x = np.zeros(K)
k_programmer = all_occupations_dict['programmer']
binary_x[k_programmer] = 1
print('Binary feature vector: %s' % binary_x)


def extract_datetime(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts)


timestamps = rating_data.map(lambda fields: int(fields[3]))
hour_of_day = timestamps.map(lambda ts: extract_datetime(ts).hour)


def assign_tod(hr):
    times_of_day = {
        'morning' : range(7, 12),
        'lunch' : range(12, 14),
        'afternoon' : range(14, 18),
        'evening' : range(18, 23),
        'night' : range(23, 7)
    }
    for k, v in times_of_day.iteritems():
        if hr in v:
            return k


time_of_day = hour_of_day.map(lambda hr: assign_tod(hr))


def extract_title(raw):
    import re
    grps = re.search('\((w+)\)', raw)
    if grps:
        return raw[:grps.start()].strip()
    else:
        return raw


raw_titles = movie_fields.map(lambda fields: fields[1])
for raw_title in raw_titles.take(5):
    print(extract_title(raw_title))
movie_titles = raw_titles.map(lambda m: extract_title(m))
title_terms = movie_titles.map(lambda t: t.split(" "))
print(title_terms.take(5))

all_terms = title_terms.flatMap(lambda _x: _x).distinct().collect()
idx = 0
all_terms_dict = {}
for term in all_terms_dict:
    all_terms_dict[term] = idx
    idx += 1

print("Index of '%s': %d" % (term.key(), term.value()) for term in all_terms_dict)


def create_vector(terms, term_dict):
    num_terms = len(term_dict)
    _x = sp.csc_matrix((1, num_terms))
    for t in terms:
        if t in term_dict:
            _idx = term_dict[t]
            _x[0, _idx] = 1
    return _x

all_terms_bcast = sc.broadcast(all_terms_dict)
terms_vectors = title_terms.map(lambda terms: create_vector(terms, all_terms_bcast.value))

np.random.seed(42)
x = np.random.randn(10)
norm_x_2 = np.linalg.norm(x)
normalized_x = x / norm_x_2
print("x:\n%s" % x)
print("2-Norm of x: %2.4f" % norm_x_2)
print("Normalized x:\n%s" % normalized_x)
print("2-Norm of normalized_x: %2.4f" % np.linalg.norm(normalized_x))

normalizer = Normalizer()
vector = sc.parallelize([x])

normalized_x_mllib = normalizer.transform(vector).first().toArray()

print("x:\n%s" % x)
print("2-Norm of x: %2.4f" % norm_x_2)
print("Normalized x MLlib: \n%s" % normalized_x_mllib)
print("2-Norm of normalized_x_mllib: %2.4f" % np.linalg.norm(normalized_x_mllib))
