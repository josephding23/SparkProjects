from pyspark.context import SparkContext
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
import matplotlib.pyplot
from matplotlib.pylab import hist

sc = SparkContext(appName="Bike Sharing")
path = "../data/hour_noheader.csv"
raw_data = sc.textFile(path)
num_data = raw_data.count()
records = raw_data.map(lambda x: x.split(","))
first = records.first()
print(first)
print(num_data)
records.cache()


def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()


print("Mapping of first categorical feasture column: %s" % get_mapping(records, 2))
mappings = [get_mapping(records, i) for i in range(2, 10)]
cat_len = sum(map(len, mappings))
num_len = len(records.first()[11:15])
total_len = num_len + cat_len

print("Feature vector length for categorical features: %d" % cat_len)
print("Feature vector length for numerical features: %d" % num_len)
print("Total feature vector length: %d" % total_len)


def extra_features(record):
    cat_vec = np.zeros(cat_len)
    i = 0
    step = 0
    for field in record[2:9]:
        m = mappings[i]
        idx = m[field]
        cat_vec[idx + step] = 1
        i = i + 1
        step = step + len(m)
    num_vec = np.array([float(field) for field in record[10:14]])
    return np.concatenate((cat_vec, num_vec))


def extract_label(record):
    return float(record[-1])


data = records.map(lambda r: LabeledPoint(extract_label(r), extra_features(r)))
first_point = data.first()
print("Raw data: " + str(first[2:]))
print("Label: " + str(first_point.label))
print("Linear Model feature vector: \n" + str(first_point.features))
print("Linear Model feature vector length: " + str(len(first_point.features)))


def extract_features_dt(record):
    return np.array([float(field) for field in record[2:14]])


data_dt = records.map(lambda r: LabeledPoint(extract_label(r), extract_features_dt(r)))
first_point_dt = data_dt.first()
print("Decision Tree feature vector: " + str(first_point_dt.features))
print("Decision Tree feature vector length: " + str(len(first_point_dt.features)))

linear_model = LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept=False)
true_vs_predicted = data.map(lambda p: (p.label, linear_model.predict(p.features)))
print("Linear Model predictions: " + str(true_vs_predicted.take(5)))

dt_model = DecisionTree.trainRegressor(data_dt, {})
preds = dt_model.predict(data_dt.map(lambda p: p.features))
actual = data.map(lambda p: p.label)
true_vs_predicted_dt = actual.zip(preds)
print("Decision Tree predictions" + str(true_vs_predicted_dt.take(5)))
print("Decision Tree depth: " + str(dt_model.depth()))
print("Decision Tree number of nodes: " + str(dt_model.numNodes()))


def squared_error(actual, pred):
    return (pred - actual)**2


def abs_error(actual, pred):
    return np.abs(pred - actual)


def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1))**2


mse = true_vs_predicted.map(lambda tp: squared_error(tp[0], tp[1])).mean()
mae = true_vs_predicted.map(lambda tp: abs_error(tp[0], tp[1])).mean()
rmsle = np.sqrt(true_vs_predicted.map(lambda tp: squared_log_error(tp[0], tp[1])).mean())
print("Linear Model - Mean Squared Error: %2.4f" % mse)
print("Linear Model - Mean Absolute Error: %2.4f" % mae)
print("Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle)

mse_dt = true_vs_predicted_dt.map(lambda tp: squared_error(tp[0], tp[1])).mean()
mae_dt = true_vs_predicted_dt.map(lambda tp: abs_error(tp[0], tp[1])).mean()
rmsle_dt = np.sqrt(true_vs_predicted_dt.map(lambda tp: squared_log_error(tp[0], tp[1])).mean())
print("Decision Tree - Mean Squared Error: %2.4f" % mse_dt)
print("Decision Tree - Mean Absolute Error: %2.4f" % mae_dt)
print("Decision Tree - Root Mean Squared Log Error: %2.4f" % rmsle_dt)

targets = records.map(lambda r: float(r[-1])).collect()
hist(targets, bins=40, color="lightblue", density=True)
fig1 = matplotlib.pyplot.gcf()
fig1.set_size_inches(16, 10)
#fig1.savefig("../resources/hist1.jpg")
fig1.show()

log_targets = records.map(lambda r: np.log(float(r[-1]))).collect()
hist(log_targets, bins=40, color="lightblue", density=True)
fig2 = matplotlib.pyplot.gcf()
fig2.set_size_inches(16, 10)
fig2.show()
#fig2.savefig("../resources/hist2.jpg")

sqrt_targets = records.map(lambda r: np.sqrt(float(r[-1]))).collect()
hist(sqrt_targets, bins=40, color="lightblue", density=True)
fig3 = matplotlib.pyplot.gcf()
fig3.show()
fig3.savefig("../resources/hist3.jpg")

data_log = data.map(lambda lp: LabeledPoint(np.log(lp.label), lp.features))
model_log = LinearRegressionWithSGD.train(data_log, iterations=10, step=0.1)
true_vs_predicted_log = data_log.map(lambda p: (np.exp(p.label), np.exp(model_log.predict(p.features))))
mse_log = true_vs_predicted_log.map(lambda tp: squared_error(tp[0], tp[1])).mean()
mae_log = true_vs_predicted_log.map(lambda tp: abs_error(tp[0], tp[1])).mean()
rmsle_log = np.sqrt(true_vs_predicted_log.map(lambda tp: squared_log_error(tp[0], tp[1])).mean())
print("Mean Squared Error: %2.4f" % mse_log)
print("Mean Absolute Error: %2.4f" % mae_log)
print("Root Mean Squared Log Error: %2.4f" % rmsle_log)
print("Non log-transformed predictions:\n" + str(true_vs_predicted.take(3)))
print("Log-transformed predictions:\n" + str(true_vs_predicted_log.take(3)))

data_dt_log = data_dt.map(lambda lp: LabeledPoint(np.log(lp.label), lp.features))
dt_model_log = DecisionTree.trainRegressor(data_dt_log, {})

preds_log = dt_model_log.predict(data_dt_log.map(lambda p: p.features))
actual_log = data_dt_log.map(lambda p: p.label)
true_vs_predicted_dt_log = actual_log.zip(preds_log).map(lambda tp: (np.exp(tp[0]), np.exp(tp[1])))

mse_log_dt = true_vs_predicted_dt_log.map(lambda tp: squared_error(tp[0], tp[1])).mean()
mae_log_dt = true_vs_predicted_dt_log.map(lambda tp: abs_error(tp[0], tp[1])).mean()
rmsle_log_dt = np.sqrt(true_vs_predicted_dt_log.map(lambda tp: squared_log_error(tp[0], tp[1])).mean())
print("Mean Squared Error: %2.4f" % mse_log_dt)
print("Mean Absolute Error: %2.4f" % mae_log_dt)
print("Root Mean Squared Log Error: %2.4f" % rmsle_log_dt)
print("Non log-transformed predictions:\n" + str(true_vs_predicted_dt.take(3)))
print("Log-transformed predictions:\n" + str(true_vs_predicted_dt_log.take(3)))


data_with_idx = data.zipWithIndex().map(lambda kv: (kv[1], kv[0]))
test = data_with_idx.sample(False, 0.2, 42)
train = data_with_idx.subtractByKey(test)

train_data = train.map(lambda idx_p: idx_p[1])
test_data = test.map(lambda idx_p: idx_p[1])
train_size = train_data.count()
test_size = test_data.count()
print("Training data size: %d" % train_size)
print("Test data size: %d" % test_size)
print("Total data size: %d" % num_data)

data_with_idx_dt = data_dt.zipWithIndex().map(lambda kv: (kv[1], kv[0]))
test_dt = data_with_idx_dt.sample(False, 0.2, 42)
train_dt = data_with_idx_dt.subtractByKey(test_dt)
train_data_dt = train_dt.map(lambda idx_p: idx_p[1])
test_data_dt = test_dt.map(lambda idx_p: idx_p[1])


def evaluate(train, test, iterations, step, regParam, regType, intercept):
    model = LinearRegressionWithSGD.train(train, iterations, step, regParam=regParam, regType=regType, intercept=intercept)
    _tp = test.map(lambda p: (p.label, model.predict(p.features)))
    _rmsle = np.sqrt(_tp.map(lambda tp: squared_log_error(tp[0], tp[1])).mean())
    return _rmsle


def evaluate_dt(train, test, maxDepth, maxBins):
    model = DecisionTree.trainRegressor(train, {}, impurity='variance', maxDepth=maxDepth, maxBins=maxBins)
    preds = model.predict(test.map(lambda p: p.features))
    actual = test.map(lambda p: p.label)
    _tp = actual.zip(preds)
    rmsle = np.sqrt(_tp.map(lambda tp: squared_log_error(tp[0], tp[1])).mean())
    return rmsle