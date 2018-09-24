from pyspark import SparkContext
import matplotlib.pyplot as plt
import numpy as np

val sc = new SparkContext("local[2]", "Stumble Upon")
val raw_data = sc.textFile("./data/machine-learning-databases/adult.data")
val data = raw_data.map(line => line.split(", ")).filter(fields => fields.length == 15)
data.cache()

    val marriage_data = data.map { fields => (fields(5), 1) }.reduceByKey((x, y) => x + y)
    val education_data = data.map { fields => (fields(3), 1)}.reduceByKey((x, y) => x + y)
    val doc_marriage_data = data.filter(fields => fields(3) == "Doctorate").map(fields =>
      (fields(5), 1)).reduceByKey((x, y) => x + y)