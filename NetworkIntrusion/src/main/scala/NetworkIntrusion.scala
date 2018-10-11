import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering._
import org.apache.spark.rdd._

object NetworkIntrusion {

  def distance(a: Vector, b: Vector) =
    math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d => d * d).sum)

  def distToCentroid(datum: Vector, model: KMeansModel) = {
    val cluster = model.predict(datum)
    val centroid = model.clusterCenters(cluster)
    distance(centroid, datum)
  }

  def clusteringScore(data: RDD[Vector], k: Int) = {
    val kmeans = new KMeans()
    kmeans.setK(k)
    val model = kmeans.run(data)
    data.map(datum => distToCentroid(datum, model)).mean()
  }

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local[*]", "Network")
    val rawData = sc.textFile("F:\\Downloads\\kddcup.data_10_percent\\kddcup.data")
    rawData.map(_.split(",").last).countByValue().toSeq.sortBy(_._2).reverse.foreach(println)

    val labelsAndData = rawData.map { line =>
      val buffer = line.split(",").toBuffer
      buffer.remove(1, 3)
      val label = buffer.remove(buffer.length-1)
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label, vector)
    }
    val data = labelsAndData.values.cache

    (5 to 40 by 5).map(k => (k, clusteringScore(data, k))).foreach(println)

    val dataAsArray = data.map(_.toArray)
    val numCols = dataAsArray.first.length
    val n = dataAsArray.count
    val sums = dataAsArray.reduce(
      (a, b) => a.zip(b).map(t => t._1 + t._2 * t._2))
    val sumSquares = dataAsArray.fold(
      new Array[Double](numCols)
      )((a, b) => a.zip(b).map(t => t._1 + t._2 * t._2))
    val stdevs = sumSquares.zip(sums).map {
      case(sumSq, sum) => math.sqrt(n * sumSq - sum * sum) / n
    }
    val means = sums.map(_ / n)


    def normalize(datum: Vector) = {
      val normalizedArray = (datum.toArray, means, stdevs).zipped.map(
        (value, mean, stdev) =>
          if (stdev <= 0) value - mean
          else (value - mean) / stdev
      )
      Vectors.dense(normalizedArray)
    }

    val normalizedData = data.map(normalize).cache


    val kmeans = new KMeans()
    kmeans.setK(120)
    val model = kmeans.run(normalizedData)
    model.clusterCenters.foreach(println)
    val clusterLabelCount = labelsAndData.map { case (label, datum) =>
      val cluster = model.predict(datum)
      (cluster, label)
    }.countByValue()

    clusterLabelCount.toSeq.sorted.foreach {
      case ((cluster, label), count) =>
        println(f"$cluster%1s$label%18s$count%8s")
    }
  }
}
