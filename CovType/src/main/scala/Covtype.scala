import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

object Covtype {
  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }

  def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
    val countsByCategory = data.map(_.label).countByValue()
    val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
    counts.map(_.toDouble / counts.sum)
  }

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local[*]", "Covtype")
    val rawData = sc.textFile("F:\\Downloads\\machine-learning-databases\\covtype.data\\covtype.data")

    val dataOneHot = rawData.map { line =>
      val values = line.split(",").map(_.toDouble)
      val featureVector = Vectors.dense(values.init)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }

    val dataValue = rawData.map { line =>
      val values = line.split(",").map(_.toDouble)
      val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
      val soil = values.slice(14, 54).indexOf(1.0).toDouble
      val featureVector = Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }


    for (data <- Array(dataOneHot, dataValue))
      yield {
        val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
        trainData.cache
        cvData.cache
        testData.cache
        val model1 = DecisionTree.
          trainClassifier(trainData,7, Map[Int, Int](), "gini", 4, 100)
        val model2 = DecisionTree.
          trainClassifier(trainData, 7, Map[Int, Int](), "entropy", 20, 300)
        val metrics1 = getMetrics(model1, cvData)
        val metrics2 = getMetrics(model2, cvData)
        println("Model1: " + metrics1.precision)
        println("Model2: " + metrics2.precision)

        val trainPriorProbabilities = classProbabilities(trainData)
        val cvPriorProbabilities = classProbabilities(cvData)
        println(trainPriorProbabilities.zip(cvPriorProbabilities).map {
          case (trainProb, cvProb) => trainProb * cvProb }.sum)
      }
  }
}