package AdultBase.database

import AdultBase.database.AdultBaseFileOperation._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession

object AdultBaseLinearSupportMachineVector {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getRootLogger.setLevel(Level.WARN)

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("AdultBaseDatabase")
      .getOrCreate()

    val indexed_training = spark.read.format("csv")
      .option("sep", ",")
      .option("header", "true")
      .schema(getAdultIndexSchema)
      .load("./data/indexed_adult_training")

    val indexed_test = spark.read.format("csv")
      .option("sep", ",")
      .option("header", "true")
      .schema(getAdultIndexSchema)
      .load("./data/indexed_adult_test")

    val indexed_info_elements = Array("workclassIndex", "educationIndex", "maritial_statusIndex", "occupationIndex",
      "relationshipIndex", "raceIndex")

    val assembler = new VectorAssembler()
      .setInputCols(indexed_info_elements)
      .setOutputCol("features")

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(10)

    val index_pipeline = new Pipeline().setStages(Array(assembler, featureIndexer))

    val training_table = index_pipeline.fit(indexed_training).transform(indexed_training)
    training_table.show(truncate = false)

    val test_table = index_pipeline.fit(indexed_test).transform(indexed_test)
    test_table.show(truncate = false)


    val lsvc = new LinearSVC()
      .setMaxIter(500)
      .setRegParam(0.1)
      .setLabelCol("sexIndex")
      .setFeaturesCol("features")

    val lsvcModel = lsvc.fit(training_table)
    val predictions = lsvcModel.transform(test_table)

    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val accuracy = evaluator
      .setLabelCol("sexIndex")
      .evaluate(predictions)

    println(accuracy)

  }
}
