import java.util.Arrays

import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.clustering._
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.classification.{LinearSVC, LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes}
import org.apache.spark.ml.regression._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext


object MainGuide {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val sc = new SparkContext("local[*]", "test")
    //import sqlContext.implicits._

    val observations = sc.parallelize(
      Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(3.0, 30.0, 300.0)
      )
    )

    // Compute column summary statistics.
    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    println(summary.mean)  // a dense vector containing the mean value for each column
    println(summary.variance)  // column-wise variance
    println(summary.numNonzeros)  // number of nonzeros in each column
    }
}

