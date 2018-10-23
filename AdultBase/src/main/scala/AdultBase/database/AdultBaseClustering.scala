package AdultBase.database

import AdultBase.database.AdultBaseFileOperation._
import org.apache.spark.ml.clustering.{KMeans, KMeansSummary}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.StructType
import org.apache.spark.SparkContext
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}

object AdultBaseClustering {

    def main(args: Array[String]): Unit = {
      Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
      Logger.getRootLogger.setLevel(Level.WARN)

      val spark = SparkSession
        .builder()
        .master("local[*]")
        .appName("AdultBaseDatabase")
        .getOrCreate()

      import spark.implicits._
      val data_table = spark.read.format("csv")
        .option("sep", ",")
        .option("header", "true")
        .schema(getAdultSchema)
        .load("./data/adult_training")

      val indexed_table = getIndexerPipline.fit(data_table).transform(data_table)

      val info_elements = Array("age", "workclassIndex", "educationIndex", "maritial_statusIndex",
        "occupationIndex", "relationshipIndex", "raceIndex", "sexIndex", "native_countryIndex")

      val assembler = new VectorAssembler()
          .setInputCols(info_elements)
          .setOutputCol("features")

      val defaultAttr = NumericAttribute.defaultAttr
      val attrs = Array("age", "workclassIndex", "educationIndex", "maritial_statusIndex",
        "occupationIndex", "relationshipIndex", "raceIndex", "sexIndex", "native_countryIndex")
        .map(defaultAttr.withName)
      val attrGroup = new AttributeGroup("features", attrs.asInstanceOf[Array[Attribute]])

      indexed_table.select("age", "workclassIndex", "educationIndex", "maritial_statusIndex",
        "occupationIndex", "relationshipIndex", "raceIndex", "sexIndex", "native_countryIndex").show(8)
      val indexed_assembled_table = assembler.transform(indexed_table)

      indexed_assembled_table.show(8)

      val kmeans = new KMeans()
        .setK(8)
        .setMaxIter(500)
        .setSeed(1L)
        .setFeaturesCol("features")

      val model = kmeans.fit(indexed_assembled_table)
      val predictions = model.transform(indexed_assembled_table)
      println(model.explainParams())

      val evaluator = new ClusteringEvaluator()
      val silhouette = evaluator.evaluate(predictions)

      println(silhouette)

      model.clusterCenters.foreach(println)

      val clusters = model.clusterCenters
        .map(fields => Row(Vectors.dense(fields.toArray.map(num => math.round(num).toDouble))))
      val clusters_dt = spark.sparkContext.parallelize(clusters)
      val cluster_table =
        spark.createDataFrame(clusters_dt, StructType(Array(attrGroup.toStructField())))
      cluster_table.show()

      val vecToArray = udf( (xs: Vector) => xs.toArray )
      val dfArr = cluster_table.withColumn("featuresArray" , vecToArray($"features") )
      dfArr.select("featuresArray").show(truncate = false)

      val sqlExpr = info_elements.zipWithIndex.map{ case (alias, idx) =>
        col("featuresArray").getItem(idx).as(alias) }

      val split_table = dfArr.select(sqlExpr : _*)
      split_table.show(truncate = false)

      val cluster_info_split_table = getConverterPipline
        .fit(split_table).transform(split_table).select("age", "workclass", "education", "maritial_status",
        "occupation", "relationship", "race", "sex")
      cluster_info_split_table.show(truncate = false)
    }
}
