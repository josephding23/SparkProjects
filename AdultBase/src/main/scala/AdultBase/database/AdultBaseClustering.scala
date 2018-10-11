package AdultBase.database

import AdultBase.database.AdultBaseFileOperation._
import org.apache.spark.ml.clustering.KMeans
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
      val df = spark.read.format("csv")
        .option("sep", ",")
        .option("header", "true")
        .schema(getAdultIndexSchema)
        .load("./data/adult_training")

      val info_elements = Array("age", "workclassIndex", "educationIndex", "maritial_statusIndex",
        "occupationIndex", "relationshipIndex", "raceIndex", "sexIndex", "hours_per_week", "native_countryIndex")

      val assembler = new VectorAssembler()
          .setInputCols(info_elements)
          .setOutputCol("features")

      val defaultAttr = NumericAttribute.defaultAttr
      val attrs = Array("age", "workclassIndex", "educationIndex", "maritial_statusIndex",
        "occupationIndex", "relationshipIndex", "raceIndex", "sexIndex", "hours_per_week", "native_countryIndex")
        .map(defaultAttr.withName)
      val attrGroup = new AttributeGroup("features", attrs.asInstanceOf[Array[Attribute]])

      val indexed_table = getIndexerPipline.fit(df).transform(df)
        .select("age", "workclassIndex", "educationIndex", "maritial_statusIndex",
          "occupationIndex", "relationshipIndex", "raceIndex", "sexIndex", "hours_per_week", "native_countryIndex")

      val indexed_assembled_table = assembler.transform(indexed_table)

      indexed_assembled_table.show(8)

      val kmeans = new KMeans()
        .setK(8)
        .setMaxIter(500)
        .setSeed(1L)
        .setFeaturesCol("features")

      val model = kmeans.fit(indexed_assembled_table)
      val predictions = model.transform(indexed_assembled_table)

      val evaluator = new ClusteringEvaluator()
      val silhouette = evaluator.evaluate(predictions)
      println(silhouette)

      val clusters = model.clusterCenters
        .map(fields => Row(Vectors.dense(fields.toArray.map(num => math.round(num).toDouble))))

      spark.stop()
      val sc = new SparkContext("local[*]", "AdultBase")
      val clusters_dt = sc.parallelize(clusters)
      sc.stop()

      val reSpark = SparkSession
        .builder()
        .master("local[*]")
        .appName("AdultBaseDatabase")
        .getOrCreate()

      val cluster_table =
        reSpark.createDataFrame(clusters_dt, StructType(Array(attrGroup.toStructField())))

      val vecToArray = udf( (xs: Vector) => xs.toArray )
      val dfArr = cluster_table.withColumn("featuresArray" , vecToArray($"features") )
      val sqlExpr = info_elements.zipWithIndex.map{ case (alias, idx) =>
        col("featuresArray").getItem(idx).as(alias) }
      val split_table = dfArr.select(sqlExpr : _*)

      val cluster_info_split_table = getConverterPipline
        .fit(split_table).transform(split_table)
        .drop("workclassIndex", "educationIndex", "maritial_statusIndex",
          "occupationIndex", "relationshipIndex", "raceIndex", "sexIndex")
      cluster_info_split_table.show()
    }
}
