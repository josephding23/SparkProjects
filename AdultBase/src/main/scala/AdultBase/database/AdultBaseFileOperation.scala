package AdultBase.database

import java.text.NumberFormat

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.{DataFrame, Encoders, SparkSession}
import org.apache.spark.sql.types.StructType

object AdultBaseFileOperation {
  case class Adult (age: Int, workclass: String, fnlwgt: Int,
                     education: String, education_num: Int,
                     maritial_status: String, occupation: String,
                     relationship: String, race: String,
                     sex: String, capital_gain: Int, capital_loss: Int,
                     hours_per_week: Int, native_country: String)

  case class AdultIndexed (age: Int, workclassIndex: Double,
                           educationIndex: Double, maritial_statusIndex: Double,
                           occupationIndex: Double, relationshipIndex: Double,
                           raceIndex: Double, sexIndex: Double,
                           native_countryIndex: Double)

  case class AdultIndexedInfo(age:Int, workclass: String, fnlwgt: Int,
                              education: String, education_num: Int,
                              maritial_status: String, occupation: String,
                              relationship: String, race: String,
                              sex: String, capital_gain: Int, capital_loss: Int,
                              hours_per_week: Int, native_country: String,
                              workclassIndex: Double,
                              educationIndex: Double, maritial_statusIndex: Double,
                              occupationIndex: Double, relationshipIndex: Double,
                              raceIndex: Double, sexIndex: Double,
                              native_countryIndex: Double)

  def getTrainingData: DataFrame = {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("AdultBaseDatabase")
      .getOrCreate()

    import spark.implicits._

    val df_training = spark.sparkContext
      .textFile("./data/machine-learning-databases/adult.data")
      .map(fields => fields.split(", ")).filter(lines => lines.length == 15)
      .map(attributes =>
        Adult(attributes(0).toInt, attributes(1), attributes(2).toInt, attributes(3), attributes(4).toInt,
          attributes(5), attributes(6), attributes(7), attributes(8), attributes(9), attributes(10).toInt,
          attributes(11).toInt, attributes(12).toInt, attributes(13))
      ).toDF()

    df_training
  }

  def getTestData: DataFrame = {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("AdultBaseDatabase")
      .getOrCreate()

    import spark.implicits._

    val df_test = spark.sparkContext
      .textFile("./data/machine-learning-databases/adult.data")
      .map(fields => fields.split(", ")).filter(lines => lines.length == 15)
      .map(attributes =>
        Adult(attributes(0).toInt, attributes(1), attributes(2).toInt, attributes(3), attributes(4).toInt,
          attributes(5), attributes(6), attributes(7), attributes(8), attributes(9), attributes(10).toInt,
          attributes(11).toInt, attributes(12).toInt, attributes(13))
      ).toDF()

    df_test
  }

  def getAdultSchema: StructType = Encoders.product[Adult].schema

  def getAdultIndexSchema: StructType = Encoders.product[AdultIndexed].schema

  def getAdultIndexWithInfoSchema: StructType = Encoders.product[AdultIndexedInfo].schema


  def getWorkclassIndexer: StringIndexer = new StringIndexer()
    .setInputCol("workclass").setOutputCol("workclassIndex").setHandleInvalid("keep")
    .setStringOrderType("frequencyAsc")

  def getEduIndexer: StringIndexer = new StringIndexer()
    .setInputCol("education").setOutputCol("educationIndex").setHandleInvalid("keep")
    .setStringOrderType("frequencyAsc")

  def getMaritalIndexer: StringIndexer = new StringIndexer()
    .setInputCol("maritial_status").setOutputCol("maritial_statusIndex").setHandleInvalid("keep")
    .setStringOrderType("frequencyAsc")

  def getOccupationIndexer: StringIndexer = new StringIndexer()
    .setInputCol("occupation").setOutputCol("occupationIndex").setHandleInvalid("keep")
    .setStringOrderType("frequencyAsc")

  def getRelationshipIndexer: StringIndexer = new StringIndexer()
    .setInputCol("relationship").setOutputCol("relationshipIndex").setHandleInvalid("keep")
    .setStringOrderType("frequencyAsc")

  def getRaceIndexer: StringIndexer = new StringIndexer()
    .setInputCol("race").setOutputCol("raceIndex").setHandleInvalid("keep")
    .setStringOrderType("frequencyAsc")

  def getSexIndexer: StringIndexer = new StringIndexer()
    .setInputCol("sex").setOutputCol("sexIndex").setHandleInvalid("keep")
    .setStringOrderType("frequencyAsc")

  def getNativeIndexer: StringIndexer =  new StringIndexer()
    .setInputCol("native_country").setOutputCol("native_countryIndex").setHandleInvalid("keep")
    .setStringOrderType("frequencyAsc")

  private val indexer_pipeline = new Pipeline().setStages(Array(
    getWorkclassIndexer, getEduIndexer, getMaritalIndexer, getOccupationIndexer,
    getRelationshipIndexer, getRaceIndexer, getSexIndexer, getNativeIndexer
  ))

  def getIndexerPipline: Pipeline = indexer_pipeline

  private val converter_pipeline = new Pipeline().setStages(Array(
    new IndexToString()
      .setInputCol("workclassIndex").setOutputCol("workclass")
      .setLabels(getWorkclassIndexer.fit(getTrainingData).labels),
    new IndexToString()
      .setInputCol("educationIndex").setOutputCol("education")
      .setLabels(getEduIndexer.fit(getTrainingData).labels),
    new IndexToString()
      .setInputCol("maritial_statusIndex").setOutputCol("maritial_status")
      .setLabels(getMaritalIndexer.fit(getTrainingData).labels),
    new IndexToString()
      .setInputCol("occupationIndex").setOutputCol("occupation")
      .setLabels(getOccupationIndexer.fit(getTrainingData).labels),
    new IndexToString()
      .setInputCol("relationshipIndex").setOutputCol("relationship")
      .setLabels(getRelationshipIndexer.fit(getTrainingData).labels),
    new IndexToString()
      .setInputCol("raceIndex").setOutputCol("race")
      .setLabels(getRaceIndexer.fit(getTrainingData).labels),
    new IndexToString()
      .setInputCol("sexIndex").setOutputCol("sex")
      .setLabels(getSexIndexer.fit(getTrainingData).labels)
  ))

  def getConverterPipline: Pipeline = converter_pipeline


  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val df_training = getTrainingData
    val df_test = getTestData

    val format = NumberFormat.getPercentInstance
    format.setMaximumFractionDigits(4)

    val indexed_df_training = getIndexerPipline.fit(df_training).transform(df_training)
      .select("age", "workclassIndex", "fnlwgt", "educationIndex", "maritial_statusIndex",
        "occupationIndex", "relationshipIndex", "raceIndex", "sexIndex", "native_countryIndex")

    val indexed_df_test = getIndexerPipline.fit(df_test).transform(df_test)
      .select("age", "workclassIndex", "fnlwgt", "educationIndex", "maritial_statusIndex",
        "occupationIndex", "relationshipIndex", "raceIndex", "sexIndex", "native_countryIndex")

    val indexed_df_training_withInfo = getIndexerPipline.fit(df_training).transform(df_training)
    val indexed_df_test_withInfo = getIndexerPipline.fit(df_training).transform(df_training)


    df_training.write.format("csv")
      .option("header", "true")
      .option("sep", ",")
      .option("nanValue", "unknown")
      .mode("ignore")
      .save("./data/adult_training")

    df_test.write.format("csv")
      .option("header", "true")
      .option("sep", ",")
      .option("nanValue", "unknown")
      .mode("ignore")
      .save("./data/adult_test")


    indexed_df_training.write.format("csv")
      .option("header", "true")
      .option("sep", ",")
      .mode("ignore")
      .save("./data/indexed_adult_training")

    indexed_df_test.write.format("csv")
      .option("header", "true")
      .option("sep", ",")
      .mode("ignore")
      .save("./data/indexed_adult_test")

    indexed_df_training_withInfo.write.format("csv")
      .option("header", "true")
      .option("sep", ",")
      .option("nanValue", "unknown")
      .mode("ignore")
      .save("./data/indexed_adult_training_withInfo")

    indexed_df_test_withInfo.write.format("csv")
      .option("header", "true")
      .option("sep", ",")
      .option("nanValue", "unknown")
      .mode("ignore")
      .save("./data/indexed_adult_test_withInfo")
  }
}
