package AdultBase.database

import AdultBase.manip
import java.text.NumberFormat

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession

object  AdultBaseStatistics {
  case class Adult (
                     age: Int, workclass: String, fnlwgt: Int,
                     education: String, education_num: Int,
                     maritial_status: String, occupation: String,
                     relationship: String, race: String,
                     sex: String, capital_gain: Int, capital_loss: Int,
                     hours_per_week: Int, native_country: String
                   )


  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val format = NumberFormat.getPercentInstance
    format.setMaximumFractionDigits(4)

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

    val df_test = spark.sparkContext
      .textFile("./data/machine-learning-databases/adult.data")
      .map(fields => fields.split(", ")).filter(lines => lines.length == 15)
      .map(attributes =>
        Adult(attributes(0).toInt, attributes(1), attributes(2).toInt, attributes(3), attributes(4).toInt,
          attributes(5), attributes(6), attributes(7), attributes(8), attributes(9), attributes(10).toInt,
          attributes(11).toInt, attributes(12).toInt, attributes(13))
      ).toDF()


    df_training.write.format("csv").option("header", "true").save("./data/adult_training")
    df_test.write.format("csv").option("header", "true").save("./data/adult_test")
    
    df_training.printSchema()
    println(df_training.count())
    df_training.show(8)

    val elderly_data = df_training.filter($"age" > 70)
    println(elderly_data.count())
    elderly_data.show(8)
    df_training.filter($"race" === "Other").show(8)

    df_training.groupBy($"race").count().orderBy("count").show()
    df_training.groupBy($"education").count().orderBy("count").show()

    df_training.createOrReplaceTempView("adult")
    spark.sql("SELECT * FROM adult").show(8)
    val dude = spark.sql("SELECT * FROM adult WHERE fnlwgt=77516")
    println(dude.map(info => "Aged " + info.getAs[String]("age") +
      ", occupation is " + info.getAs[String]("occupation") +
      ", race is " + info.getAs[String]("race")).first())

    df_training.write.bucketBy(5, "race").sortBy("age").saveAsTable("race_age_table")
    spark.sql("SELECT * FROM race_age_table").show(8)

    val total_divorce_rate: Double = df_training.filter($"maritial_status" === "Divorced").count.toDouble/
      df_training.count.toDouble
    val doc_divorce_rate: Double = df_training.
      filter($"education" === "Doctorate" and $"maritial_status" === "Divorced").count.toDouble /
      df_training.filter($"education" === "Doctorate").count.toDouble
    val preschool_divorce_rate: Double = df_training.
      filter($"education" === "Preschool" and $"maritial_status" === "Divorced").count.toDouble /
      df_training.filter($"education" === "Preschool").count.toDouble
    val HS_divorce_rate: Double = df_training.
      filter($"education" === "HS-grad" and $"maritial_status" === "Divorced").count.toDouble /
      df_training.filter($"education" === "HS-grad").count.toDouble


    println("Doctorate divorce rate: " + format.format(doc_divorce_rate))
    println("Preschool divorce rate: " + format.format(preschool_divorce_rate))
    println("High School graduation divorce rate: " + format.format(HS_divorce_rate))
    println("Total divorce rate: " + format.format(total_divorce_rate))

    df_training.write
      .partitionBy("workclass")
      .bucketBy(6, "relationship")
      .saveAsTable("workclass_relationship_table")

    df_training.filter($"education" === "Doctorate")
      .select($"maritial_status", $"fnlwgt", $"sex", $"occupation", $"education")
      .write.csv("./data/doctorates_marriage_occupation_data")

  }
}
