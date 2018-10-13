package AdultBase.database

import java.text.NumberFormat

import breeze.plot._
import AdultBase.database.AdultBaseFileOperation._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object  AdultBaseStatistics {

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

    val df_training = spark.read.format("csv")
      .option("sep", ",")
      .option("header", "true")
      .schema(getAdultSchema)
      .load("./data/adult_training")
      .persist()

    val df_indexed = spark.read.format("csv")
      .option("sep", ",")
      .option("header", "true")
      .schema(getAdultIndexSchema)
      .load("./data/indexed_adult_training")
      .persist()

    val df_indexed_withInfo = spark.read.format("csv")
      .option("sep", ",")
      .option("header", "true")
      .schema(getAdultIndexWithInfoSchema)
      .load("./data/indexed_adult_training_withInfo")
      .persist()

    df_indexed_withInfo.show()
    df_indexed_withInfo.select("workclass", "workclassIndex")
      .distinct().orderBy("workclassIndex").show()


    df_training.printSchema()
    println(df_training.count())
    df_training.show(8)

    val elderly_data = df_training.filter($"age" > 70)
    println(elderly_data.count())
    elderly_data.show(8)
    df_training.filter($"race" === "Other").show(8)

    val race_data = df_indexed.select("raceIndex").collect().map(fields => fields(0).toString.toDouble)
    val marriage_data = df_indexed.select("maritial_statusIndex").collect().map(fields => fields(0).toString.toDouble.toInt)

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

    val total_divorce_rate: Double = df_training.filter($"maritial_status" === "Divorced").count.toDouble /
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

    val figure = Figure()
    val p0 = figure.subplot(0)
    p0 += hist (marriage_data)
    p0.title_=("Marriage")
  }
}
