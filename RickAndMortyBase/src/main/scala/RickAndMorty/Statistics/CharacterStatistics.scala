package RickAndMorty.Statistics

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object CharacterStatistics {
  case class Character (
                         ID: Int,	Name: String, Status: String,	Species: String, Type: String,
                         Gender: String, Origin: Int, Location: Int, Episode_Created: Array[Int], Null: String
                       )
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getRootLogger.setLevel(Level.WARN)

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("RickAndMorty")
      .getOrCreate()

    import spark.implicits._
    val characters_raw_data = spark.read.format("csv")
      .option("sep", "\t")
      .option("header", "true")
      .load("./data/rick-and-morty-data/characters.csv")

    val episodes_raw_data = spark.read.format("csv")
        .option("sep", "\t")
        .option("header", "true")
        .load("./data/rick-and-morty-data/episodes.csv")

    val locations_raw_data = spark.read.format("csv")
      .option("sep", "\t")
      .option("header", "true")
      .load("./data/rick-and-morty-data/locations.csv")
      //.as[Character]
    //characters_raw_data.show()

    episodes_raw_data.show()
    locations_raw_data.show()
    characters_raw_data.groupBy("species").count().orderBy(desc("count")).show()
  }
}
