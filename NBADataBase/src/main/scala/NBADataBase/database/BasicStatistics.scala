package NBADataBase.database

import NBADataBase.database.FileOperation.getStatsSchema
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.bround
import java.io._

object BasicStatistics {
    def main(args: Array[String]): Unit = {
      val spark = SparkSession
        .builder()
        .master("local[*]")
        .appName("NBADataBase")
        .getOrCreate()

      import spark.implicits._

      val season_stats = spark.read.format("csv")
        .option("sep", ",")
        .option("header", "true")
        .schema(getStatsSchema)
        .load("./resources/Seasons_Stats.csv")
        .persist()

      val lebron_stats = season_stats
        .withColumn("AveragePoints", col = bround($"Points" / $"Games", 3))
        .withColumn("AverageRebounds", col = bround($"TotalRebounds" / $"Games", 3))
        .withColumn("AverageAssists", col = bround($"Assists" / $"Games", 3))
        .withColumn("AverageSteals", col = bround($"Steals" / $"Games", 3))
        .withColumn("AverageBlocks", col = bround($"Blocks" / $"Games", 3))
        .filter($"Player" === "LeBron James").orderBy("Year")
        .select("Year", "Player", "Age", "Games", "Team",
          "AveragePoints", "AverageRebounds", "AverageAssists", "AverageSteals", "AverageBlocks",
          "FieldGoalPercentage", "ThreePointPercentage")

      val avePointsList = lebron_stats.select("AveragePoints").collect()
        .map(_(0)).toList
      val aveReboundsList = lebron_stats.select("AverageRebounds").collect()
        .map(_(0)).toList
      val aveAssistsList = lebron_stats.select("AverageAssists").collect()
        .map(_(0)).toList
      val aveStealsList = lebron_stats.select("AverageSteals").collect()
        .map(_(0)).toList
      val aveBlocksList = lebron_stats.select("AverageBlocks").collect()
        .map(_(0)).toList
      val gamesList = lebron_stats.select("Games").collect()
        .map(_(0)).toList
      val fieldPercent = lebron_stats.select("FieldGoalPercentage").collect()
        .map(_(0)).toList
      val threePercent = lebron_stats.select("ThreePointPercentage").collect()
        .map(_(0)).toList
      val yearList = lebron_stats.select("Year").collect()
        .map(_(0)).toList

      val writer = new PrintWriter(new File("./data/lebron_data.csv" ))
      writer.write(yearList.mkString(",") + "\r\n")
      writer.write(avePointsList.mkString(",") + "\r\n")
      writer.write(aveReboundsList.mkString(",") + "\r\n")
      writer.write(aveAssistsList.mkString(",") + "\r\n")
      writer.write(aveStealsList.mkString(",") + "\r\n")
      writer.write(aveBlocksList.mkString(",") + "\r\n")
      writer.write(gamesList.mkString(",") + "\r\n")
      writer.write(fieldPercent.mkString(",") + "\r\n")
      writer.write(threePercent.mkString(",") + "\r\n")
      writer.close()
      /*
      lebron_stats.write.format("csv")
        .option("header", "true")
        .option("sep", ",")
        .mode("ignore")
        .save("./data/LeBron_Stats")*/
  }
}
