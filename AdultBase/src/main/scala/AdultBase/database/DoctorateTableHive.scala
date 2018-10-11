package AdultBase.database

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Row, SaveMode, SparkSession}

object DoctorateTableHive {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    case class Doctorate(marriage: String, id: Int, sex: String, occupation: String, education: String)

    val warehouseLocation = new File("spark-warehouse").getAbsolutePath
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Doctorate Hive")
      .config("spark.sql.warehouse.dir", warehouseLocation)
      .enableHiveSupport()
      .getOrCreate()

    import spark.implicits._
    import spark.sql

    val indexer_pipeline = new Pipeline().setStages(Array(
        new StringIndexer()
          .setInputCol("occupation").setOutputCol("occupationIndex"),
        new StringIndexer()
          .setInputCol("marriage").setOutputCol("marriageIndex"),
        new StringIndexer()
          .setInputCol("sex").setOutputCol("sexIndex")))

    val converter_pipeline = new Pipeline().setStages(Array(
      new IndexToString()
        .setInputCol("occupationIndex").setOutputCol("occupationString"),
      new IndexToString()
        .setInputCol("marriageIndex").setOutputCol("marriageString"),
      new IndexToString()
        .setInputCol("sexIndex").setOutputCol("sexString")))


    //sql("DROP TABLE IF EXISTS doc")
    sql("CREATE TABLE IF NOT EXISTS doc (marriage STRING, id INT, sex STRING, occupation STRING, education STRING) " +
      "ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' " +
      "LINES TERMINATED BY '\n' STORED AS TEXTFILE")
    sql("LOAD DATA LOCAL INPATH './data/doctorates_marriage_occupation_data' OVERWRITE INTO TABLE doc")

    val doc_table = sql("SELECT * FROM doc").drop($"education")

    val indexed_doc_table = indexer_pipeline.fit(doc_table).transform(doc_table)
        .drop("marriage", "sex", "occupation")
    indexed_doc_table.show(8)

    val original_doc_table = converter_pipeline.fit(indexed_doc_table).transform(indexed_doc_table)
      .drop("marriageIndex", "sexIndex", "occupationIndex")
    original_doc_table.show(8)

    val marriage_doc = doc_table.groupBy("marriage").count().orderBy("count")
    doc_table.groupBy("sex").count().orderBy("count").show()
    doc_table.groupBy("occupation").count().orderBy("count").show()

    val divorced_male = sql("SELECT * FROM doc WHERE sex='Male' AND marriage='Divorced' ORDER BY ID")
    divorced_male.show(8)
    val recordsDoc = divorced_male.map {
      case Row(marriage: String, id: Int, sex: String, occupation: String, education: String) =>
        val suffix = sex match {
          case "Male"=> s"His"
          case "Female" => s"Her"
        }
        s"Doc. $id is $sex, $suffix marriage status is $marriage, occupation is $occupation"
    }
    recordsDoc.show(8, truncate = false)

  }
}
