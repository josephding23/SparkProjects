package AdultBase.rdd

import AdultBase.manip.BasicDataManipulation._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

object AdultBaseKMean {

  def rddComputeKMean(): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    val sc = new SparkContext("local[*]", "AdultBase")
    val raw_data = sc.textFile("./data/machine-learning-databases/adult.data")
    val data = raw_data.map(line => line.split(", ")).filter(fields => fields.length == 15)
    data.cache()

    val number_set = data.map(fields => fields(2).toInt).collect().toSet
    val education_types = data.map(fields => fields(3)).distinct.collect()
    val marriage_types = data.map(fields => fields(5)).distinct.collect()
    val family_condition_types = data.map(fields => fields(7)).distinct.collect()
    val occupation_category_types = data.map(fields => fields(1)).distinct.collect()
    val occupation_types = data.map(fields => fields(6)).distinct.collect()
    val racial_types = data.map(fields => fields(8)).distinct.collect()
    val nationality_types = data.map(fields => fields(13)).distinct.collect()
    println(marriage_types.length)

    val education_dict = acquireDict(education_types)
    val marriage_dict = acquireDict(marriage_types)
    val family_condition_dict = acquireDict(family_condition_types)
    val occupation_category_dict = acquireDict(occupation_category_types)
    val occupation_dict = acquireDict(occupation_types)
    val racial_dict = acquireDict(racial_types)
    val nationality_dict = acquireDict(nationality_types)

    val sex_dict = Map("Male" -> 1, "Female" -> 0)

    val marriage_data = data.map { fields => (fields(5), 1) }.reduceByKey((x, y) => x + y).sortBy(_._2)
    val education_data = data.map { fields => (fields(3), 1)}.reduceByKey((x, y) => x + y).sortBy(_._2)
    val doc_marriage_data = data.filter(fields => fields(3) == "Doctorate").map(fields =>
      (fields(5), 1)).reduceByKey((x, y) => x + y).sortBy(_._2)

    println("Marriage data:")
    marriage_data.foreach(println)
    println("Education data:")
    education_data.foreach(println)
    println("Doctor Marriage data:")
    doc_marriage_data.foreach(println)

    val married_set = List("Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse")
    //val unmarried_data = data.filter(fields => married_set.contains(fields(5)))
    val data_set = data.map { fields =>
      val number = fields(2).toInt
      val education = education_dict(fields(3))
      val marriage = marriage_dict(fields(5))
      val family_condition = family_condition_dict(fields(7))
      val occupation_category = occupation_category_dict(fields(1))
      val occupation = occupation_dict(fields(6))
      val sex = sex_dict(fields(9))
      val race = racial_dict(fields(8))
      val nationality = nationality_dict(fields(13))
      Vectors.dense(education, occupation, occupation_category, sex, family_condition, race, nationality, marriage)}

    data_set.take(30).foreach(println)

    val kmeans = new KMeans().setK(4).setSeed(1L).setMaxIterations(80)
    val kMeansModel = kmeans.run(data_set)

    kMeansModel.clusterCenters.foreach { println }

    val core_data = kMeansModel.clusterCenters.toSet
    core_data.foreach(println)

    for (fields <- core_data) {
        val education = dictReverse(education_dict)(fields(0).toInt)
        val occupation = dictReverse(occupation_dict)(fields(1).toInt)
        val occupation_category = dictReverse(occupation_category_dict)(fields(2).toInt)
        val sex = dictReverse(sex_dict)(fields(3).toInt)
        val family_condition = dictReverse(family_condition_dict)(fields(4).toInt)
        val race = dictReverse(racial_dict)(fields(5).toInt)
        val nationality = dictReverse(nationality_dict)(fields(6).toInt)
        val marriage = dictReverse(marriage_dict)(fields(7).toInt)
      val core_tag = Array[String](education, occupation, occupation_category, sex, family_condition, race, nationality, marriage)
      for (data <- core_tag) {
        print(data + ", ")
      }
      println()
    }

    //core_data.foreach(println)


    //val kMeansCost = kMeansModel.computeCost(data_set)
    //println("K-Means Cost: " + kMeansCost)

  }
}
