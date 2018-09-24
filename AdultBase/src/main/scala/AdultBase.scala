import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans

object AdultBase {

  def acquireDict(types: Array[String]): Map[String, Int] = {
    var idx = 0
    var dict: Map[String, Int] = Map()
    for (item <- types) {
      dict += (item -> idx)
      idx += 1
    }
    dict
  }


  def nearestNumber(cores: Set[Int], numbers: Set[Int]): Set[Int] = {
    var arr: Set[Int] = Set()
    for (core: Int <- cores) {
      var r = 0
      val num = core.toInt
      while(!numbers.contains(num+r) && !numbers.contains(num-r)) {
        r += 1
      }
      if(numbers.contains(num+r))
        arr = arr + (num+r)
      else if(numbers.contains(num-r))
        arr = arr + (num-r)
    }
    arr
  }

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local[2]", "Stumble Upon")
    val raw_data = sc.textFile("./data/machine-learning-databases/adult.data")
    val data = raw_data.map(line => line.split(", ")).filter(fields => fields.length == 15)
    data.cache()

    val number_set = data.map(fields => fields(2).toInt).collect().toSet
    val education_types = data.map(fields => fields(3)).distinct.collect()
    val marriage_types = data.map(fields => fields(5)).distinct.collect()
    val family_condition_types = data.map(fields => fields(7)).distinct.collect()
    val occupation_category_types = data.map(fields => fields(1)).distinct.collect()
    val occupation_types = data.map(fields => fields(6)).distinct.collect()

    val education_dict = acquireDict(education_types)
    val marriage_dict = acquireDict(marriage_types)
    val family_condition_dict = acquireDict(family_condition_types)
    val occupation_category_dict = acquireDict(occupation_category_types)
    val occupation_dict = acquireDict(occupation_types)
    val sex_dict = Map("Male" -> 1, "Female" -> 0)

    val marriage_data = data.map { fields => (fields(5), 1) }.reduceByKey((x, y) => x + y)
    val education_data = data.map { fields => (fields(3), 1)}.reduceByKey((x, y) => x + y)
    val doc_marriage_data = data.filter(fields => fields(3) == "Doctorate").map(fields =>
      (fields(5), 1)).reduceByKey((x, y) => x + y)

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
      Vectors.dense(number, education, marriage, family_condition, occupation)}
    data_set.cache

    data_set.take(10).foreach(println)

    val kMeansModel = KMeans.train(data_set, k = 4, maxIterations = 80)
    kMeansModel.clusterCenters.foreach { println }

    val numbers = kMeansModel.clusterCenters.map(centers => centers(0).toInt).toSet
    val core_numbers = nearestNumber(numbers, number_set)
    val core_data = data.filter(fields => core_numbers.contains(fields(2).toInt))

    for (core <- core_data) {
      for (data <- core) {
        print(data + ", ")
      }
      println()
    }

    core_data.foreach(println)


    val kMeansCost = kMeansModel.computeCost(data_set)
    println("K-Means Cost: " + kMeansCost)

    /*
    val writer_mar = new PrintWriter(new File("./data/marriage_data.txt" ))
    val writer_edu = new PrintWriter(new File("./data/education_data.txt" ))
    val writer_doc_mar = new PrintWriter(new File("./data/doc_marriage_data.txt" ))
    for (marriage <- marriage_data) {
      writer_mar.write(marriage._1 + "," + marriage._2 + "\n")
    }
    for (education <- education_data) {
      writer_edu.write(education._1 + "," + education._2 + "\n")
    }
    for (doc_mar <- doc_marriage_data) {
      writer_doc_mar.write(doc_mar._1 + "," + doc_mar._2 + "\n")
    }
    println("Finish File Writing!")*/

  }
}