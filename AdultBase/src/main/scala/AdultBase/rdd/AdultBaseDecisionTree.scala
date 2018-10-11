package AdultBase.rdd

import AdultBase.manip.BasicDataManipulation._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

object AdultBaseDecisionTree {

  def rddComputeDecisionTree(): Unit = {
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
      val featureVector = Vectors.dense(education, occupation, occupation_category, sex, family_condition, race, nationality)
      val label = marriage
      LabeledPoint(label, featureVector)}

    data_set.take(30).foreach(println)



    val Array(trainData, cvData) = data_set.randomSplit(Array(0.9, 0.1))
    trainData.cache
    cvData.cache

    val model = DecisionTree.
      trainClassifier(trainData, 7, Map[Int, Int](), "entropy", 10, 100)

    val predictionsAndLabels = cvData.map(example =>
      (model.predict(example.features), example.label)
    )
    val metrics = new MulticlassMetrics(predictionsAndLabels)
    println(metrics.precision)


    val evaluations =
    for (impurity <- Array("gini", "entropy");
         depth <- Array(1, 10, 25);
         bins <- Array(10, 50, 150))
    yield{
      val _model = DecisionTree.
        trainClassifier(trainData, 7, Map[Int, Int](), impurity, depth, bins)
      val _predictionsAndLabels = cvData.map(example =>
        (_model.predict(example.features), example.label)
      )
      val _accuracy = new MulticlassMetrics(_predictionsAndLabels).precision
      ((depth, bins, impurity), _accuracy)
    }

    evaluations.sortBy(_._2).reverse.foreach(println)

    /*
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
    }*/

    //core_data.foreach(println)


    //val kMeansCost = kMeansModel.computeCost(data_set)
    //println("K-Means Cost: " + kMeansCost)

  }
}
