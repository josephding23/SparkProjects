package AdultBase.manip

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

object BasicDataManipulation {
  def acquireDict(types: Array[String]): Map[String, Int] = {
    var idx = 0
    var dict: Map[String, Int] = Map()
    for (item <- types) {
      dict += (item -> idx)
      idx += 1
    }
    dict
  }

  def dictReverse(map: Map[String, Int]): Map[Int, String] = {
    var dict: Map[Int, String] = Map()
    for (item <- map) {
      dict += (item._2 -> item._1)
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

  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label))
    new MulticlassMetrics(predictionsAndLabels)
  }
}