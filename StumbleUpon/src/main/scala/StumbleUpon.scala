import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.StandardScaler

object StumbleUpon {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local[2]", "Stumble Upon")
    val rawData = sc.textFile("./data/train_noheader.tsv")
    val records = rawData.map(line => line.split("\t"))

    val data = records.map {r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d =>
        if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }
    data.cache
    val numData = data.count
    //println(numData)

    val nbData = records.map {r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d =>
        if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }

    val numIterations = 10
    val maxTreeDepth = 5
    val lrModel = LogisticRegressionWithSGD.train(data, numIterations)
    val svmModel = SVMWithSGD.train(data, numIterations)
    val nbModel = NaiveBayes.train(nbData)
    val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)

    val dataPoint = data.first
    val prediction = lrModel.predict(dataPoint.features)
    //println(prediction)
    val trueLabel = dataPoint.label
    //println(trueLabel)
    val predictions = lrModel.predict(data.map(lp => lp.features))
    //println(predictions.take(5))

    val lrTotalCorrect = data.map {point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val lfAccuracy = lrTotalCorrect / data.count

    val svmTotalCorrect = data.map { point =>
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val svmAccuracy = svmTotalCorrect / numData

    val nbTotalCorrect = nbData.map { point =>
      if (nbModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val nbAccuracy = nbTotalCorrect / numData

    val dtTotalCorrect = data.map { point =>
      val score = dtModel.predict(point.features)
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == point.label) 1 else 0
    }.sum
    val dtAccuracy = dtTotalCorrect / numData

    println("lfAccuracy: " + lfAccuracy)
    println("scmAccuracy: " + svmAccuracy)
    println("nbAccuracy: " + nbAccuracy)
    println("dtAccuracy: " + dtAccuracy)

    val metrics = Seq(lrModel, svmModel).map {model =>
      val scoreAndLabels = data.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }

    val nbMetrics = Seq(nbModel).map { model =>
      val scoreAndLabels = nbData.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }

    val dtMetrics = Seq(dtModel).map{ model =>
      val scoreAndLabels = data.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }

    val allMetrics = metrics ++ nbMetrics ++ dtMetrics
    allMetrics.foreach{ case (m, pr, roc) =>
      println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under POC: ${roc * 100.0}%2.4f%%")}

    val vectors = data.map(lp => lp.features)
    val matrix = new RowMatrix(vectors)
    val matrixSummary = matrix.computeColumnSummaryStatistics()
    //println(matrixSummary.mean)
    //println(matrixSummary.min)
    //println(matrixSummary.max)
    //println(matrixSummary.variance)
    //println(matrixSummary.numNonzeros)

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    println(data.first.features)
    println(scaledData.first.features)

    val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
    val lrTotalCorrectScaled = scaledData.map { point =>
      if (lrModelScaled.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracyScaled = lrTotalCorrectScaled / numData
    val lrPredictionsVsTrue = scaledData.map { point =>
      (lrModelScaled.predict(point.features), point.label)
    }
    val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
    val lrPr = lrMetricsScaled.areaUnderPR
    val lrRoc = lrMetricsScaled.areaUnderROC
    println(f"${lrModelScaled.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaled* 100}%2.4f%%\nArea under PR: " +
      f"${lrPr * 100}%2.4f%%\nArea under ROC: ${lrRoc * 100.0}%2.4f%%")

    val categories = records.map(r => r(3)).distinct.distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size
    println(categories)
    println(numCategories)

    val dataCategories = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if(d == "?") 0.0 else d.toDouble)
      val features = categoryFeatures ++ otherFeatures
      LabeledPoint(label, Vectors.dense(features))
    }
    println(dataCategories.first)

    val scaledCats = new StandardScaler(withMean = true, withStd = true).fit(dataCategories.map(lp => lp.features))
    val scaledDataCats = dataCategories.map(lp =>
    LabeledPoint(lp.label, scaledCats.transform(lp.features)))
    println(dataCategories.first.features)
    println(scaledDataCats.first.features)

    val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats, numIterations)
    val lrTotalCorrectedScaledCats = scaledDataCats.map { point =>
      if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracyScaledCats = lrTotalCorrectedScaledCats / numData
    val lrPredictionsVsTrueCats = scaledDataCats.map { point =>
      (lrModelScaledCats.predict(point.features), point.label)
    }
    val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
    val lrPrCats = lrMetricsScaledCats.areaUnderPR
    val lrRocCats = lrMetricsScaledCats.areaUnderROC
    println(f"${lrModelScaledCats.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaledCats * 100}%2.4f%%\nArea under PR: " +
      f"${lrPrCats * 100.0}%2.4f%%\nArea under ROC: ${lrRocCats * 100.0}%2.4f%%")

    val dataNB = records.map {r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      LabeledPoint(label, Vectors.dense(categoryFeatures))
    }
    val nbModelCats = NaiveBayes.train(dataNB)
    val nbTotalCorrectCats = dataNB.map { point =>
      if (nbModelCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val nbAccuracyCats = nbTotalCorrectCats / numData
    val nbPredictionsVsTrueCats = dataNB.map { point => (nbModelCats.predict(point.features), point.label)}
    val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
    val nbPrCats = nbMetricsCats.areaUnderPR
    val nbRocCats = nbMetricsCats.areaUnderROC()
    println(f"${nbModelCats.getClass.getSimpleName}\nAccuracy: ${nbAccuracyCats * 100}%2.4f%%\nArea under PR: " +
      f"${nbPrCats * 100.0}%2.4f%%\nArea under ROC: ${nbRocCats * 100.0}%2.4f%%")
  }
}