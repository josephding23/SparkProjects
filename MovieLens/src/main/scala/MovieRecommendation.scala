import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.jblas.DoubleMatrix
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.evaluation.RankingMetrics

package MovieLens {
  class traitsExtraction {
    def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
      vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
    }

    def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], k: Int): Double = {
      val predK = predicted.take(k)
      var score = 0.0
      var numHits = 0.0
      for ((p, i) <- predK.zipWithIndex) {
        if (actual.contains(p)) {
          numHits += 1.0
          score += numHits / (i.toDouble + 1.0)
        }
      }
      if (actual.isEmpty) {
        1.0
      } else {
        score / scala.math.min(actual.size, k).toDouble
      }
    }

    def recommendMovies(): Unit = {
      val sc = new SparkContext("local[2]", "MovieLens100k")
      val rawData = sc.textFile("data/ml-100k/u.data")
      //println(rawData.first())
      val rawRatings = rawData.map(_.split("\t").take(3))
      //println(rawRatings.first())
      val ratings = rawRatings.map { case Array(user, movie, rating) =>
        Rating(user.toInt, movie.toInt, rating.toDouble)}
      //println(ratings.first())
      val model = ALS.train(ratings, 50, 10, 0.01)
      /*
      println(model.userFeatures.count())
      val predictedRating = model.predict(789, 123)
      println(predictedRating)*/
      val userId = 789
      val K = 10
      val topKRecs = model.recommendProducts(userId, K)
      println(topKRecs.mkString("\n"))
      val movies = sc.textFile("./data/ml-100k/u.item")
      val titles = movies.map(line => line.split("\\|").take(2)).map(array
      => (array(0).toInt, array(1))).collectAsMap()
      //println(titles(123))*/

      val moviesForUser = ratings.keyBy(_.user).lookup(789)
      println(moviesForUser.size)
      moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product), rating.rating)).foreach(println)
      topKRecs.map(rating => (titles(rating.product), rating.rating)).foreach(println)

      //val aMatrix = new DoubleMatrix(Array(1.0, 2.0, 3.0))
      val itemId = 567
      val itemFactor = model.productFeatures.lookup(itemId).head
      val itemVector = new DoubleMatrix(itemFactor)
      println(cosineSimilarity(itemVector, itemVector))

      val sims = model.productFeatures.map {case (id, factor) =>
        val factorVector = new DoubleMatrix(factor)
        val sim = cosineSimilarity(factorVector, itemVector)
        (id, sim)
      }
      val sortedSims = sims.top(K)(Ordering.by[(Int, Double), Double] { case
        (id, similarity) => similarity })
      //println(sortedSims.take(10).mkString("\n"))

      val sortedSims2 = sims.top(K+1)(Ordering.by[(Int, Double), Double] {
        case (id, similarity) => similarity })
      //println(sortedSims2.slice(1, 11).map{ case (id, sim) => (titles(id), sim)}.mkString("\n"))

      val actualRating = moviesForUser.take(1)(0)
      val predictedRating = model.predict(actualRating.user,
        actualRating.product)
      println(predictedRating)

      //val squaredError = math.pow(predictedRating - actualRating.rating, 2.0)
      val userProducts = ratings.map{ case Rating(user, product, rating) => (user, product)}
      val predictions = model.predict(userProducts).map{
        case Rating(user, product, rating) => ((user, product), rating)
      }
      val ratingsAndPredictions = ratings.map{
        case Rating(user, product, rating) => ((user, product), rating)}.join(predictions)
      val MSE = ratingsAndPredictions.map{
        case ((user, product), (actual, predicted)) => math.pow((actual - predicted), 2)
      }.reduce(_ + _) / ratingsAndPredictions.count
      println("Mean Squared Error = "  + MSE)
      val RMSE = math.sqrt(MSE)
      println("Root Mean Squared Error = " + RMSE)

      val actualMovies = moviesForUser.map(_.product)
      //println(actualMovies)
      val predictedMovies = topKRecs.map(_.product)
      val apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)
      //println(apk10)

      val itemFactors = model.productFeatures.map { case (id, factor)
      => factor }.collect()
      val itemMatrix = new DoubleMatrix(itemFactors)
      //println(itemMatrix.rows, itemMatrix.columns)
      val imBroadcast = sc.broadcast(itemMatrix)
      val allRecs = model.userFeatures.map{ case (userId, array) =>
        val userVector = new DoubleMatrix(array)
        val scores = imBroadcast.value.mmul(userVector)
        val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
        val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
        (userId, recommendedIds)
      }
      val userMovies = ratings.map{ case Rating(user, product, rating) =>
        (user, product)}.groupBy(_._1)
      //val K = 10
      val MAPK = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2).toSeq
        avgPrecisionK(actual, predicted, K)
      }.reduce(_ + _) / allRecs.count
      println("Mean Average Precision at K = " + MAPK)

      val predictedAndTrue = ratingsAndPredictions.map{ case ((user,
      product), (predicted, actual)) => (predicted, actual)}
      val regressionMetrics = new RegressionMetrics(predictedAndTrue)
      println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
      println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)

      val predictedAndTrueForRanking= allRecs.join(userMovies).map{ case
        (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2)
        (predicted.toArray, actual.toArray)}
      val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
      println("Mean Average Precision = " + rankingMetrics)

      val MAPK2000 = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2).toSeq
        avgPrecisionK(actual, predicted, 2000)
      }.reduce(_ + _) / allRecs.count
      println("Mean Average Precision = " + MAPK2000)
    }
  }
}
