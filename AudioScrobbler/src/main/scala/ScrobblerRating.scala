import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation._
import org.apache.spark.executor._
import org.apache.spark.rdd._

object ScrobblerRating {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local[*]", "Music Recommend")
    val rawUserArtistData = sc.textFile("F:\\Downloads\\AudioScrobbler\\user_artist_data.txt")
    rawUserArtistData.take(5).foreach(println)

    val rawArtistData = sc.textFile("F:\\Downloads\\AudioScrobbler\\artist_data.txt")

    val artistByID = rawArtistData.flatMap{ line =>
      val (id, name) = line.span(_ != '\t')
      if (name.isEmpty) {
        None
      } else {
        try {
          Some((id.toInt, name.trim))
        } catch {
          case e: NumberFormatException => None
        }
      }
    }

    val rawArtistAlias = sc.textFile("F:\\Downloads\\AudioScrobbler\\artist_alias.txt")
    val artistAlias = rawArtistAlias.flatMap{ line =>
      val tokens = line.split("\t")
      if (tokens(0).isEmpty) {
        None
      } else {
        Some((tokens(0).toInt, tokens(1).toInt))
      }
    }.collectAsMap()

    val bArtistAlias = sc.broadcast(artistAlias)

    val trainData = rawUserArtistData.map {line =>
      val Array(userID, artistID, count) = line.split(" ").map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      Rating(userID, finalArtistID, count)
    }

    val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)
    model.userFeatures.mapValues(_.mkString(", ")).first()

    val rawArtistForUser = rawUserArtistData.map(_.split(" ")).
      filter { case Array(user,_ ,_ ) => user.toInt == 2093242}

    val existingProducts = rawArtistForUser.map { case Array(_, artist, _) => artist.toInt}.collect().toSet

    artistByID.filter { case (id, name) =>
      existingProducts.contains(id)}.values.collect().foreach(println)

    val recommendations = model.recommendProducts(2093242, 5)
    recommendations.foreach(println)

    val recommendedProductIDs = recommendations.map(_.product).toSet
    artistByID.filter { case (id, name) =>
      recommendedProductIDs.contains(id)
    }.values.collect().foreach(println)


  }
}