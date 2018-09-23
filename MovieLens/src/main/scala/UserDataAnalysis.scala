import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import breeze.plot._

package MovieLens {
  class UserDataAnalysis {
    def analyzeUserData: Unit = {
      val sc = new SparkContext("local[2]", "MovieLens100k")
      val user_data = sc.textFile("data/ml-100k/u.user")
      //println(user_data.first())
      println(user_data.first().split("\\|"))
      val user_fields = user_data.map(line => line.split("\\|"))
      val num_users = user_fields.map(fields => fields(0)).count()
      val num_genders = user_fields.map(fields => fields(2)).distinct().count()
      val num_occupations = user_fields.map(fields => fields(3)).distinct().count()
      val num_zipcodes = user_fields.map(fields => fields(4)).distinct().count()
      println("Users: %d, genders: %d, occupations: %d, ZIP codes: %d".
        format(num_users, num_genders, num_occupations, num_zipcodes))
      val ages = user_fields.map(x => x(1).toInt).collect()
      //hist(ages, bins=20, color)
      val f = Figure()
      val p = f.subplot(0)
      p += hist(ages, 20)
      f.saveas("./resources/1.png")
    }
  }
}
