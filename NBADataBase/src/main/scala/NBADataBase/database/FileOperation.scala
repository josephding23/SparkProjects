package NBADataBase.database

import org.apache.spark.sql.{Encoders, SparkSession}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._


object FileOperation {
  case class seasonStats (ID: Int, Year: Int, Player: String, Pos: String, Age: Int, Team: String,
                          Games: Double, GameStarted: Double, MinutesPlayed: Double, PlayerEfficiencyRating: Double,
                          TrueShootingPercentage: Double, TreePointAttemptRate: Double, FreeThrowRate: Double,
                          OffensiveReboundPercentage: Double, DefensiveReboundPercentage: Double, TotalReboundPercentage: Double,
                          AssistPercentage: Double, StealPercentage: Double, BlockPercentagee: Double, TurnoverPercentage: Double,
                          UsagePercentage: Double, blanl: String,
                          OffensiveWinShares: Double, DefensiveWinShares: Double, WinShares: Double,
                          WinSharesPer48Minutes: Double, blank2: String,
                          OffensiveBoxPlusOrMinus: String, DefensiveBoxPlusOrMinus: String, BoxPlusOrMinus: Double,
                          ValueOverReplacement: Double,
                          FieldGoals: Double, FieldGoalAttempts: Double, FieldGoalPercentage: Double,
                          ThreePointGoals: Double, ThreePointAttempts: Double, ThreePointPercentage: Double,
                          TwoPointGoals: Double, TwoPointAttempts: Double, TwoPointPercentage: Double,
                          EffectiveFieldGoalPercentage: Double,
                          FreeThrows: Double, FreeThrowAttempts: Double, FreeThrowPercentage: Double,
                          OffensiveRebounds: Double, DefensiveRebounds: Double, TotalRebounds: Double,
                          Assists: Double, Steals: Double, Blocks: Double, Turnovers: Double,
                          PersonalFouls: Double, Points: Double)

  case class player (ID: Int, Player: String, height: Double, weight: Double, collage: String, born: String,
                     birth_city: String, birth_state: String)

  def getStatsSchema: StructType = Encoders.product[seasonStats].schema

  def getPlayerSchema: StructType = Encoders.product[player].schema

}