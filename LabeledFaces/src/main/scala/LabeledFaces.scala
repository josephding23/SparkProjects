import org.apache.spark.SparkContext
import java.awt.image.BufferedImage

import javax.imageio.ImageIO
import java.io.File

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.StandardScaler
import breeze.linalg.DenseMatrix
import breeze.linalg.csvwrite
import org.apache.spark.ml.linalg.DenseMatrix

object LabeledFaces {


  def loadImageFromFile(path: String): BufferedImage = {
    ImageIO.read(new File(path))
  }


  def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
    val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    val g = bwImage.getGraphics
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    bwImage
  }


  def getPixelsFromImage(image: BufferedImage): Array[Double] = {
    val width = image.getWidth
    val height = image.getHeight
    val pixels = Array.ofDim[Double](width * height)
    image.getData.getPixels(0, 0, width, height, pixels)
  }


  def extractPixels(path: String, width: Int, height: Int): Array[Double] = {
    val raw = loadImageFromFile(path)
    val processed = processImage(raw, width, height)
    getPixelsFromImage(processed)
  }


  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local[2]", "Labeled Faces")
    val path = "./data/lfw/*"
    val rdd = sc.wholeTextFiles(path)
    //val first = rdd.firs
    //println(first)
    val files = rdd.map { case (fileName, content) => fileName.replace("file:", "")}
    //println(files.count)

    val aePath = "./data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
    val aeImage = loadImageFromFile(aePath)
    val greyImage = processImage(aeImage, 100, 100)
    ImageIO.write(greyImage, "jpg", new File("./tmp/aeGray.jpg"))

    val pixels = files.map(f => extractPixels(f, 50, 50))
    println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))

    val vectors = pixels.map(p => Vectors.dense(p))
    vectors.setName("image-vectors")
    vectors.cache

    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)
    val scaledVectors = vectors.map(v => scaler.transform(v))

    val matrix = new RowMatrix(scaledVectors)
    val K = 10
    val pc = matrix.computePrincipalComponents(10)

    val rows = pc.numRows
    val cols = pc.numCols
    println(rows, cols)

    val pcBreeze = new breeze.linalg.DenseMatrix(rows, cols, pc.toArray)
    csvwrite(new File("./tmp/pc.csv"), pcBreeze)

    val projected = matrix.multiply(pc)
    println(projected.numRows, projected.numCols)
    println(projected.rows.take(5).mkString("\n"))

    val svd = matrix.computeSVD(10, computeU = true)
    println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols})")
    println(s"S dimension: (${svd.s.size}, )")
    println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols})")
  }
}
