name := "AdultBase"

version := "0.1"

scalaVersion := "2.11.8"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.1"

// https://mvnrepository.com/artifact/org.apache.spark/spark-streaming
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.3.1"

// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.1"

// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib-local
libraryDependencies += "org.apache.spark" %% "spark-mllib-local" % "2.3.1"

// https://mvnrepository.com/artifact/org.scalanlp/breeze-viz
libraryDependencies += "org.scalanlp" %% "breeze-viz" % "0.13.2"
