// spark-shell --driver-memory 4G --executor-memory 12G -i testing.scala
import org.apache.spark.rdd.RDD
import org.apache.spark.util.IntParam
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import java.time.Year
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import scala.collection.mutable.ListBuffer

case class Citibike(duration: Integer, startTime: String, endTime: String, startStationID: String, startStationName: String, 
	startStationLat: Float, startStationLong: Float, endStationID: String, endStationName: String, endStationLat: Float, 
	endStationLong: Float, bikeID: Integer, userType: String, birthYear: Integer, gender: Integer)


def toGender(g: Integer): String = {

	if(g == 1){"Male"}
	else if(g == 2){"Female"}
	else{"Unknown"}
}

spark.udf.register("toGender", toGender _)

def timeOfDay(hr: Integer): String = {

	if(hr >= 22 || hr < 5){"Late Night"}
	else if(hr >= 5 && hr < 12){"Morning"}
	else if(hr >= 12 && hr < 18){"Afternoon"}
	else{"Evening"}
}

spark.udf.register("dayTime", timeOfDay _) 

def ageGroup(age: Int): String = {

	if(age <= 24){"16 to 24"}
	else if(age <= 32){"25 to 32"}
	else if(age <= 40){"33 to 40"}
	else if(age <= 48){"41 to 48"}
	else if(age <= 56){"49 to 56"}
	else if(age <= 64){"57 to 64"}
	else if(age <= 72){"65 to 72"}
	else{"73 to 80"}
}

spark.udf.register("groupAge", ageGroup _)

def day(d: Int): String = {

	if(d == 1){"Sunday"}
	else if(d == 2){"Monday"}
	else if(d == 3){"Tuesday"}
	else if(d == 4){"Wednesday"}
	else if(d == 5){"Thursday"}
	else if(d == 6){"Friday"}
	else{"Saturday"}
}

spark.udf.register("day", day _)

def weekDayEnd(day: String): String = {

	if(day == "Saturday" || day == "Sunday"){"Weekend"}
	else{"Weekday"}
}

spark.udf.register("weekDayEnd", weekDayEnd _)

def dist(start_Lat: Float, start_Long: Float, end_Lat: Float, end_Long: Float): Double = {
	var r = 6371
	var latDist = scala.math.toRadians(end_Lat - start_Lat)
	var longDist = scala.math.toRadians(end_Long - start_Long)
	var left = scala.math.pow(scala.math.sin(latDist/2), 2)
	var rightF = scala.math.cos(start_Lat) * scala.math.cos(end_Lat)
	var rightS = scala.math.pow(scala.math.sin(longDist/2), 2)
	var right = rightF * rightS
	var temp = scala.math.asin(scala.math.sqrt(left + right))
	var d = 2 * r * temp
	d
}

spark.udf.register("coordDist", dist _)

def currentAge(yearOfBirth: Int): Integer = {Year.now.getValue - yearOfBirth}

spark.udf.register("age", currentAge _)

def parseCitibike(str: String): Citibike = {
	val line = str.split(",")
	Citibike(line(0).toInt, line(1), line(2), line(3), line(4), line(5).toFloat, line(6).toFloat, line(7), line(8), line(9).toFloat, line(10).toFloat, 
	line(11).toInt, line(12), line(13).toInt, line(14).toInt)
}

def parseRDD(rdd: RDD[String]): RDD[Citibike] = {
	val header = rdd.first
	rdd.filter(_(0) != header(0)).map(parseCitibike).cache()
}

val cdJan = parseRDD(sc.textFile("201801-citibike-tripdata.csv"))
val cdFeb = parseRDD(sc.textFile("201802-citibike-tripdata.csv"))
val cdMar = parseRDD(sc.textFile("201803-citibike-tripdata.csv"))
val cdApr = parseRDD(sc.textFile("201804-citibike-tripdata.csv"))
val cdMay = parseRDD(sc.textFile("201805-citibike-tripdata.csv"))
val cdJun = parseRDD(sc.textFile("201806-citibike-tripdata.csv"))
val cdJul = parseRDD(sc.textFile("201807-citibike-tripdata.csv"))
val cdAug = parseRDD(sc.textFile("201808-citibike-tripdata.csv"))
val cdSep = parseRDD(sc.textFile("201809-citibike-tripdata.csv"))
val cdOct = parseRDD(sc.textFile("201810-citibike-tripdata.csv"))

val twenty18 = cdJan.union(cdFeb).union(cdMar).union(cdApr).union(cdMay).union(cdJun).union(cdJul).union(cdAug).union(cdSep).union(cdOct)
val twenty18df = twenty18.toDF()
twenty18df.createOrReplaceTempView("df")

val temp = spark.sql("Select duration, CAST(substring(startTime, 2, 24) as Timestamp), CAST(substring(endTime, 2, 24) as Timestamp), startStationID, startStationName, startStationLat, startStationLong, endStationID, endStationName, endStationLat, endStationLong, bikeID, userType, age(birthyear) as age, toGender(gender) as gender from df")
val nonFiltered = temp.select($"duration", $"CAST(substring(startTime, 2, 24) AS TIMESTAMP)".alias("startTime"), $"CAST(substring(endTime, 2, 24) AS TIMESTAMP)".alias("endTime"), $"startStationName", $"startStationLat", $"startStationLong", $"endStationName", $"endStationLat", $"endStationLong", $"bikeID", $"userType", $"age", $"gender")
val representedPeople = nonFiltered.filter($"gender" !== "Unknown").filter($"age" <= 65).filter($"startStationName" !== $"endStationName")
val shortTerm = representedPeople.filter($"duration" <= 60 * 30).filter($"duration" >= 120).filter(substring($"userType", 2, 10) !== "Subscriber")
val annualMembers = representedPeople.filter($"duration" <= 60 * 45).filter(substring($"userType", 2, 10) === "Subscriber")
val df = shortTerm.unionAll(annualMembers)

df.createOrReplaceTempView("cb")

val rfVals = spark.sql("Select startStationName as start, endStationName as end, (CAST(duration AS Double)) as label, age, gender, userType from cb")

val startIndexed = new StringIndexer().setInputCol("start").setOutputCol("startIndex").fit(rfVals).transform(rfVals)
val endIndexed = new StringIndexer().setInputCol("end").setOutputCol("endIndex").fit(startIndexed).transform(startIndexed)
val genderIndexed = new StringIndexer().setInputCol("gender").setOutputCol("genderIndex").fit(endIndexed).transform(endIndexed)
val userTypeIndexed = new StringIndexer().setInputCol("gender").setOutputCol("userTypeIndex").fit(genderIndexed).transform(genderIndexed)
val assembler = new VectorAssembler().setInputCols(Array("startIndex", "endIndex", "age", "genderIndex", "userTypeIndex")).setOutputCol("features").transform(userTypeIndexed)
val rf = assembler.select($"label", $"features")
val subset = rf.sample(false, 0.1)
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "variance"
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(rf)
val Array(train, test) = subset.randomSplit(Array(0.7, 0.3))
var scores = new ListBuffer[Double]()



for(i <- 0 to 30){

	println("Max Depth = " + i)
	val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures").setMaxBins(828).setMaxDepth(i)
	val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))
	val model = pipeline.fit(train)
	val predictions = model.transform(test)
	val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
	val rmse = evaluator.evaluate(predictions)
	scores += rmse
	println("RMSE: " + rmse)
}






//val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures").setMaxBins(828).setMaxDepth(1)
//val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))
//val model = pipeline.fit(train)
//val predictions = model.transform(test)
//val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
//val rmse = evaluator.evaluate(predictions)



//val tree = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
//println(tree.toDebugString)



















