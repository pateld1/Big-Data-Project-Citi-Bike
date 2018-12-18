// spark-shell --driver-memory 4G --executor-memory 12G
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.IntParam
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import java.time.Year
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression

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

// Question 1 - What are the top 5 destinations for Citi bikers based on leaving in in the morning/afternoon/evenung?
val times = spark.sql("Select endStationName, endTime, dayTime(hour(endTime)) as Time from cb")
val morning = times.filter($"Time" === "Morning")
morning.createOrReplaceTempView("m")
val afternoon = times.filter($"Time" === "Afternoon")
afternoon.createOrReplaceTempView("a")
val evening = times.filter($"Time" === "Evening")
evening.createOrReplaceTempView("e")
val latenight = times.filter($"Time" === "Late Night")
latenight.createOrReplaceTempView("l")

val morniC = spark.sql("Select distinct(endStationName), COUNT(endStationName) as Number from m group by endStationName order by Number desc limit 10")
val afterC = spark.sql("Select distinct(endStationName), COUNT(endStationName) as Number from a group by endStationName order by Number desc limit 10")
val eveneC = spark.sql("Select distinct(endStationName), COUNT(endStationName) as Number from e group by endStationName order by Number desc limit 10")
val latenC = spark.sql("Select distinct(endStationName), COUNT(endStationName) as Number from l group by endStationName order by Number desc limit 10")

println("Top 5 Desinations in the Morning:")
morniC.show()

println("Top 5 Desinations in the Afternoon:")
afterC.show()

println("Top 5 Destinations in the Evening:")
eveneC.show()

println("Top 5 Destinations Late Night:")
latenC.show()

//morniC.write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("morningData")
//afterC.write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("afternoonData")
//eveneC.write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("nightData")
//latenC.write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("latenightData")

// Question 2 - What are the distribution of usertype per age?

val ages = spark.sql("Select age, substring(userType, 2, 10) as userType from cb order by age")
val subscribers = ages.filter($"userType" === "Subscriber")
val customers = ages.filter($"userType" !== "Subscriber")
subscribers.createOrReplaceTempView("subs")
customers.createOrReplaceTempView("custs")

val subsAge = spark.sql("Select DISTINCT(age) as ageS, COUNT(userType) as NumSubs from subs group by ageS order by ageS asc")
val custsAge = spark.sql("Select DISTINCT(age) as ageC, COUNT(userType) as NumCusts from custs group by ageC order by ageC asc")

val subsCustsAge = subsAge.join(custsAge, $"ageS" === $"ageC")
subsCustsAge.createOrReplaceTempView("subsCustsAgeDF")

val subsCustsAgeTotal = spark.sql("Select ageS, NumSubs, NumCusts, NumSubs + NumCusts as Total, 100 * (NumSubs / (NumSubs + NumCusts)) as PercSubs, 100 * (NumCusts / (NumSubs + NumCusts)) as PercCusts from subsCustsAgeDF order by ageS")

println("Distribution of UserType per Age:")
subsCustsAgeTotal.show(100, false)
//subsCustsAgeTotal.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("subsCustsAgeTotalPerc")

// Question 3 - What is the distribution of bike uses depending on the time of day of start/end and weekday/weekend? Male/Fenale?
val info = spark.sql("Select startTime, endTime, gender from cb")
info.createOrReplaceTempView("temp")
val info2 = spark.sql("Select hour(startTime) as startHour, weekDayEnd(day(dayOfWeek(startTime))) as startDay, hour(endTime) as endHour, weekDayEnd(day(dayOfWeek(endTime))) as endDay, gender from temp")

val maleDay = info2.filter($"gender" === "Male")
val femaleDay = info2.filter($"gender" === "Female")

maleDay.createOrReplaceTempView("md")
femaleDay.createOrReplaceTempView("fd")

val maleStart = spark.sql("Select DISTINCT startHour as startHour, startDay, COUNT(startHour) as Num, 'Leaving' from md group by startHour, startDay order by startHour")
val femaleStart = spark.sql("Select DISTINCT startHour as startHour, startDay, COUNT(startHour) as Num, 'Leaving' from fd group by startHour, startDay order by startHour")
val maleEnd = spark.sql("Select DISTINCT endHour as endHour, endDay, COUNT(endHour) as Num, 'Arriving' from md group by endHour, endDay order by endHour")
val femaleEnd = spark.sql("Select DISTINCT endHour as endHour, endDay, COUNT(endHour) as Num, 'Arriving' from fd group by endHour, endDay order by endHour")

val maleCombined = maleStart.union(maleEnd)
val femaleCombined = femaleStart.union(femaleEnd)

val maleWeekend = maleCombined.filter($"startDay" === "Weekend").select($"startHour".alias("Hour"), $"Num", $"Leaving".alias("Type"))
val maleWeekday = maleCombined.filter($"startDay" === "Weekday").select($"startHour".alias("Hour"), $"Num", $"Leaving".alias("Type"))
val femaleWeekend = femaleCombined.filter($"startDay" === "Weekend").select($"startHour".alias("Hour"), $"Num", $"Leaving".alias("Type"))
val femaleWeekday = femaleCombined.filter($"startDay" === "Weekday").select($"startHour".alias("Hour"), $"Num", $"Leaving".alias("Type"))

println("Distribution of Men using Bikes")
maleCombined.show(100, false)
println("Distribution of Women using Bikes:")
femaleCombined.show(100, false)

//maleWeekend.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("maleWeekend")
//maleWeekday.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("maleWeekday")
//femaleWeekend.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("femaleWeekend")
//femaleWeekday.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("femaleWeekday")

// Question 4 - What are the most common bikes used? Their total hours used? 
val info = spark.sql("Select bikeID, duration from cb")
info.createOrReplaceTempView("bikeInfo")

val mostUsedBikes = spark.sql("Select DISTINCT bikeID, COUNT(bikeID) as Num from bikeInfo group by bikeID order by Num desc limit 10")
mostUsedBikes.write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("mostUsedBikes")

println("Most Used Bikes:")
mostUsedBikes.show(false)

val leastUsedCount = spark.sql("Select DISTINCT bikeID, COUNT(bikeID) from bikeInfo group by bikeID having COUNT(bikeID) < 5")
println("There are these many bikes that were used less than 5 times in 2018:" )
leastUsedCount.count()


val mostHoursUsedBikes = spark.sql("Select DISTINCT bikeID, COUNT(bikeID) as Num, SUM(duration)/(60*60) as HoursUsed from bikeInfo group by bikeID order by HoursUsed desc limit 10")
mostHoursUsedBikes.write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("mostHoursUsedBikes")

println("Most Hours Used Bikes")
mostHoursUsedBikes.show(false)

// Question 5 - What is the average speed that bikers travel at by gender and age? 
val geoInfo = spark.sql("Select duration, startStationName, endStationName, startStationLat, startStationLong, endStationLat, endStationLong, age, gender from cb")
geoInfo.createOrReplaceTempView("gInfo")

val distdf = spark.sql("Select duration /(60*60) as TimeHR, startStationName, endStationName, coordDist(startStationLat, startStationLong, endStationLat, endStationLong) as DistanceKM, age, gender from gInfo")
distdf.createOrReplaceTempView("distDF")

val speed = spark.sql("Select DistanceKM/TimeHR as Speed, Age, gender from distDF where startStationName != endStationName")
speed.createOrReplaceTempView("speedDF")

val distStats = spark.sql("Select DISTINCT Age, AVG(DistanceKM) as avgDist, MAX(DistanceKM) as maxDist, MIN(DistanceKM) as minDist, gender from distDF group by Age, gender order by Age")

println("Distance Statistics by Age and Gender:")
distStats.show(200, false)

val avgSpeed = spark.sql("Select DISTINCT Age, AVG(Speed) as avgSpeed, gender from speedDF group by Age, gender order by Age")

println("Average Speed by Age and Gender:")
avgSpeed.show(200, false)

//distStats.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("distStats")
//avgSpeed.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").option("quote", "\u0000").save("avgSpeed")


// Question 6 - Create a linear regression model that would predict the time duration of a trip depending on distance 

val distTime = distdf.select($"TimeHR".alias("label"), $"DistanceKM")
val assembler = new VectorAssembler().setInputCols(Array("DistanceKM")).setOutputCol("features").transform(distTime).select($"label", $"features")


val lr = new LinearRegression().setRegParam(0.3)
val model = lr.fit(assembler)
val summary = model.summary

println("Model R^2: " + summary.r2)
println("Model RMSE: " + summary.rootMeanSquaredError)




