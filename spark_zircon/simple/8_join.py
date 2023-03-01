import os
import sys
from pyspark.sql import SparkSession, functions as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def main(spark):

    from util.file import get_absolute_file_path

    file_path = get_absolute_file_path("../data/PEP_2017_PEPANNRES.csv")

    censusDf = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("encoding", "cp1252") \
        .load(file_path)

    censusDf = censusDf.drop("GEO.id") \
        .drop("rescen42010") \
        .drop("resbase42010") \
        .drop("respop72010") \
        .drop("respop72011") \
        .drop("respop72012") \
        .drop("respop72013") \
        .drop("respop72014") \
        .drop("respop72015") \
        .drop("respop72016") \
        .withColumnRenamed("respop72017", "pop2017") \
        .withColumnRenamed("GEO.id2", "countyId") \
        .withColumnRenamed("GEO.display-label", "county")

    censusDf.sample(0.1).show(3, False)
    censusDf.printSchema()

    file_path = get_absolute_file_path("../data/InstitutionCampus.csv")

    higherEdDf = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(file_path)

    higherEdDf = higherEdDf \
        .filter("LocationType = 'Institution'") \
        .withColumn("addressElements", F.split(F.col("Address"), " "))

    higherEdDf = higherEdDf.withColumn("addressElementCount",
                                       F.size(F.col("addressElements")))

    higherEdDf = higherEdDf.withColumn("zip9",
                                       F.element_at(F.col("addressElements"),
                                                    F.col("addressElementCount"
                                                          )))

    higherEdDf = higherEdDf.withColumn("splitZipCode",
                                       F.split(F.col("zip9"), "-"))

    higherEdDf = higherEdDf \
        .withColumn("zip", F.col("splitZipCode").getItem(0)) \
        .withColumnRenamed("LocationName", "location") \
        .drop("DapipId", "OpeId", "ParentName", "ParentDapipId",
              "LocationType", "Address", "GeneralPhone", "AdminName",
              "AdminPhone", "AdminEmail", "Fax", "UpdateDate", "zip9",
              "addressElements", "addressElementCount", "splitZipCode")

    higherEdDf.sample(0.1).show(3, False)
    higherEdDf.printSchema()

    file_path = get_absolute_file_path("../data/COUNTY_ZIP_092018.csv")

    countyZipDf = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(file_path)

    countyZipDf = countyZipDf.drop("res_ratio", "bus_ratio",
                                   "oth_ratio", "tot_ratio")

    countyZipDf.sample(0.1).show(3, False)
    countyZipDf.printSchema()

    institPerCountyJoinCondition = higherEdDf["zip"] == countyZipDf["zip"]

    institPerCountyDf = higherEdDf.join(countyZipDf,
                                        institPerCountyJoinCondition,
                                        "inner").drop(countyZipDf["zip"])

    institPerCountyDf.filter(F.col("zip") == 27517).show(20, False)
    institPerCountyDf.printSchema()

    instPerCountyConditn = institPerCountyDf["county"] == censusDf["countyId"]
    institPerCountyDf = institPerCountyDf.join(censusDf,
                                               instPerCountyConditn,
                                               "left").drop(censusDf["county"])

    institPerCountyDf = institPerCountyDf \
        .drop("zip", "county", "countyId") \
        .distinct()

    institPerCountyDf.filter(F.col("zip") == 27517).show(20, False)
    institPerCountyDf.filter(higherEdDf["zip"] == 2138).show(20, False)
    institPerCountyDf.filter("county is null").show(200, False)
    institPerCountyDf.show(200, False)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Join") \
        .master("local[*]").getOrCreate()
    main(spark)
    spark.stop()
