import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField,
                               StringType, DoubleType)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def main(spark):
    from util.file import get_absolute_file_path

    file_path = get_absolute_file_path("../data/population_by_country.csv")

    schema = StructType([
        StructField('geo', StringType(), True),
        StructField('yr1980', DoubleType(), False)
    ])

    df = spark.read.csv(header=True, inferSchema=True,
                        schema=schema, path=file_path)

    df.createOrReplaceGlobalTempView("geodata")
    df.printSchema()

    query1 = """
        SELECT * FROM global_temp.geodata
        WHERE yr1980 < 1
        ORDER BY 2
        LIMIT 5
    """
    smallCountriesDf = spark.sql(query1)

    smallCountriesDf.show(10, False)

    query2 = """
        SELECT * FROM global_temp.geodata
        WHERE yr1980 >= 1
        ORDER BY 2
        LIMIT 5
    """

    spark2 = spark.newSession()
    slightlyBiggerCountriesDf = spark2.sql(query2)

    slightlyBiggerCountriesDf.show(10, False)

    spark2.stop()


if __name__ == "__main__":

    spark = SparkSession.builder.appName("Simple SELECT using SQL") \
        .master("local[*]") \
        .getOrCreate()

    main(spark)
    spark.stop()
