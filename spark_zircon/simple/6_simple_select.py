import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType,StructField,
                               StringType,DoubleType)


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

    df.createOrReplaceTempView('geodata')
    df.printSchema()

    query = """
      SELECT * FROM geodata
      WHERE yr1980 < 1
      ORDER BY 2
      LIMIT 5
    """

    smallCountries = spark.sql(query)
    smallCountries.show(10, False)


if __name__ == "__main__":

    spark = SparkSession.builder.appName("Simple SELECT using SQL") \
        .master("local[*]") \
        .getOrCreate()

    main(spark)
    spark.stop()
