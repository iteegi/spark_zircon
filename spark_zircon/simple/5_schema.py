import os
import json
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField,
                               IntegerType, DateType,
                               StringType)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def main():
    from util.file import get_absolute_file_path

    absolute_file_path = get_absolute_file_path("../data/books.csv")

    spark = SparkSession.builder \
        .appName("Complex CSV with a schema to Dataframe") \
        .master("local[*]").getOrCreate()

    schema = StructType([StructField('id', IntegerType(), False),
                        StructField('authorId', IntegerType(), True),
                        StructField('title', StringType(), False),
                        StructField('releaseDate', DateType(), True),
                        StructField('link', StringType(), False)])

    df = spark.read.format("csv") \
        .option("header", True) \
        .option("multiline", True) \
        .option("sep", ";") \
        .option("dateFormat", "MM/dd/yyyy") \
        .option("quote", "*") \
        .schema(schema) \
        .load(absolute_file_path)

    df.show(20, 25, False)
    df.printSchema()

    schemaAsJson = df.schema.json()
    parsedSchemaAsJson = json.loads(schemaAsJson)

    print("*** Schema as JSON: {}".format(json.dumps(parsedSchemaAsJson,
                                                     indent=2)))

    spark.stop()


if __name__ == '__main__':
    main()
