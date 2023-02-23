"""CSV ingestion in a dataframe and manipulation."""

import os
import json
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


Path_to_file = str


def main(session: SparkSession, file_path: Path_to_file) -> None:
    df = session.read.csv(header=True, inferSchema=True,
                          path=file_path)

    df.printSchema()

    print(f'We have {df.count()} records.')

    df = df.withColumn("county", F.lit("Wake")) \
        .withColumnRenamed("HSISID", "datasetId") \
        .withColumnRenamed("NAME", "name") \
        .withColumnRenamed("ADDRESS1", "address1") \
        .withColumnRenamed("ADDRESS2", "address2") \
        .withColumnRenamed("CITY", "city") \
        .withColumnRenamed("STATE", "state") \
        .withColumnRenamed("POSTALCODE", "zip") \
        .withColumnRenamed("PHONENUMBER", "tel") \
        .withColumnRenamed("RESTAURANTOPENDATE", "dateStart") \
        .withColumnRenamed("FACILITYTYPE", "type") \
        .withColumnRenamed("X", "geoX") \
        .withColumnRenamed("Y", "geoY") \
        .drop("OBJECTID", "PERMITID", "GEOCODESTATUS")

    df = df.withColumn("id",
                       F.concat(F.col("state"), F.lit("_"),
                                F.col("county"), F.lit("_"),
                                F.col("datasetId")))

    df.show(5)
    df.printSchema()

    print(f'Partition count before repartition: {df.rdd.getNumPartitions()}')

    df = df.repartition(4)

    print(f'Partition count after repartition: {df.rdd.getNumPartitions()}')

    # Schema as JSON
    schemaAsJson = df.schema.json()
    parsedSchemaAsJson = json.loads(schemaAsJson)

    print(f'{json.dumps(parsedSchemaAsJson, indent=2)}')


if __name__ == '__main__':
    from util.file import get_absolute_file_path

    relative_path = "../data/Restaurants_in_Wake_County_NC.csv"
    file_path = get_absolute_file_path(relative_path)

    session = SparkSession.builder.appName("Restaurants in Wake County, NC") \
        .master("local[*]").getOrCreate()
    main(session, file_path)
    session.stop()
