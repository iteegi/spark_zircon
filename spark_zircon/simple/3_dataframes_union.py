"""CSV ingestion in a dataframe and manipulation."""

import os
import json
import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def get_wake_df(df_wake: DataFrame) -> DataFrame:
    """Builds the dataframe containing the Wake county restaurants.

    :param df_wake: Source dataframe
    :type df_wake: DataFrame
    :return: Cleared dataframe
    :rtype: DataFrame
    """
    df_wake.printSchema()

    print(f'We have {df_wake.count()} records.')

    drop_cols = ["OBJECTID", "GEOCODESTATUS", "PERMITID"]

    df_clean = df_wake.withColumn("county", F.lit("Wake")) \
        .withColumnRenamed("HSISID", "datasetId") \
        .withColumnRenamed("NAME", "name") \
        .withColumnRenamed("ADDRESS1", "address1") \
        .withColumnRenamed("ADDRESS2", "address2") \
        .withColumnRenamed("CITY", "city") \
        .withColumnRenamed("STATE", "state") \
        .withColumnRenamed("POSTALCODE", "zip") \
        .withColumnRenamed("PHONENUMBER", "tel") \
        .withColumnRenamed("RESTAURANTOPENDATE", "dateStart") \
        .withColumn("dateEnd", F.lit(None)) \
        .withColumnRenamed("FACILITYTYPE", "type") \
        .withColumnRenamed("X", "geoX") \
        .withColumnRenamed("Y", "geoY") \
        .drop(*drop_cols)

    df_clean = df_clean.withColumn("id",
                                   F.concat(F.col("state"), F.lit("_"),
                                            F.col("county"), F.lit("_"),
                                            F.col("datasetId")))

    df_clean.show(5)
    df_clean.printSchema()

    return df_clean


def get_durham_df(df_durham: DataFrame) -> DataFrame:
    """Builds the dataframe containing the Durham county restaurants.

    :param df_durham: Source dataframe
    :type df_durham: DataFrame
    :return: Cleared dataframe
    :rtype: DataFrame
    """
    df_durham.printSchema()
    print(f"We have {df_durham.count()} records.")

    drop_cols = ["fields", "geometry", "record_timestamp", "recordid"]

    df_clean = df_durham.withColumn("county", F.lit("Durham")) \
        .withColumn("datasetId", F.col("fields.id")) \
        .withColumn("name", F.col("fields.premise_name")) \
        .withColumn("address1", F.col("fields.premise_address1")) \
        .withColumn("address2", F.col("fields.premise_address2")) \
        .withColumn("city", F.col("fields.premise_city")) \
        .withColumn("state", F.col("fields.premise_state")) \
        .withColumn("zip", F.col("fields.premise_zip")) \
        .withColumn("tel", F.col("fields.premise_phone")) \
        .withColumn("dateStart", F.col("fields.opening_date")) \
        .withColumn("dateEnd", F.col("fields.closing_date")) \
        .withColumn("type", F.split(F.col("fields.type_description"),
                                    " - ").getItem(1)) \
        .withColumn("geoX", F.col("fields.geolocation").getItem(0)) \
        .withColumn("geoY", F.col("fields.geolocation").getItem(1)) \
        .drop(*drop_cols)

    df_clean = df_clean.withColumn("id", F.concat(F.col("state"), F.lit("_"),
                                                  F.col("county"), F.lit("_"),
                                                  F.col("datasetId")))

    df_clean.printSchema()
    return df_clean


def combineDataframes(df1: DataFrame, df2: DataFrame) -> None:
    """Performs the union between the two dataframes.

    :param df1: Left Dataframe to union on
    :type df1: DataFrame
    :param df2: Right Dataframe to union from
    :type df2: DataFrame
    """
    df = df1.unionByName(df2)
    df.show(5)
    df.printSchema()
    print(f"We have {df.count()} records.")
    print(f"Partition count: {df.rdd.getNumPartitions()}")


def main(session: SparkSession) -> None:
    """The processing code."""

    from util.file import get_absolute_file_path

    relative_path_wake = "../data/Restaurants_in_Wake_County_NC.csv"
    file_path_wake = get_absolute_file_path(relative_path_wake)

    relative_path_durham = "../data/Restaurants_in_Durham_County_NC.json"
    file_path_durham = get_absolute_file_path(relative_path_durham)

    df_wake = session.read.csv(
        header=True, inferSchema=True, path=file_path_wake)
    df_durham = session.read.json(file_path_durham)

    df_wake_clean = get_wake_df(df_wake)
    df_durham_clean = get_durham_df(df_durham)

    combineDataframes(df_wake_clean, df_durham_clean)


if __name__ == '__main__':
    session = SparkSession.builder.appName("Union of two DataFrames") \
        .master("local[*]").getOrCreate()
    main(session)
    session.stop()
