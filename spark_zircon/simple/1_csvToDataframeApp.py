"""CSV to a relational database."""

import os
from pyspark.sql import SparkSession

current_dir = os.path.dirname(__file__)
relative_path = "../data/books.csv"
absolute_file_path = os.path.join(current_dir, relative_path)


session = SparkSession.builder.appName(
    "CSV to Dataset").master("local[*]").getOrCreate()

df = session.read.csv(header=True, inferSchema=True, path=absolute_file_path)
df.show(5)

session.stop()
