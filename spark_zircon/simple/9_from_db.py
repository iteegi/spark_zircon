"""From PostgreSQL."""

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark create RDD example") \
    .getOrCreate()

user = 'posrgres'
pw = 'posrgres'
table_name = 'air'

url = 'jdbc:postgresql://192.168.0.154:5432/dataset?user='+user+'&password='+pw
properties = {'driver': 'org.postgresql.Driver', 'password': pw, 'user': user}

df = spark.read.jdbc(url=url, table=table_name, properties=properties)

df.show(5)
df.printSchema()
