"""Calculate PI."""

import time
from pyspark.sql import SparkSession
from random import random
from operator import add


def throw_darts(_):
    """Calculate if the dart hit the circle with radius 1."""
    x = random() * 2 - 1
    y = random() * 2 - 1
    return 1 if x ** 2 + y ** 2 <= 1 else 0


def main(spark: SparkSession) -> None:
    """Approximately calculate the number PI.

    :param spark: Entry point to PySpark
    :type spark: SparkSession
    """
    slices = 10
    numberOfThrows = 100000 * slices
    print(f"About to throw {numberOfThrows} darts")

    t0 = int(round(time.time() * 1000))

    numList = []

    for x in range(numberOfThrows):
        numList.append(x)

    incrementalRDD = spark.sparkContext.parallelize(numList)

    t1 = int(round(time.time() * 1000))
    print(f"Initial dataframe built in {t1 - t0} ms")

    dartsRDD = incrementalRDD.map(throw_darts)

    t2 = int(round(time.time() * 1000))
    print(f"Throwing darts done in {t2 - t1} ms")

    dartsInCircle = dartsRDD.reduce(add)

    t3 = int(round(time.time() * 1000))
    print(f"Analyzing result in {t3 - t2} ms")

    print(f"Pi is roughly {4.0 * dartsInCircle/numberOfThrows}")


if __name__ == '__main__':
    t0 = int(round(time.time() * 1000))

    spark = SparkSession.builder.appName("PySpark Pi") \
        .master("local[*]") \
        .getOrCreate()

    t1 = int(round(time.time() * 1000))
    print(f"Session initialized in {t1 - t0} ms")

    main(spark)
    spark.stop()
