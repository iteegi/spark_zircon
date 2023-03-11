import os
import sys

from sklearn.metrics import r2_score

from pyspark.sql.dataframe import DataFrame
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


Path_to_file = str
Dense_vector = DataFrame
Evaluation = float
R2 = float


def transData(data: DataFrame) -> Dense_vector:
    """Convert the data to dense vector (features and label)."""
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]), r[-1]]) \
        .toDF(['features', 'label'])


def get_rsme(data: DataFrame, labelCol: str, predictionCol: str) -> Evaluation:
    """Evaluates the output with optional parameters."""
    evaluator = RegressionEvaluator(labelCol=labelCol,
                                    predictionCol=predictionCol,
                                    metricName="rmse")

    return evaluator.evaluate(data)


def get_r2_score(data: DataFrame, label: str, prediction: str) -> R2:
    """Return coefficient of determination."""
    y_true = data.select(label).toPandas()
    y_pred = data.select(prediction).toPandas()

    return r2_score(y_true, y_pred)


def main(spark: SparkSession, file_path: Path_to_file) -> None:
    df = spark.read.format('csv') \
        .options(header='true',
                 inferschema='true') \
        .load(file_path, header=True)

    df.show(5, True)
    df.printSchema()

    df.describe().show()

    transformed = transData(df)
    transformed.show(5)

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values
    # are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features",
                                   outputCol="indexedFeatures",
                                   maxCategories=4).fit(transformed)

    data = featureIndexer.transform(transformed)
    data.show(5, True)

    # split the data
    (trainingData, testData) = transformed.randomSplit([0.6, 0.4])
    trainingData.show(5)
    testData.show(5)

    dt = DecisionTreeRegressor(featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[featureIndexer, dt])

    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)
    predictions.select("features", "label", "prediction").show(5)

    rsme = get_rsme(predictions, "label", "prediction")
    print(f"Root Mean Squared Error (RMSE) on test data = {rsme}")

    r2 = get_r2_score(predictions, "label", "prediction")
    print(f"r2_score: {r2}")


if __name__ == '__main__':
    from util.file import get_absolute_file_path

    relative_path = "../data/Advertising.csv"
    file_path = get_absolute_file_path(relative_path)

    spark = SparkSession \
        .builder \
        .appName("Decision tree regression") \
        .master("local[*]") \
        .getOrCreate()

    main(spark, file_path)
    spark.stop()
