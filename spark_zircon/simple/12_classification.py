import os
import sys
import itertools
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorIndexer, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


Path_to_file = str


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues) -> None:
    """Prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_dummy(df: DataFrame,
              categoricalCols: List[str],
              continuousCols: List[str],
              labelCol: str) -> DataFrame:
    """Get dummy variables and concat with continuous variables for modeling.

    :param df: the dataframe
    :type df: DataFrame
    :param categoricalCols: the name list of the categorical data
    :type categoricalCols: List[str]
    :param continuousCols: the name list of the numerical data
    :type continuousCols: List[str]
    :param labelCol: the name of label column
    :type labelCol: str
    :return: feature matrix
    :rtype: DataFrame
    """

    from pyspark.ml import Pipeline
    from pyspark.sql.functions import col
    from pyspark.ml.feature import (StringIndexer,
                                    OneHotEncoder,
                                    VectorAssembler)

    indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                for c in categoricalCols]

    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(),
                              outputCol=f"{indexer.getOutputCol()}_encoded")
                for indexer in indexers]

    assembler = VectorAssembler(
        inputCols=[encoder.getOutputCol() for encoder in encoders]
        + continuousCols,
        outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    model = pipeline.fit(df)
    data = model.transform(df)

    data = data.withColumn('label', col(labelCol))

    return data.select('features', 'label')


def visualization(predictions: DataFrame) -> None:
    class_temp = (predictions.select("label")
                  .groupBy("label")
                  .count()
                  .sort('count', ascending=False)
                  .toPandas())
    class_temp = class_temp["label"].values.tolist()
    class_names = list(map(str, class_temp))

    y_true = predictions.select("label")
    y_true = y_true.toPandas()

    y_pred = predictions.select("predictedLabel")
    y_pred = y_pred.toPandas()

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    print(f"confusion_matrix:\n {cnf_matrix}")

    plt.figure()
    plot_confusion_matrix(cnf_matrix,
                          classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix,
                          classes=class_names,
                          normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def main(spark: SparkSession, file_path: Path_to_file) -> None:
    df = spark.read.format('csv') \
        .options(header='true',
                 inferschema='true',
                 sep=';') \
        .load(file_path, header=True)

    df.drop('day', 'month', 'poutcome').show(5)
    df.printSchema()

    # Deal with categorical data and Convert the data to dense vector
    catcols = ['job', 'marital', 'education', 'default',
               'housing', 'loan', 'contact', 'poutcome']

    num_cols = ['balance', 'duration', 'campaign', 'pdays', 'previous',]
    labelCol = 'y'

    data = get_dummy(df, catcols, num_cols, labelCol)
    data.show(5)

    # Index labels, adding metadata to the label column
    labelIndexer = StringIndexer(inputCol='label',
                                 outputCol='indexedLabel').fit(data)
    labelIndexer.transform(data).show(5, True)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values
    # are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features",
                                   outputCol="indexedFeatures",
                                   maxCategories=4).fit(data)
    featureIndexer.transform(data).show(5, True)

    # Split the data into training and test sets (40% held out for testing)
    (trainingData, testData) = data.randomSplit([0.6, 0.4])

    trainingData.show(5, False)
    testData.show(5, False)

    # Fit Logistic Regression Model
    logr = LogisticRegression(featuresCol='indexedFeatures',
                              labelCol='indexedLabel')

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction",
                                   outputCol="predictedLabel",
                                   labels=labelIndexer.labels)
    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(
        stages=[labelIndexer, featureIndexer, logr, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("features", "label", "predictedLabel").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel",
        predictionCol="prediction",
        metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Test Error = {1.0 - accuracy}")

    lrModel = model.stages[2]
    trainingSummary = lrModel.summary

    # Obtain the receiver-operating characteristic as a dataframe
    # and areaUnderROC.
    trainingSummary.roc.show(5)
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy() \
        .max('F-Measure') \
        .select('max(F-Measure)') \
        .head(5)

    visualization(predictions)


if __name__ == '__main__':
    from util.file import get_absolute_file_path

    relative_path = "../data/bank.csv"
    file_path = get_absolute_file_path(relative_path)

    spark = SparkSession \
        .builder \
        .appName("Classification") \
        .master("local[*]") \
        .getOrCreate()

    main(spark, file_path)
    spark.stop()
