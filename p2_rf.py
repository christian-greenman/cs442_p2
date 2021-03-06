# # import pandas as pd
# # import pyspark
# from pyspark.sql import SparkSession
#
#
# spark = (SparkSession.builder.master("local").appName("cs442_p2").getOrCreate())
# df = spark.read.csv("/home/christian/Desktop/cs442/p2/winequality-white.csv")
# df.printSchema()


# """
# Random Forest Regressor Example.
# """
# $example on$
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
# from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
# $example off$
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder
import numpy as np
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

import matplotlib.pyplot as plt
# from pyspark.mllib.classification import LogisticRegressionWithLBFGS
# from pyspark.mllib.evaluation import BinaryClassificationMetrics
# from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RandomForestRegressorExample")\
        .getOrCreate()

    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").option("delimiter",";").option("header","true").option("inferSchema","true").load("winequality-white.csv")

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.

    # print(data.head())
    # data = pd.to_numeric(data)
    featureList = []
    for col in data.columns:
        if col != 'quality':
            featureList.append(col)

    print(featureList)
    assembler = VectorAssembler(inputCols=featureList, outputCol="features")
    # featureIndexer =\
    #     VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)



    # Train a RandomForest model.
    rf = RandomForestRegressor(labelCol="quality", featuresCol="features")

    # Chain indexer and forest in a Pipeline
    # pipeline = Pipeline(stages=[featureIndexer, rf])
    pipeline = Pipeline(stages=[assembler, rf])
    # Train model.  This also runs the indexer.

    paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 50, num = 2)]) \
    .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 2)]) \
    .build()


    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator().setLabelCol('quality'),
                          numFolds=3)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.8, 0.2])

    cvModel = crossval.fit(trainingData)

    # Make predictions.
    predictions = cvModel.transform(testData)

    results = predictions.select(['prediction', 'quality'])
    predictionAndLabels=results.rdd
    metrics = MulticlassMetrics(predictionAndLabels)
    # print(metrics)
    # cm=metrics.confusionMatrix().toArray()
    # accuracy=(cm[0][0]+cm[1][1])/cm.sum()
    # precision=(cm[0][0])/(cm[0][0]+cm[1][0])
    # recall=(cm[0][0])/(cm[0][0]+cm[0][1])

    #fucking shit doesn't work
    # f1Score = metrics.fMeasure(2.0)
    # print("f1Score",f1Score)

    # round prediction values to nearest point
    pred_df = predictions.toPandas()
    pred_df['prediction'] = pred_df['prediction'].round(0)
    predictions = spark.createDataFrame(pred_df)


    # print(predictions.head())
    # for index, row in predictions.iterrows():
    #     row['quality'] = round(row['quality'], 0)
    # print(predictions)
    # evaluator = RegressionEvaluator(labelCol="quality", predictionCol="prediction", metricName="rmse").setLabelCol('quality')
    # rmse = evaluator.evaluate(predictions)
    # print(rmse)
    # rfPred = cvModel.transform(data)
    # rfResult = rfPred.toPandas()



    # manual f1 score calculations
    test_df = testData.toPandas()
    pred_df['prediction'] = pred_df['prediction'].astype(int)
    pred_df['true_pos'] = np.where(pred_df['prediction'] == test_df['quality'], 1, 0)
    pred_df['false_pos'] = np.where(pred_df['prediction'] != test_df['quality'], 1, 0)

    # print(len(test_df))
    # print(pred_df.head())
    # print(test_df.head())
    # print(pred_df['true_pos'].sum())
    # print(pred_df['false_pos'].sum())
    #
    print("f1 score: ", 2/(1 + ((pred_df['true_pos'].sum()+pred_df['false_pos'].sum())/(pred_df['true_pos'].sum()))))
    # 623/775   .80387


    f = open("result.txt", "a")
    f.write("f1 score: " + str(2/(1 + ((pred_df['true_pos'].sum()+pred_df['false_pos'].sum())/(pred_df['true_pos'].sum())))))












    # plt.plot(pred_df.prediction, test_df.quality, 'bo')
    # plt.xlabel('Quality')
    # plt.ylabel('Prediction')
    # plt.suptitle("Model Performance RMSE: %f" % rmse)
    # plt.show()

    # print(rfResult)


    # predictionAndLabels = testData.map(lambda lp: (float(cvModel.predict(lp.features)), lp.label))
    # metrics = MulticlassMetrics(predictions)
    # print(metrics.precision())
    # print(metrics.recall())
    # print(fMeasure())
    # # Select example rows to display.
    # predictions.select("prediction", "quality", "features").show(5)
    #
    # # Select (prediction, true label) and compute test error
    # evaluator = RegressionEvaluator(
    #     labelCol="label", predictionCol="prediction", metricName="rmse")
    # rmse = evaluator.evaluate(predictions)
    # print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    #
    # rfModel = model.stages[1]
    # print(rfModel)  # summary only
    # # $example off$
    #
    spark.stop()
