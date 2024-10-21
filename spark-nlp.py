# %%
#Setup for ML and SQL functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StringIndexer, OneHotEncoder, CountVectorizer
from pyspark.ml.classification import LogisticRegression, NaiveBayes, LinearSVC
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col,isnan, when, count, trim, regexp_replace, explode, udf
from pyspark.sql.types import StringType, StructType, StructField, IntegerType
path = "/FileStore/shared_uploads/c19305451@mytudublin.ie/train_assignment.csv"

#this ensures that multiline text is not read as two texts
df = spark.read.option("multiLine", "true").csv(path, header=True)

#samples half the dataset as it took too long to run. Takes half of the dataset
df = df.sample(withReplacement=False, fraction=0.5, seed=42)

#ensuring the target is an integer instead of text
df = df.withColumn("target", df["target"].cast(IntegerType()))

#checking and displaying null counts
null_counts = df.select([count(when(col(c).isNull() | (col(c) == ""), c)).alias(c) for c in df.columns])

display(null_counts)

# %%
#this gets the number of 0's and 1's 
amounts_targets_0 = df.select("target").where(df.target == 0)
amounts_targets_1 = df.select("target").where(df.target == 1)

print(amounts_targets_0.count())
print(amounts_targets_1.count())

#undersampling 

#sets the 0 as the larger target(non-disaster) and 1 as the smaller target
larger_df = df.filter(col("target") == 0)
smaller_df = df.filter(col("target") == 1)

#gets the ratio of 0 to 1 to be able to undersample accordingly

ratio = larger_df.count()/smaller_df.count()
sampled_df = larger_df.sample(False, 1/ratio)
df_us = sampled_df.unionAll(smaller_df)

#shows the new amount of targets(0 and 1)
amounts_targets_0_s= df_us.select("target").where(df_us.target == 0)
amounts_targets_1_s  = df_us.select("target").where(df_us.target == 1)

print(amounts_targets_0_s.count())
print(amounts_targets_1_s.count())


# %%
#removing urls using a simple regex that looks for a string that starts with http/https
regex = r'https?:\/\/[^\s]+?(?=\s|$)'
df_new = df_us.withColumn('text', regexp_replace(df_us.text, regex, ''))
df_new.display()

# %%
#all the columns that need to be checked
columns = ['text', 'location', 'keyword']

#removing punctuation from location, keywords and text

for column in columns:
    df_new = df_new.withColumn(column, regexp_replace(df_new[column], "[_():';,.!?\\-]", ''))
df_new.display()

# %%
#removing hashtags and @s - set as a list to make it easier
to_replace = ["@", "#"]

for column in columns:
    for replace in to_replace:
        df_new = df_new.withColumn(column, regexp_replace(df_new[column], replace, ''))
df_new.display()

# %%

columns = ['text', 'location', 'keyword']
#removing non-alphanumeric characters with regex
for column in columns:
   df_new = df_new.withColumn(column, regexp_replace(df_new[column], "[^a-zA-Z0-9\s]", ''))
df_new.display()

# %%
#make 20s to an underscore as they represent spaces in keyword
df_no20 = df_new.withColumn('keyword', regexp_replace(df_new.keyword, "20", '_'))
df_no20.display()

# %%
#inputting placeholders
df_pl = df_no20.withColumn('keyword', when(col('keyword').isNull(), 'unknown').otherwise(col('keyword')))
df_pl = df_pl.withColumn('location', when(trim(col('location')).isNull() | (trim(col('location')) == ""), 'missing').otherwise(col('location')))

df_pl.display()


# %%
#removing stopwords from text - new stopwords added as these dont necessarily mean anything but are added
stopword_list = ["im", "amp", "like"]
stopword_list.extend(StopWordsRemover().getStopWords())
stopword_list = list(set(stopword_list))
#tokenize first

tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+", gaps=True)
df_pl = tokenizer.transform(df_pl)

remover = StopWordsRemover(inputCol="words", outputCol="words_filtered", stopWords=stopword_list)

df_pl = remover.transform(df_pl)
#df_pl.show()

#frequency encoding -location(too many for one hot encoding)

frequency_df = df_pl.groupBy("location").count().withColumnRenamed("count", "location_encoded_freq")
df_pl = df_pl.join(frequency_df, on="location", how="left")
df_pl.display()

# %%
#splitting dataset
train_df, test_df = df_pl.randomSplit(weights=[0.7,0.3], seed=42)


# %%
#there are too many values for location to be one hot encoded
location_distinctcheck = train_df.select("location").distinct()
print(location_distinctcheck.count())

keyword_distinctcheck = train_df.select("keyword").distinct()
print(keyword_distinctcheck.count())


# %%
#one hot encoding

#indexing the keywords first
keyw_index = StringIndexer(inputCol="keyword", outputCol="keyword_indexed", handleInvalid='keep')
keyindexmodel = keyw_index.fit(train_df)

indexed_df = keyindexmodel.transform(train_df)
indexed_df.display()

#applying one hot encoding
keyw_encoder = OneHotEncoder(inputCol="keyword_indexed", outputCol="keyword_encoded")
keyw_encoded_df = keyw_encoder.fit(indexed_df).transform(indexed_df)
keyw_encoded_df.display()

# %%
#word cloud

#this counts each word that was tokenised before and adds them to a new column
exploded_df = keyw_encoded_df.withColumn("words_filtered", explode(col("words_filtered")))
exploded_df.display()

word_freq = exploded_df.groupBy("words_filtered").count()

#only showing the top 1000 words

list_freq = word_freq.orderBy(col("count").desc()).head(1000)
word_freq_list = spark.createDataFrame(list_freq)
word_freq_list.display()

# %%
#count vectorizer
countVect = CountVectorizer(inputCol="words_filtered", outputCol="features_cv")
countVectModel = countVect.fit(keyw_encoded_df)
data = countVectModel.transform(keyw_encoded_df)
data.display()

# %%
#Setting up the features and targets

#setting up the encoded location and making it so it can be added to the pipeline.
assembler_loc = VectorAssembler(inputCols=["location_encoded_freq"], outputCol = "location_encoded")
data = assembler_loc.transform(data)

#these are the features now - added to one column for ease of use
input = ["location_encoded", "keyword_encoded", "features_cv"]

vecAssembler = VectorAssembler(inputCols=input, outputCol="features")

data_ml = vecAssembler.transform(data)
data_ml.display()


# %%
#logistic regression

lr = LogisticRegression(featuresCol="features", labelCol="target", regParam=1.0)

# %%
#pipeline
pipeline_lr = Pipeline(stages=[keyw_index, keyw_encoder, countVect, assembler_loc, vecAssembler, lr])

pipelinemodel = pipeline_lr.fit(train_df)

train_predictions = pipelinemodel.transform(train_df)
test_pred = pipelinemodel.transform(test_df)

# %%
test_pred.select("features", "target", "prediction", "probability").show()

# %%
f1Evaluate = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")
binaryEvaluate = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol="target")
accuracyEvaluate = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="target")
print(f"Accuracy: {accuracyEvaluate.evaluate(test_pred)}")
print(f"AUC: {binaryEvaluate.evaluate(test_pred)}")
print(f"F1 score {f1Evaluate.evaluate(test_pred)}")


# %%
#cross validation to choose the best model - building a grid to see which hyperparameters give the best model

paramGrid_lr = ParamGridBuilder().addGrid(lr.regParam, [0.5, 1.0, 1.5]).addGrid(lr.elasticNetParam, [0.0, 0.3, 0.5, 1.0]).build()
validator_lr = CrossValidator(estimator=pipeline_lr, estimatorParamMaps=paramGrid_lr, evaluator=binaryEvaluate, numFolds=3, parallelism=2)

# %%
#training the best model
train_df.cache() #caches the train data to speed up the process
crossModel = validator_lr.fit(train_df)
train_df.unpersist()

# %%
#tests best model
cv_pred = crossModel.transform(test_df)
cv_pred.select("features", "target", "prediction", "probability").display()


# %%
#shows accuracy, AUC and f1 score of best model
bestModel = crossModel.bestModel

lrModel = bestModel.stages[-1]
print(lrModel.getRegParam())
print(lrModel.getElasticNetParam())

print(f"Accuracy: {accuracyEvaluate.evaluate(cv_pred)}")
print(f"AUC: {binaryEvaluate.evaluate(cv_pred)}")
print(f"F1 score: {f1Evaluate.evaluate(cv_pred)}")

# %%
#naive bayes
nb = NaiveBayes(featuresCol="features", modelType="complement", labelCol="target")

pipeline_nb = Pipeline(stages=[keyw_index, keyw_encoder, countVect, assembler_loc, vecAssembler, nb])

pipelinemodelnb = pipeline_nb.fit(train_df)

train_predictions_nb = pipelinemodelnb.transform(train_df)
test_pred_nb = pipelinemodelnb.transform(test_df)

test_pred_nb.select("features", "target", "prediction", "probability").show()

# %%
print(f"Accuracy: {accuracyEvaluate.evaluate(test_pred_nb)}")
print(f"AUC: {binaryEvaluate.evaluate(test_pred_nb)}")
print(f"F1 score : {f1Evaluate.evaluate(test_pred_nb)}")

# %%
#cross validation
paramGrid_nb = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.3, 0.5, 1.0, 2.0]).build()
validator_nb = CrossValidator(estimator=pipeline_nb, estimatorParamMaps=paramGrid_nb, evaluator=binaryEvaluate, numFolds=3, parallelism=2)

# %%
train_df.cache()
crossModel_nb = validator_nb.fit(train_df)
train_df.unpersist()

# %%
cv_pred_nb = crossModel_nb.transform(test_df)
cv_pred_nb.select("features", "target", "prediction", "probability").display()

# %%
bestModel_nb = crossModel_nb.bestModel

nbModel = bestModel_nb.stages[-1]
print(nbModel.getSmoothing())
print(f"Accuracy: {accuracyEvaluate.evaluate(cv_pred_nb)}")
print(f"AUC: {binaryEvaluate.evaluate(cv_pred_nb)}")
print(f"F1 score : {f1Evaluate.evaluate(cv_pred_nb)}")

# %%
#SVM - Linear SVC
svc = LinearSVC(featuresCol="features", labelCol="target", maxIter=10, regParam=0.1)

pipeline_svc = Pipeline(stages=[keyw_index, keyw_encoder, countVect, assembler_loc, vecAssembler, svc])

pipelinemodelsvc = pipeline_svc.fit(train_df)

train_predictions_svc = pipelinemodelsvc.transform(train_df)
test_pred_svc = pipelinemodelsvc.transform(test_df)

test_pred_svc.select("features", "target", "prediction").show()

# %%
print(f"Accuracy: {accuracyEvaluate.evaluate(test_pred_svc)}")
print(f"AUC: {binaryEvaluate.evaluate(test_pred_svc)}")
print(f"f1 score: {f1Evaluate.evaluate(test_pred_svc)}")


# %%
paramGrid_svc = ParamGridBuilder().addGrid(svc.regParam, [0.0, 0.3, 0.5, 1.0]).addGrid(svc.fitIntercept, [True, False]).build()
validator_svc = CrossValidator(estimator=pipeline_svc, estimatorParamMaps=paramGrid_svc, evaluator=binaryEvaluate, numFolds=3, parallelism=2)

# %%
train_df.cache()
crossModel_svc = validator_svc.fit(train_df)
train_df.unpersist()
cv_pred_svc = crossModel_svc.transform(test_df)
cv_pred_svc.select("features", "target", "prediction").display()

# %%
bestModel_svc = crossModel_svc.bestModel

svcModel = bestModel_svc.stages[-1]
print(svcModel.getRegParam())
print(svcModel.getFitIntercept())
print(f"Accuracy: {accuracyEvaluate.evaluate(cv_pred_svc)}")
print(f"AUC: {binaryEvaluate.evaluate(cv_pred_svc)}")
print(f"f1 score: {f1Evaluate.evaluate(cv_pred_svc)}")

# %%
#doing the same pre-processing to the kaggle test data

path_kaggle = "/FileStore/shared_uploads/c19305451@mytudublin.ie/test.csv"
test_kg = spark.read.option("multiLine", "true").csv(path_kaggle, header=True)


#removing urls
regex = r'https?:\/\/[^\s]+?(?=\s|$)'
test_kg = test_kg.withColumn('text', regexp_replace(test_kg.text, regex, ''))
test_kg.display()

columns = ['text', 'location', 'keyword']

#removing punctuation from location, keywords and text

for column in columns:
    test_kg = test_kg.withColumn(column, regexp_replace(test_kg[column], "[_():';,.!?\\-]", ''))
test_kg.display()

#removing hashtags and @s
to_replace = ["@", "#"]
for column in columns:
    for replace in to_replace:
        test_kg = test_kg.withColumn(column, regexp_replace(test_kg[column], replace, ''))
test_kg.display()

columns = ['text', 'location', 'keyword']

for column in columns:
   test_kg = test_kg.withColumn(column, regexp_replace(test_kg[column], "[^a-zA-Z0-9\s]", ''))
test_kg.display()

#make 20s to an underscore as they seem to represent spaces in keyword
kg_no20 = test_kg.withColumn('keyword', regexp_replace(test_kg.keyword, "20", '_'))
kg_no20.display()

#inputting placeholders
kg_pl = kg_no20.withColumn('keyword', when(col('keyword').isNull(), 'unknown').otherwise(col('keyword')))
kg_pl = kg_pl.withColumn('location', when(trim(col('location')).isNull() | (trim(col('location')) == ""), 'missing').otherwise(col('location')))
#df_new = df_new.withColumn('location', when(col('location').isNull(), 'missing').otherwise(col('location')))
#df_new = df_new.withColumn('keyword', when(col('keyword') == "", 'unknown').otherwise(col('keyword')))
#df_new = df_new.withColumn('location', when(col('location') == "", 'missing').otherwise(col('location')))
df_pl.display()




#removing stopwords from text
stopword_list = ["like", "im", "amp"]
stopword_list.extend(StopWordsRemover().getStopWords())
stopword_list = list(set(stopword_list))
#tokenize first

tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+", gaps=True)
kg_pl = tokenizer.transform(kg_pl)

remover = StopWordsRemover(inputCol="words", outputCol="words_filtered", stopWords=stopword_list)

kg_pl = remover.transform(kg_pl)
#df_pl.show()

#frequency encoding -location(too many for one hot encoding)

frequency_kg = kg_pl.groupBy("location").count().withColumnRenamed("count", "location_encoded_freq")
kg_pl = kg_pl.join(frequency_kg, on="location", how="left")
kg_pl.display()


# %%
#preparing the best model on the cleaned kaggle dataset and displaying it so it can be downloaded.
kg_pred = bestModel_nb.transform(kg_pl)
kg_predictions = kg_pred.withColumnRenamed("prediction", "target")
kg_predictions.select("id", "target").display()


