# Databricks notebook source
# MAGIC %md
# MAGIC # Predict the outcome of an encounter, given historic data
# MAGIC ##<img src="https://databricks.com/wp-content/themes/databricks/assets/images/header_logo_2x.png" alt="logo" width="150"/> 
# MAGIC In this notebook, we use RWD from simulated EHR, to predict whether a patient is at high risk of a certain condition of interest.
# MAGIC Users of the notebook can choose the condition to model (default to drug overdose), a window of time to take into account for looking into patient's history (default to 90 days)
# MAGIC and the max number of comorbidities to take into account. We then proceeed to use distributed ML with `spark` and [hyperopt](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt) to train a model while leveraging [MLFlow](https://docs.databricks.com/applications/mlflow/index.html#mlflow) to manage the end-to-end life cycle.

# COMMAND ----------

# MAGIC %md
# MAGIC ## <div style="text-align: center; line-height: 0; padding-top: 9px;"> <img src="https://dbdb.io/media/logos/delta-lake.png" width=100></div>
# MAGIC ## 0. Create a DeltaLake by calling the [Clinical Delta Lake](https://databricks.com/notebooks/00-etl-rwd.html) notebook 

# COMMAND ----------

# MAGIC %run ./00-etl-rwd

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Specify paths and parameters

# COMMAND ----------

dbutils.widgets.text('condition', 'drug overdose', 'Condition to model')
dbutils.widgets.text('num_conditions', '10', '# of comorbidities to include')
dbutils.widgets.text('num_days', '90', '# of days to use')
dbutils.widgets.text('num_days_future', '10', '# of days in future for forecasting')

# COMMAND ----------

condition=dbutils.widgets.get('condition')
num_conditions=int(dbutils.widgets.get('num_conditions'))
num_days=int(dbutils.widgets.get('num_days'))
num_days_future=int(dbutils.widgets.get('num_days_future'))

# COMMAND ----------

# DBTITLE 0,load data for training
user=dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
## Specify the path to delta tables on dbfs
delta_root_path = "dbfs:/home/{}/rwe-ehr/delta".format(user)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data preparation
# MAGIC To create the training dataset, we need to extract a dataset of qualified patients.

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2.1 Load Tables

# COMMAND ----------

patients = spark.read.format("delta").load(delta_root_path + '/patients').withColumnRenamed('Id', 'PATIENT')
encounters = spark.read.format("delta").load(delta_root_path + '/encounters').withColumnRenamed('PROVIDER', 'ORGANIZATION')
organizations = spark.read.format("delta").load(delta_root_path + '/organizations')

# COMMAND ----------

# MAGIC %md
# MAGIC Create a dataframe of all patient encounteres

# COMMAND ----------

patient_encounters = (
  encounters
  .join(patients, ['PATIENT'])
  .join(organizations, ['ORGANIZATION'])
)
display(patient_encounters.filter('REASONDESCRIPTION IS NOT NULL').limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2.2 Create a list of patients to include

# COMMAND ----------

all_patients=patient_encounters.select('PATIENT').dropDuplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Get the list of patients with the target condition (cases) and those without the condition (controls) and combine the two lists.

# COMMAND ----------

positive_patients = (
  patient_encounters
  .select('PATIENT')
  .where(lower("REASONDESCRIPTION").like("%{}%".format(condition)))
  .dropDuplicates()
  .withColumn('is_positive',lit(True))
)

negative_patients = (
  all_patients
  .join(positive_patients,on=['PATIENT'],how='left_anti')
  .limit(positive_patients.count())
  .withColumn('is_positive',lit(False))
)

patients_to_study = positive_patients.union(negative_patients)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we limit encounters to patients under study. These patients have been selected to ensure we have a balanced set of cases and controls for our model training:

# COMMAND ----------

qualified_patient_encounters_df = (
  patient_encounters
  .join(patients_to_study,on=['PATIENT'])
  .filter("DESCRIPTION is not NUll")
)
qualified_patient_encounters_df.count()

# COMMAND ----------

# DBTITLE 0,distribution of age by gender
display(qualified_patient_encounters_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3. Feature Engineering
# MAGIC Now we want to add features representing patient's encounter history to the training data. 

# COMMAND ----------

# MAGIC %md
# MAGIC First we need to identify most common comorbidities:

# COMMAND ----------

comorbid_conditions = (
  positive_patients.join(patient_encounters, ['PATIENT'])
  .where(col('REASONDESCRIPTION').isNotNull())
  .dropDuplicates(['PATIENT', 'REASONDESCRIPTION'])
  .groupBy('REASONDESCRIPTION').count()
  .orderBy('count', ascending=False)
  .limit(num_conditions)
)

display(comorbid_conditions)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we define a function that for every comborbid condition adds a binary feature indicating whether a given encounter was realted to the that condition.

# COMMAND ----------

def add_comorbidities(qualified_patient_encounters_df,comorbidity_list):
  
  output_df = qualified_patient_encounters_df
  
  idx = 0
  for comorbidity in comorbidity_list:
      output_df = (
        output_df
        .withColumn("comorbidity_%d" % idx, (output_df['REASONDESCRIPTION'].like('%' + comorbidity['REASONDESCRIPTION'] + '%')).cast('int'))
        .withColumn("comorbidity_%d"  % idx,coalesce(col("comorbidity_%d" % idx),lit(0))) # replacing null values with 0
        .cache()
      )
      idx += 1
  return(output_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we to construct comborbidity features that capturing the number of times a patient has been diagnosed with any of the comorbid conditios within a given window of time.

# COMMAND ----------

def add_recent_encounters(encounter_features):
  
  lowest_date = (
      encounter_features
      .select('START_TIME')
      .orderBy('START_TIME')
      .limit(1)
      .withColumnRenamed('START_TIME', 'EARLIEST_TIME')
    )
  
  output_df = (
       encounter_features
      .crossJoin(lowest_date)
      .withColumn("day", datediff(col('START_TIME'), col('EARLIEST_TIME')))
      .withColumn("patient_age", datediff(col('START_TIME'), col('BIRTHDATE')))
    )
  
  w = (
    Window.orderBy(output_df['day'])
    .partitionBy(output_df['PATIENT'])
    .rangeBetween(-int(num_days), -1)
  )
  
  for comorbidity_idx in range(num_conditions):
      col_name = "recent_%d" % comorbidity_idx
      
      output_df = (
        output_df
        .withColumn(col_name, sum(col("comorbidity_%d" % comorbidity_idx)).over(w))
        .withColumn(col_name,coalesce(col(col_name),lit(0)))
      )
  
  return(output_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now we need to add the target labels, indicating which encounter falls within a given window of time (in the future) that would result the patient being diagnosed with the condition of interest. 

# COMMAND ----------

def add_label(encounter_features,num_days_future):

  w = (
      Window.orderBy(encounter_features['day'])
      .partitionBy(encounter_features['PATIENT'])
      .rangeBetween(0,num_days_future)
    )
  
  output_df = (
    encounter_features
    .withColumn('label', max(col("comorbidity_0")).over(w))
    .withColumn('label',coalesce(col('label'),lit(0)))
  )
  
  return(output_df)

# COMMAND ----------

def modify_features(encounter_features):
  return(
    encounter_features
    .withColumn('START_YEAR',year(col('START_TIME')))
    .withColumn('START_MONTH',month(col('START_TIME')))
    .withColumn('ZIP',col('ZIP').cast('string'))
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have defined functions to add required features and the label, we proceed to add these features

# COMMAND ----------

comorbidity_list=comorbid_conditions.collect()
encounter_features=add_comorbidities(qualified_patient_encounters_df, comorbidity_list)
encounter_features=add_recent_encounters(encounter_features)
encounter_features=modify_features(encounter_features)
encounter_features=add_label(encounter_features,num_days_future)

# COMMAND ----------

display(encounter_features)

# COMMAND ----------

# MAGIC %md
# MAGIC Using mlflow tracking API we log notebook level parameters, which automatically also starts a new experiment run

# COMMAND ----------

mlflow.log_params({'condition':condition,'num_conditions':num_conditions,'num_days':num_days,'num_days_future':num_days_future})

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now we write these features into a feature store within Delta Lake. To ensure reproducibility, we add mlflow experiment id and the runid as a column to the the feature store. The advantage of this approach is that we receive more data, we can add new features to the featurestore that can be re-used to referred to in the future.

# COMMAND ----------

run=mlflow.active_run()
(
  encounter_features
  .withColumn('mlflow_experiment_id',lit(run.info.experiment_id))
  .withColumn('mlflow_run_id',lit(run.info.run_id))
  .write.format('delta').mode('overWrite').save(delta_root_path+'/encounter_features')
)
# mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 4. Data QC and Pre-processing 
# MAGIC Before moving to the next step, we need to make sure our dataset is not imbalanced

# COMMAND ----------

dataset_df = spark.read.format('delta').load(delta_root_path+'/encounter_features')

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's take a look at the distribution of lables 

# COMMAND ----------

display(dataset_df.groupBy('label').count())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We note that classes are imbalanced, so for downstream training we need to adjust the distribution of classes. Since we have enough data we choose to downsample the `0` class.

# COMMAND ----------

df1 = dataset_df.filter('label==1')
n_df1=df1.count()
df2 = dataset_df.filter('label==0').sample(False,0.9).limit(n_df1)
training_dataset_df = df1.union(df2).sample(False,1.0)
display(training_dataset_df.groupBy('label').count())

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder
import numpy as np

def pre_process(training_dataset_pdf):
  
  X_pdf=training_dataset_pdf.drop('label',axis=1)
  y_pdf=training_dataset_pdf['label']

  onehotencoder = OneHotEncoder(handle_unknown='ignore')
  one_hot_model = onehotencoder.fit(X_pdf.values)
  X=one_hot_model.transform(X_pdf)
  y=y_pdf.values

  return(X,y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## <div style="text-align: left; line-height: 0; padding-top: 0px;"> <img src="https://secure.meetupstatic.com/photos/event/c/1/7/6/600_472189526.jpeg" width=100> + <img src="https://i.postimg.cc/TPmffWrp/hyperopt-new.png" width=40></div>
# MAGIC ## 5. Model selection and hyperparameter tuning
# MAGIC Before training the model, we would like to find the best set of parameters (judged based on the highest score)

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow

# COMMAND ----------

encounter_cols=['ENCOUNTERCLASS','COST','PROVIDER_ZIP','START_YEAR']
patient_cols = ['RACE','GENDER','ZIP','patient_age']
comorbidty_cols = ['recent_%s'%idx for idx in range(0,num_conditions)] + ['comorbidity_%s'%idx for idx in range(0,num_conditions)]
selected_features = encounter_cols+patient_cols+comorbidty_cols

# COMMAND ----------

training_dataset_pdf = training_dataset_df.select(selected_features+['label']).toPandas().dropna()
X,y=pre_process(training_dataset_pdf)

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.model_selection import cross_val_score

# COMMAND ----------

from math import exp

def params_to_lr(params):
  return {
    'penalty':          'elasticnet',
    'multi_class':      'ovr',
    'random_state':     43,
    'n_jobs':           -1,
    'solver':           'saga',
    'tol':              exp(params['tol']), # exp() here because hyperparams are in log space
    'C':                exp(params['C']),
    'l1_ratio':         exp(params['l1_ratio'])
  }

def tune_model(params):
  with mlflow.start_run(run_name='tunning-logistic-regression',nested=True) as run:
    clf = LogisticRegression(**params_to_lr(params)).fit(X, y)
    loss = - cross_val_score(clf, X, y,n_jobs=-1, scoring='f1').min()
    return {'status': STATUS_OK, 'loss': loss}

# COMMAND ----------

from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK
search_space = {
  # use uniform over loguniform here simply to make metrics show up better in mlflow comparison, in logspace
  'tol':                  hp.uniform('tol', -3, 0),
  'C':                    hp.uniform('C', -2, 0),
  'l1_ratio':             hp.uniform('l1_ratio', -3, -1),
}

spark_trials = SparkTrials(parallelism=2)
best_params = fmin(fn=tune_model, space=search_space, algo=tpe.suggest, max_evals=32, rstate=np.random.RandomState(43), trials=spark_trials)

# COMMAND ----------

params_to_lr(best_params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## <div style="text-align: left; line-height: 0; padding-top: 0px;"> <img src="https://secure.meetupstatic.com/photos/event/c/1/7/6/600_472189526.jpeg" width=100></div>
# MAGIC ## 6. Model Training
# MAGIC Train an optimized model and log the model with mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Train and log the model

# COMMAND ----------

import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature
## since we want the model to output probabilities (risk) rather than predicted labels, we overwrite 
## mlflow.pyfun's predict method:

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]


def train(params):
  with mlflow.start_run(run_name='training-logistic-regression',nested=True) as run:
    mlflow.log_params(params_to_lr(params))
    
    X_arr=training_dataset_pdf.drop('label',axis=1).values
    y_arr=training_dataset_pdf['label'].values
  
    
    ohe = OneHotEncoder(handle_unknown='ignore')
    clf = LogisticRegression(**params_to_lr(params)).fit(X, y)
    
    pipe = Pipeline([('one-hot', ohe), ('clf', clf)])
    
    lr_model = pipe.fit(X_arr, y_arr)
    
    score=cross_val_score(clf, ohe.transform(X_arr), y_arr,n_jobs=-1, scoring='accuracy').mean()
    wrapped_lr_model = SklearnModelWrapper(lr_model)
    
    model_name= '-'.join(condition.split())    
    mlflow.log_metric('accuracy',score)
    mlflow.pyfunc.log_model(model_name, python_model=wrapped_lr_model)
  
    displayHTML('The model accuracy is: <b style="color:Tomato"> %s </b>'%(score))
    return(mlflow.active_run().info)

# COMMAND ----------

run_info=train(best_params)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 6.2 Register the model with MLflow model registry

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now you can registered the trained model in model registry (you can also use GUI for this)

# COMMAND ----------

model_name= '-'.join(condition.split())    
artifact_path = model_name
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_info.run_id, artifact_path=artifact_path)

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

model_details

# COMMAND ----------

# MAGIC %md
# MAGIC Now you can take a look at the model registry and see the V1 of the model registered. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 7. Load the model for scoring
# MAGIC Now we can use the model to asses risk on new data

# COMMAND ----------

import mlflow
best_run=mlflow.search_runs(filter_string="tags.mlflow.runName = 'training-logistic-regression'",order_by=['metrics.accuracy DESC']).iloc[0]
model_name='drug-overdose'

clf=mlflow.pyfunc.load_model(model_uri="%s/%s"%(best_run.artifact_uri,model_name))
clf_udf=mlflow.pyfunc.spark_udf(spark, model_uri="%s/%s"%(best_run.artifact_uri,model_name))

# COMMAND ----------

features_df=spark.read.format('delta').load("%s/encounter_features/"%delta_root_path).limit(100)

# COMMAND ----------

display(features_df.select('PATIENT','Enc_Id','START_TIME',clf_udf(*selected_features).alias('risk_score')))

# COMMAND ----------

clf.predict(features_df.select(selected_features).toPandas())
