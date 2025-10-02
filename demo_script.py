"""
Credit Card Fraud Detection using PySpark

This script trains two logistic regression models using PySpark on a highly imbalanced 
credit card fraud dataset. One model is unweighted, the other uses class weights to address 
imbalance. It uses Spark ML Pipelines, and evaluation is done via AUC and sklearn metrics.

"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


def create_spark_session():
    """
    Creates and returns a SparkSession.
    """
    return SparkSession.builder.appName("Credit Card Fraud Detection").getOrCreate()


def load_and_prepare_data(spark):
    """
    Tries to load data from Kaggle path, then local fallback path.
    """
    kaggle_path = "/kaggle/input/creditcardfraud/creditcard.csv"
    local_path = "data/creditcard_sample.csv"  # Make sure this exists in your repo

    path = kaggle_path if os.path.exists(kaggle_path) else local_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {kaggle_path} or {local_path}")

    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn("label", col("Class").cast("integer")).drop("Class")
    return df.na.drop()


def add_class_weights(df, weight_value=20.0):
    """
    Adds a 'weight' column to handle class imbalance.

    Args:
        df (DataFrame): Spark DataFrame with a 'label' column.
        weight_value (float): Weight to assign to the minority class.

    Returns:
        DataFrame: DataFrame with additional 'weight' column.
    """
    return df.withColumn("weight", when(col("label") == 1, weight_value).otherwise(1.0))


def assemble_features(df, input_cols):
    """
    Uses VectorAssembler to combine feature columns into a single vector column.

    Args:
        df (DataFrame): Input DataFrame.
        input_cols (list): List of feature column names.

    Returns:
        DataFrame: Transformed DataFrame with 'features' column.
    """
    assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
    return assembler.transform(df).select("features", "label")


def train_models(train_data, train_weighted, feature_cols):
    """
    Trains both unweighted and class-weighted logistic regression models.

    Args:
        train_data (DataFrame): Training data without weights.
        train_weighted (DataFrame): Training data with weights.
        feature_cols (list): List of feature column names.

    Returns:
        tuple: (unweighted model, weighted model pipeline)
    """
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    lr_weighted = LogisticRegression(featuresCol="features", labelCol="label", weightCol="weight")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    pipeline = Pipeline(stages=[assembler, lr_weighted])

    model_unweighted = lr.fit(train_data)
    model_weighted = pipeline.fit(train_weighted)

    return model_unweighted, model_weighted


def evaluate_model(predictions, name="Model"):
    """
    Evaluates model performance using AUC and prints class distribution.

    Args:
        predictions (DataFrame): Model predictions with 'label' and 'prediction' columns.
        name (str): Optional name for the model.
    
    Returns:
        float: AUC score.
    """
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
    auc = evaluator.evaluate(predictions)
    print(f"{name} AUC: {auc:.4f}")
    predictions.groupBy("label", "prediction").count().show()
    return auc


def plot_confusion_matrices(y_true, y_pred, y_true_w, y_pred_w):
    """
    Plots confusion matrices side by side for both models.

    Args:
        y_true (array-like): True labels (unweighted).
        y_pred (array-like): Predicted labels (unweighted).
        y_true_w (array-like): True labels (weighted).
        y_pred_w (array-like): Predicted labels (weighted).
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_w = confusion_matrix(y_true_w, y_pred_w)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=axes[0])
    axes[0].set_title("Unweighted Model")

    ConfusionMatrixDisplay(confusion_matrix=cm_w).plot(ax=axes[1])
    axes[1].set_title("Weighted Model")

    plt.tight_layout()
    plt.show()


def main():
    """
    Main script execution function.
    """
    # Step 1: Spark session
    spark = create_spark_session()

    # Step 2: Load and prep data
    df = load_and_prepare_data(spark)

    # Step 3: Split
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # Step 4: Add weights
    class_weight = 20.0
    train_weighted = add_class_weights(train_data, class_weight)
    test_weighted = add_class_weights(test_data, class_weight)

    # Step 5: Feature engineering
    feature_cols = [col for col in df.columns if col not in ("label", "weight")]
    train_data_assembled = assemble_features(train_data, feature_cols)
    test_data_assembled = assemble_features(test_data, feature_cols)

    # Step 6: Train models
    model_unweighted, model_weighted = train_models(train_data_assembled, train_weighted, feature_cols)

    # Step 7: Evaluate
    predictions = model_unweighted.transform(test_data_assembled)
    predictions_weighted = model_weighted.transform(test_weighted)

    auc_unweighted = evaluate_model(predictions, "Unweighted")
    auc_weighted = evaluate_model(predictions_weighted, "Weighted")

    # Step 8: Sklearn metrics
    y_true = predictions.select("label").toPandas()
    y_pred = predictions.select("prediction").toPandas()
    y_true_w = predictions_weighted.select("label").toPandas()
    y_pred_w = predictions_weighted.select("prediction").toPandas()

    print("\nClassification Report - Unweighted Model")
    print(classification_report(y_true, y_pred))

    print("\nClassification Report - Weighted Model")
    print(classification_report(y_true_w, y_pred_w))

    # Step 9: Plot confusion matrices
    plot_confusion_matrices(y_true, y_pred, y_true_w, y_pred_w)


if __name__ == "__main__":
    main()
