import rasterio
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def assess_raster_accuracy(model_raster_path, truth_raster_path):
    """
    Assess the accuracy of a binary raster produced by a model compared to a ground truth raster.

    Parameters:
        model_raster_path (str): Path to the model-generated raster file.
        truth_raster_path (str): Path to the ground truth raster file.

    Returns:
        dict: A dictionary containing accuracy metrics (confusion matrix, accuracy, precision, recall, F1 score).
    """
    # Read model raster
    with rasterio.open(model_raster_path) as model_dataset:
        model_data = model_dataset.read(1)

    # Read truth raster
    with rasterio.open(truth_raster_path) as truth_dataset:
        truth_data = truth_dataset.read(1)

    # Ensure binary classification
    model_data = np.where(model_data > 0, 1, 0)
    truth_data = np.where(truth_data > 0, 1, 0)

    # Ensure the two rasters have the same dimensions
    if model_data.shape != truth_data.shape:
        raise ValueError("The dimensions of the model raster and truth raster do not match.")

    # Flatten the arrays for comparison
    model_flat = model_data.flatten()
    truth_flat = truth_data.flatten()

    # Calculate accuracy metrics
    cm = confusion_matrix(truth_flat, model_flat, labels=[0, 1])
    accuracy = accuracy_score(truth_flat, model_flat)
    precision = precision_score(truth_flat, model_flat, pos_label=1)
    recall = recall_score(truth_flat, model_flat, pos_label=1)
    f1 = f1_score(truth_flat, model_flat, pos_label=1)

    # Return metrics as a dictionary
    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

if __name__ == "__main__":
	import os
	print("Current working directory:", os.getcwd())
	model_raster = "DJI_20240807124909_0250_under_bryozoan_mask.tif"
	truth_raster = "DJI_20240807124909_0250_truth.tif"
	results = assess_raster_accuracy(model_raster, truth_raster)

	print("Confusion Matrix:")
	print(np.array(results["confusion_matrix"]))
	print(f"Accuracy: {results['accuracy']:.4f}")
	print(f"Precision: {results['precision']:.4f}")
	print(f"Recall: {results['recall']:.4f}")
	print(f"F1 Score: {results['f1_score']:.4f}")












