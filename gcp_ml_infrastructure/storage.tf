resource "google_storage_bucket" "ml_data_bucket" {
  # Bucket names must be globally unique across all of Google Cloud
  name          = "${var.project_id}-kidney-disease-ml"
  location      = var.region
  
  # Set to true so you can easily run 'terraform destroy' later without errors
  force_destroy = true 
  
  uniform_bucket_level_access = true
  storage_class               = "STANDARD"

  # Cleans up failed uploads to save storage costs
  lifecycle_rule {
    condition {
      age = 1
    }
    action {
      type = "AbortIncompleteMultipartUpload"
    }
  }
}

# Folder marker for raw and processed CSVs
resource "google_storage_bucket_object" "datasets_folder" {
  name    = "datasets/"
  content = "Folder marker for raw and processed CSVs"
  bucket  = google_storage_bucket.ml_data_bucket.name
}

# Folder marker for MLflow models and plots
resource "google_storage_bucket_object" "artifacts_folder" {
  name    = "mlflow-artifacts/"
  content = "Folder marker for MLflow models and plots"
  bucket  = google_storage_bucket.ml_data_bucket.name
}
