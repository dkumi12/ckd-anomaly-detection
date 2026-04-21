output "bucket_name" {
  value       = google_storage_bucket.ml_data_bucket.name
  description = "The name of the provisioned GCS bucket to use in your Python scripts."
}

output "mlflow_tracking_uri" {
  value       = google_cloud_run_v2_service.mlflow_server.uri
  description = "Set this URL as your MLFLOW_TRACKING_URI environment variable in your ML Python scripts."
}
