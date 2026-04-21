resource "google_cloud_run_v2_service" "mlflow_server" {
  name     = "mlflow-tracking-server"
  location = var.region
  project  = var.project_id

  template {
    containers {
      image = "europe-west1-docker.pkg.dev/chronic-kd/ml-repo/mlflow:latest"
      
      # Starting the server
      args = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", "8080"
      ]

      ports {
        container_port = 8080
      }
      
      # FIXES THE CRASH: Explicitly giving the container 2GB of RAM
      resources {
        limits = {
          memory = "2048Mi"
        }
      }
      
      # Mandatory: Keeps model files in GCS
      env {
        name  = "MLFLOW_DEFAULT_ARTIFACT_ROOT"
        value = "gs://${google_storage_bucket.ml_data_bucket.name}/mlflow-artifacts"
      }

      # Safely writes local metadata to the writable /tmp folder
      env {
        name  = "MLFLOW_BACKEND_STORE_URI"
        value = "sqlite:////tmp/mlflow.db"
      }
    } 
  }
}

# Allow public access
resource "google_cloud_run_service_iam_member" "public_access" {
  location = google_cloud_run_v2_service.mlflow_server.location
  project  = google_cloud_run_v2_service.mlflow_server.project
  service  = google_cloud_run_v2_service.mlflow_server.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}