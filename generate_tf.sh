#!/bin/bash

# Define the directory name
DIR_NAME="gcp_ml_infrastructure"

# Create the directory and navigate into it
mkdir -p "$DIR_NAME"
cd "$DIR_NAME" || exit

echo "Creating Terraform files in ./$DIR_NAME..."

# 1. Create providers.tf
# Note: Using 'EOF' in quotes prevents Bash from trying to evaluate Terraform's syntax as Bash variables.
cat << 'EOF' > providers.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
EOF
echo " - Created providers.tf"

# 2. Create variables.tf
cat << 'EOF' > variables.tf
variable "project_id" {
  description = "Your unique GCP Project ID"
  type        = string
}

variable "region" {
  description = "The GCP region for the resources"
  type        = string
  default     = "europe-west1" 
}
EOF
echo " - Created variables.tf"

# 3. Create storage.tf
cat << 'EOF' > storage.tf
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
EOF
echo " - Created storage.tf"

# 4. Create outputs.tf
cat << 'EOF' > outputs.tf
output "bucket_name" {
  value       = google_storage_bucket.ml_data_bucket.name
  description = "The name of the provisioned GCS bucket to use in your Python scripts."
}
EOF
echo " - Created outputs.tf"

# 5. Create a terraform.tfvars.example file for easy setup
cat << 'EOF' > terraform.tfvars.example
# Rename this file to terraform.tfvars and add your actual GCP Project ID
project_id = "YOUR_GCP_PROJECT_ID_HERE"
# region = "europe-west1" # Uncomment to override the default region
EOF
echo " - Created terraform.tfvars.example"

echo ""
echo "✅ Success! Your Terraform infrastructure code is ready."
echo "To deploy, run the following commands:"
echo "  cd $DIR_NAME"
echo "  cp terraform.tfvars.example terraform.tfvars"
echo "  (Edit terraform.tfvars with your actual project ID)"
echo "  terraform init"
echo "  terraform plan"