variable "project_id" {
  description = "Your unique GCP Project ID"
  type        = string
}

variable "region" {
  description = "The GCP region for the resources"
  type        = string
  default     = "europe-west1" 
}
