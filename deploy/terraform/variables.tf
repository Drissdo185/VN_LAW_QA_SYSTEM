variable "project_id" {
  description = "The project ID to host the cluster in"
  default     = "traffic-law-thesis"
}

variable "region" {
  description = "The region the cluster in"
  default     = "asia-east1-a"
}

# variable "bucket" {
#   description = "GCS bucket for MLE project"
#   default     = "mlops-414313"
# }