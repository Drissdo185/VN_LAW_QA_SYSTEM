provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Create GKE cluster
resource "google_container_cluster" "primary" {
  name               = "${var.project_id}-gke"
  location           = var.region
  initial_node_count = 1
  
  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true

  # Network settings
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  # Enable Workload Identity for secure service account integration
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}

# Create managed node pool
resource "google_container_node_pool" "primary_nodes" {
  name       = "${google_container_cluster.primary.name}-node-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = var.gke_num_nodes

  node_config {
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",
    ]

    labels = {
      env = var.project_id
    }

    # Specify machine type for nodes
    machine_type = "n1-standard-2"
    disk_size_gb = 50
    disk_type    = "pd-standard"
    
    # Enable Workload Identity at the node level
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    tags = ["rag-system", "gke-node"]
  }
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "${var.project_id}-vpc"
  auto_create_subnetworks = "false"
}

# Subnet
resource "google_compute_subnetwork" "subnet" {
  name          = "${var.project_id}-subnet"
  region        = var.region
  network       = google_compute_network.vpc.name
  ip_cidr_range = "10.10.0.0/24"
}

# Create Service Account for the RAG application
resource "google_service_account" "rag_app_sa" {
  account_id   = "rag-app-sa"
  display_name = "RAG Application Service Account"
}

# Grant permissions
resource "google_project_iam_binding" "rag_app_sa_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/storage.objectViewer"
  ])
  
  project = var.project_id
  role    = each.key
  
  members = [
    "serviceAccount:${google_service_account.rag_app_sa.email}",
  ]
}

# Output variables for kubectl configuration
output "kubernetes_cluster_name" {
  value       = google_container_cluster.primary.name
  description = "GKE Cluster Name"
}

output "kubernetes_cluster_host" {
  value       = "https://${google_container_cluster.primary.endpoint}"
  description = "GKE Cluster Host"
  sensitive   = true
}

output "region" {
  value       = var.region
  description = "GCloud Region"
}

output "project_id" {
  value       = var.project_id
  description = "GCloud Project ID"
}