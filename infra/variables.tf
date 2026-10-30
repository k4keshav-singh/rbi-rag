variable "region" {
  description = "AWS region. ap-south-1 (Mumbai) is closest to the source data; us-east-1 is roughly 10-15% cheaper."
  type        = string
  default     = "ap-south-1"
}

variable "name" {
  description = "Name prefix for every resource."
  type        = string
  default     = "rbi-rag"
}

# --- sizing -----------------------------------------------------------------

variable "instance_type" {
  description = <<-EOT
    EC2 instance type for the API.

    t3.micro (1 GiB) is free-tier eligible for 750 h/month in the first 12
    months and fits the API with reranking OFF: measured resident memory is
    242 MB with only the embedder loaded.

    Turning reranking ON needs ~2 GiB -- measured 1,563 MB with both models --
    so use t3.small, which is NOT free-tier eligible (~$15/month).
  EOT
  type        = string
  default     = "t3.micro"
}

variable "enable_rerank" {
  description = "Run the cross-encoder reranker. Needs an instance with ~2 GiB; see instance_type."
  type        = bool
  default     = false
}

variable "db_instance_class" {
  description = "RDS instance class. db.t4g.micro is free-tier eligible for 750 h/month in the first 12 months."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage" {
  description = "RDS storage in GiB. The free tier covers 20 GiB."
  type        = number
  default     = 20
}

variable "db_engine_version" {
  description = <<-EOT
    RDS PostgreSQL version. Pinned deliberately: pgvector availability varies
    by minor version. 16.15 ships pgvector 0.8.2, against 0.8.7 in the local
    pgvector/pgvector:pg16 image -- close enough that the HNSW DDL in
    001_initial.sql is unchanged, but it is a difference worth knowing about.
  EOT
  type        = string
  default     = "16.15"
}

# --- access -----------------------------------------------------------------

variable "allowed_http_cidrs" {
  description = <<-EOT
    CIDRs allowed to reach the API. Defaults to nothing: open it to your own
    address rather than to the internet. A public demo with no auth in front
    of an LLM is a bill waiting to happen.
  EOT
  type        = list(string)
  default     = []
}

variable "allowed_ssh_cidrs" {
  description = "CIDRs allowed to SSH. Empty disables SSH ingress entirely; use SSM Session Manager instead."
  type        = list(string)
  default     = []
}

variable "ssh_key_name" {
  description = "Existing EC2 key pair name, or null for none (SSM Session Manager still works)."
  type        = string
  default     = null
}

# --- application ------------------------------------------------------------

variable "container_image" {
  description = "Image to run, e.g. <account>.dkr.ecr.<region>.amazonaws.com/rbi-rag:<tag>."
  type        = string
}

variable "anthropic_model" {
  description = "Model used for answering."
  type        = string
  default     = "claude-sonnet-5-5"
}

# --- budget -----------------------------------------------------------------

variable "budget_limit_usd" {
  description = "Monthly cost budget. Alerts fire at 50%, 80% and 100% of this."
  type        = number
  default     = 10
}

variable "budget_alert_email" {
  description = "Email for budget alerts. Confirm the SNS subscription or the alerts go nowhere."
  type        = string
}
