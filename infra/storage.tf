# --- S3: the raw PDF store --------------------------------------------------

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "raw" {
  bucket = "${var.name}-raw-${random_id.suffix.hex}"

  # Deliberately not force_destroy: a bucket with objects in it will refuse to
  # be destroyed, which is the right default for the only copy of a corpus
  # that took hours of manual downloading to assemble. The teardown guide
  # says how to empty it on purpose.
  tags = { Name = "${var.name}-raw" }
}

resource "aws_s3_bucket_public_access_block" "raw" {
  bucket = aws_s3_bucket.raw.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "raw" {
  bucket = aws_s3_bucket.raw.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "raw" {
  bucket = aws_s3_bucket.raw.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# --- RDS: Postgres with pgvector --------------------------------------------

resource "random_password" "db" {
  length  = 32
  special = false # keeps the password safe to paste into a DSN unescaped
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.name}-db"
  subnet_ids = aws_subnet.private[*].id

  tags = { Name = "${var.name}-db" }
}

resource "aws_db_instance" "main" {
  identifier     = "${var.name}-db"
  engine         = "postgres"
  engine_version = var.db_engine_version
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = 0 # no autoscaling: storage growth is a silent bill
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "rbi_rag"
  username = "rbi"
  password = random_password.db.result
  port     = 5432

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false
  multi_az               = false # single-AZ: this is a demo, not production

  backup_retention_period   = 7
  skip_final_snapshot       = false
  final_snapshot_identifier = "${var.name}-final-${random_id.suffix.hex}"
  deletion_protection       = false

  auto_minor_version_upgrade = true
  apply_immediately          = true

  tags = { Name = "${var.name}-db" }
}

# --- parameters the instance reads at boot ----------------------------------
#
# SSM Parameter Store SecureString rather than Secrets Manager: functionally
# equivalent here and free, against $0.40 per secret per month.

resource "aws_ssm_parameter" "db_password" {
  name        = "/${var.name}/postgres_password"
  description = "RDS master password"
  type        = "SecureString"
  value       = random_password.db.result

  tags = { Name = "${var.name}-db-password" }
}

resource "aws_ssm_parameter" "anthropic_api_key" {
  name        = "/${var.name}/anthropic_api_key"
  description = "Anthropic API key. Terraform creates it empty; set the real value out of band."
  type        = "SecureString"
  value       = "replace-me"

  # The key never belongs in Terraform state or in a committed tfvars file.
  # Set it with `aws ssm put-parameter --overwrite`; see infra/README.md.
  lifecycle {
    ignore_changes = [value]
  }

  tags = { Name = "${var.name}-anthropic-key" }
}
