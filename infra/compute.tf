# --- IAM: least privilege ---------------------------------------------------
#
# The instance can read its two parameters and the objects in its own bucket.
# It cannot list other buckets, read other parameters, or touch RDS through
# the API -- the database is reached with a password over the network, so no
# rds:* permission is needed at all.

resource "aws_iam_role" "api" {
  name = "${var.name}-api"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

data "aws_iam_policy_document" "api" {
  statement {
    sid    = "ReadRawCorpus"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.raw.arn,
      "${aws_s3_bucket.raw.arn}/*",
    ]
  }

  statement {
    sid    = "ReadOwnParameters"
    effect = "Allow"
    actions = [
      "ssm:GetParameter",
      "ssm:GetParameters",
    ]
    # Scoped to this deployment's prefix, not ssm:* on "*".
    resources = [
      aws_ssm_parameter.db_password.arn,
      aws_ssm_parameter.anthropic_api_key.arn,
    ]
  }

  statement {
    sid       = "DecryptParameters"
    effect    = "Allow"
    actions   = ["kms:Decrypt"]
    resources = ["*"]

    # Only for the account's default SSM key, never arbitrary KMS keys.
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${var.region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "api" {
  name   = "${var.name}-api"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api.json
}

# Pull from ECR, read-only.
resource "aws_iam_role_policy_attachment" "ecr" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# Session Manager, so the instance needs no SSH key and no open port 22.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "api" {
  name = "${var.name}-api"
  role = aws_iam_role.api.name
}

# --- the instance -----------------------------------------------------------

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
}

locals {
  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region          = var.region
    name            = var.name
    image           = var.container_image
    db_host         = aws_db_instance.main.address
    db_port         = aws_db_instance.main.port
    db_name         = aws_db_instance.main.db_name
    db_user         = aws_db_instance.main.username
    db_password_arn = aws_ssm_parameter.db_password.name
    api_key_arn     = aws_ssm_parameter.anthropic_api_key.name
    anthropic_model = var.anthropic_model
    enable_rerank   = var.enable_rerank
  })
}

resource "aws_instance" "api" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.api.id]
  iam_instance_profile   = aws_iam_instance_profile.api.name
  key_name               = var.ssh_key_name

  user_data                   = local.user_data
  user_data_replace_on_change = true

  root_block_device {
    volume_size           = 20 # the free tier covers 30 GiB of gp2/gp3
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  metadata_options {
    http_tokens   = "required" # IMDSv2 only
    http_endpoint = "enabled"
  }

  tags = { Name = "${var.name}-api" }
}

# A static address, so the URL survives a stop/start. An Elastic IP attached
# to a running instance is free; one left allocated after you terminate the
# instance is billed hourly, which the teardown guide calls out.
resource "aws_eip" "api" {
  instance = aws_instance.api.id
  domain   = "vpc"

  tags = { Name = "${var.name}-api" }
}
