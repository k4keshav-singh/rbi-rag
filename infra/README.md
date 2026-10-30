# Deployment

Terraform for the free-tier shape: an EC2 instance running the API container,
RDS PostgreSQL with pgvector, S3 for the raw PDFs, least-privilege IAM, and a
cost budget with email alerts.

**Every command here is for you to run.** Nothing in this repository runs
`terraform apply`, `terraform destroy`, or any AWS CLI command that creates,
changes or deletes a resource.

---

## What this costs

| Item | First 12 months | After |
| --- | --- | --- |
| EC2 `t3.micro`, 750 h/mo | $0 (free tier) | ~$8.50 |
| RDS `db.t4g.micro` + 20 GiB, 750 h/mo | $0 (free tier) | ~$14.00 |
| S3, ~400 MB | $0 (5 GiB free) | ~$0.01 |
| SSM Parameter Store, AWS Budgets | $0 | $0 |
| Data transfer, light demo traffic | ~$0 | ~$0.50 |
| **Total** | **≈ $0–1/mo** | **≈ $23/mo** |

List-price estimates for `ap-south-1`, **not measured spend**. Watch the budget
alerts and Cost Explorer for the real figure.

Deliberately absent: no NAT gateway (~$32/mo and the database needs no egress),
no load balancer (~$18/mo; the instance serves port 80 directly behind a CIDR
allowlist), no Multi-AZ, no storage autoscaling.

> **Reranking is off by default.** Its ONNX export is 1,061 MB on disk and
> 1.32 GB resident (measured); the API needs 242 MB without it and 1,563 MB
> with it. Enabling it means `instance_type = "t3.small"`, which is **not**
> free-tier eligible (~$15/mo). Phase 4's ablation has not yet shown that
> reranking helps, so paying for it now would be buying an unmeasured
> component.

---

## Prerequisites

- Terraform >= 1.9, AWS CLI v2, Docker
- AWS credentials with permission to create VPC, EC2, RDS, S3, IAM, SSM and
  Budgets resources
- An Anthropic API key

## 1. Build and push the image

```bash
export AWS_REGION=ap-south-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/rbi-rag"
```

```bash
aws ecr create-repository --repository-name rbi-rag --region "$AWS_REGION"
```

```bash
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
```

Build for the instance's architecture. `t3.micro` is x86_64; a `t4g` instance
would need `linux/arm64`.

```bash
docker build --platform linux/amd64 -t "$ECR_REPO:v1" .
```

```bash
docker push "$ECR_REPO:v1"
```

## 2. Configure

Create `infra/terraform.tfvars` — it is git-ignored:

```hcl
container_image    = "<account>.dkr.ecr.ap-south-1.amazonaws.com/rbi-rag:v1"
budget_alert_email = "you@example.com"
budget_limit_usd   = 10

# Your own address, not 0.0.0.0/0. There is no authentication in front of an
# endpoint that spends money on LLM calls.
allowed_http_cidrs = ["203.0.113.4/32"]
```

## 3. Apply

```bash
terraform -chdir=infra init
```

```bash
terraform -chdir=infra plan -out=tfplan
```

Read the plan. Then:

```bash
terraform -chdir=infra apply tfplan
```

RDS takes roughly 10 minutes to become available.

## 4. Set the Anthropic key

Terraform creates the parameter with a placeholder and
`ignore_changes = [value]`, so the real key never enters Terraform state or a
committed file.

```bash
aws ssm put-parameter --name /rbi-rag/anthropic_api_key --type SecureString --value "sk-ant-..." --overwrite --region "$AWS_REGION"
```

Restart the container so it picks the key up:

```bash
aws ssm start-session --target "$(terraform -chdir=infra output -raw instance_id)"
```

then, in that session:

```bash
sudo systemctl restart docker && sudo docker start rbi-rag
```

## 5. Confirm the budget alert

AWS emails an SNS confirmation link. **Until you click it, no alert is ever
delivered** — a silent failure in the one safeguard that matters.

```bash
aws sns list-subscriptions-by-topic --topic-arn "$(terraform -chdir=infra output -raw budget_topic_arn)" --region "$AWS_REGION"
```

`SubscriptionArn` reading `PendingConfirmation` means it is not armed.

## 6. Load the corpus

Upload the PDFs, then build the index from the instance:

```bash
aws s3 sync data/raw "s3://$(terraform -chdir=infra output -raw raw_bucket)/raw/"
```

```bash
aws ssm start-session --target "$(terraform -chdir=infra output -raw instance_id)"
```

```bash
sudo docker exec rbi-rag python -m rbi_rag.index build
```

## 7. Check it

```bash
curl -fsS "$(terraform -chdir=infra output -raw api_url)/health"
```

`status` should be `ok`. If it is `degraded`, the `checks` object names the
dependency that is missing. Then open `api_url` in a browser.

---

## Teardown

`terraform destroy` does not remove everything, and what it leaves behind
costs money. Work through this in order.

**1. Empty the S3 bucket.** Versioning is on, so deleting objects leaves
versions behind and the bucket will refuse to be destroyed.

```bash
aws s3 rm "s3://$(terraform -chdir=infra output -raw raw_bucket)" --recursive
```

Delete the non-current versions and delete markers too:

```bash
aws s3api list-object-versions --bucket "$(terraform -chdir=infra output -raw raw_bucket)" --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' --output json > /tmp/versions.json
```

```bash
aws s3api delete-objects --bucket "$(terraform -chdir=infra output -raw raw_bucket)" --delete file:///tmp/versions.json
```

Repeat for `DeleteMarkers[]`.

**2. Destroy the stack.**

```bash
terraform -chdir=infra destroy
```

**3. Delete the RDS final snapshot.** `skip_final_snapshot = false`, so
destroy *creates* a snapshot on the way out. It is billed for storage until
deleted, and it survives the destroy silently.

```bash
aws rds describe-db-snapshots --snapshot-type manual --region "$AWS_REGION" --query 'DBSnapshots[?starts_with(DBSnapshotIdentifier, `rbi-rag`)].DBSnapshotIdentifier' --output text
```

```bash
aws rds delete-db-snapshot --db-snapshot-identifier <identifier> --region "$AWS_REGION"
```

**4. Delete the ECR images.** Not managed by Terraform; ECR storage is billed
per GB and this image is ~1.2 GB.

```bash
aws ecr delete-repository --repository-name rbi-rag --force --region "$AWS_REGION"
```

**5. Check for an orphaned Elastic IP.** Attached to a running instance it is
free; allocated and unattached it is billed hourly. Terraform releases it, but
verify if a destroy failed part-way.

```bash
aws ec2 describe-addresses --region "$AWS_REGION" --query 'Addresses[?AssociationId==null]'
```

**6. CloudWatch log groups** persist if anything created them. Check and
delete:

```bash
aws logs describe-log-groups --log-group-name-prefix /aws/rds/instance/rbi-rag --region "$AWS_REGION"
```

**7. Confirm the bill stops.** Cost Explorer lags by up to 24 hours, so check
the day after, not the same hour.

---

## Notes

- **pgvector version.** RDS PostgreSQL 16.15 ships pgvector **0.8.2**; local
  development uses the `pgvector/pgvector:pg16` image with **0.8.7**. The HNSW
  DDL in `001_initial.sql` works on both, but they are not identical builds.
- **No TLS.** The instance serves plain HTTP behind a CIDR allowlist. Putting
  a certificate in front means an ALB (~$18/mo) or CloudFront. Do not widen
  `allowed_http_cidrs` to `0.0.0.0/0` without adding both TLS and
  authentication — the endpoint spends money per request.
- **State is local.** `terraform.tfstate` holds the generated database
  password in plain text and is git-ignored. For anything shared, move to an
  S3 backend with a DynamoDB lock table.
- Validated with `terraform validate` against AWS provider ~> 5.70.
  **Never applied** — no AWS account was touched in writing this.
