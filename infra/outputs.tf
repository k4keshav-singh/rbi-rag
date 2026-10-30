output "api_url" {
  description = "The frontend and API. Reachable only from allowed_http_cidrs."
  value       = "http://${aws_eip.api.public_ip}"
}

output "instance_id" {
  description = "For `aws ssm start-session --target <id>`."
  value       = aws_instance.api.id
}

output "raw_bucket" {
  description = "S3 bucket holding the downloaded PDFs."
  value       = aws_s3_bucket.raw.id
}

output "db_endpoint" {
  description = "RDS endpoint. Private: reachable only from the API's security group."
  value       = aws_db_instance.main.address
}

output "anthropic_key_parameter" {
  description = "Set the real key here; Terraform only creates the placeholder."
  value       = aws_ssm_parameter.anthropic_api_key.name
}

output "db_password_parameter" {
  description = "SSM parameter holding the generated database password."
  value       = aws_ssm_parameter.db_password.name
}

output "budget_topic_arn" {
  description = "Confirm the emailed SNS subscription or no alert is ever delivered."
  value       = aws_sns_topic.budget.arn
}

output "estimated_monthly_cost_note" {
  description = "A reminder, not a measurement."
  value = join(" ", [
    "Estimated $0-1/month inside the 12-month free tier (EC2 t3.micro 750h,",
    "RDS db.t4g.micro 750h, S3 under 5GB), rising to roughly $23/month after",
    "it lapses. These are list-price estimates, not measured spend: watch the",
    "budget alerts and Cost Explorer for the real figure.",
  ])
}
