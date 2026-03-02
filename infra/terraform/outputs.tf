# ──────────────────────────────────────────────────────────────
# Outputs — Key values for CI/CD and operational use
# ──────────────────────────────────────────────────────────────

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = aws_lb.main.dns_name
}

output "api_url" {
  description = "API base URL"
  value       = "http://${aws_lb.main.dns_name}"
}

output "mlflow_url" {
  description = "MLflow UI URL (browser access)"
  value       = "http://${aws_lb.main.dns_name}/mlflow"
}

output "mlflow_tracking_uri" {
  description = "MLflow tracking URI for API clients (use this for MLFLOW_TRACKING_URI)"
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_api_url" {
  description = "ECR repository URL for API image"
  value       = aws_ecr_repository.api.repository_url
}

output "ecr_training_url" {
  description = "ECR repository URL for training image"
  value       = aws_ecr_repository.training.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_api_service_name" {
  description = "ECS API service name"
  value       = aws_ecs_service.api.name
}

output "s3_artifacts_bucket" {
  description = "S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.id
}

output "rds_endpoint" {
  description = "RDS endpoint (sensitive)"
  value       = aws_db_instance.mlflow.address
  sensitive   = true
}

output "github_actions_role_arn" {
  description = "IAM role ARN for GitHub Actions OIDC"
  value       = aws_iam_role.github_actions.arn
}

output "frontend_url" {
  description = "S3 static website endpoint (HTTP only)"
  value       = "http://${aws_s3_bucket_website_configuration.frontend.website_endpoint}"
}
