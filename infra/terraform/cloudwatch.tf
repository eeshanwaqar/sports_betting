# ──────────────────────────────────────────────────────────────
# CloudWatch — Log groups for ECS services
# ──────────────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name_prefix}/api"
  retention_in_days = 14

  tags = { Name = "${local.name_prefix}-api-logs" }
}

resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/ecs/${local.name_prefix}/mlflow"
  retention_in_days = 14

  tags = { Name = "${local.name_prefix}-mlflow-logs" }
}

resource "aws_cloudwatch_log_group" "training" {
  name              = "/ecs/${local.name_prefix}/training"
  retention_in_days = 7

  tags = { Name = "${local.name_prefix}-training-logs" }
}
