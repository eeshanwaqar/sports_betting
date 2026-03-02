# ──────────────────────────────────────────────────────────────
# RDS — PostgreSQL for MLflow metadata backend
# ──────────────────────────────────────────────────────────────

resource "aws_db_subnet_group" "mlflow" {
  name       = "${local.name_prefix}-mlflow-db"
  subnet_ids = aws_subnet.private[*].id

  tags = { Name = "${local.name_prefix}-mlflow-db-subnet" }
}

# ── Credentials ──────────────────────────────────────────────

resource "random_password" "db" {
  length  = 24
  special = false # Avoid URL-encoding issues in connection strings
}

resource "aws_secretsmanager_secret" "db_password" {
  name                    = "${local.name_prefix}/db-password"
  recovery_window_in_days = 0 # Immediate delete for portfolio project
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id = aws_secretsmanager_secret.db_password.id
  secret_string = jsonencode({
    username = var.db_username
    password = random_password.db.result
    host     = aws_db_instance.mlflow.address
    port     = 5432
    dbname   = var.db_name
  })
}

# ── Database Instance ────────────────────────────────────────

resource "aws_db_instance" "mlflow" {
  identifier     = "${local.name_prefix}-mlflow"
  engine         = "postgres"
  engine_version = "15.12"
  instance_class = var.db_instance_class

  allocated_storage     = 20
  max_allocated_storage = 50 # Autoscaling

  db_name  = var.db_name
  username = var.db_username
  password = random_password.db.result

  db_subnet_group_name   = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  skip_final_snapshot = true # Portfolio project
  deletion_protection = false

  backup_retention_period = 1
  multi_az                = false # Cost savings

  tags = { Name = "${local.name_prefix}-mlflow-db" }
}
