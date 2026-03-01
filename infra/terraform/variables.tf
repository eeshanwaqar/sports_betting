# ──────────────────────────────────────────────────────────────
# Variables — All configurable inputs for the EPL Predictor infra
# ──────────────────────────────────────────────────────────────

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "epl-predictor"
}

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

# ── VPC ──────────────────────────────────────────────────────

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones (minimum 2 for ALB)"
  type        = number
  default     = 2
}

# ── RDS ──────────────────────────────────────────────────────

variable "db_instance_class" {
  description = "RDS instance class (db.t3.micro is free-tier eligible)"
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "MLflow database name"
  type        = string
  default     = "mlflow"
}

variable "db_username" {
  description = "MLflow database master username"
  type        = string
  default     = "mlflow_admin"
  sensitive   = true
}

# ── ECS ──────────────────────────────────────────────────────

variable "api_desired_count" {
  description = "Number of API service tasks"
  type        = number
  default     = 1
}

variable "api_cpu" {
  description = "CPU units for API task (256 = 0.25 vCPU)"
  type        = number
  default     = 256
}

variable "api_memory" {
  description = "Memory (MiB) for API task"
  type        = number
  default     = 512
}

variable "mlflow_cpu" {
  description = "CPU units for MLflow task"
  type        = number
  default     = 256
}

variable "mlflow_memory" {
  description = "Memory (MiB) for MLflow task"
  type        = number
  default     = 1024
}

# ── Tags ─────────────────────────────────────────────────────

variable "tags" {
  description = "Common tags applied to all resources"
  type        = map(string)
  default = {
    Project   = "epl-predictor"
    ManagedBy = "terraform"
  }
}
