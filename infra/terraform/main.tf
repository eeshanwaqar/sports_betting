# ──────────────────────────────────────────────────────────────
# Main — Provider, backend, and shared data sources
# ──────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  # Remote state in S3 with DynamoDB locking.
  # Bootstrap: create the S3 bucket + DynamoDB table manually first,
  # or start with a local backend and migrate after the bucket exists.
  backend "s3" {
    bucket         = "epl-predictor-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "epl-predictor-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = var.tags
  }
}

# ── Data Sources ─────────────────────────────────────────────

data "aws_caller_identity" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

# ── Locals ───────────────────────────────────────────────────

locals {
  azs         = slice(data.aws_availability_zones.available.names, 0, var.az_count)
  account_id  = data.aws_caller_identity.current.account_id
  name_prefix = "${var.project_name}-${var.environment}"
}
