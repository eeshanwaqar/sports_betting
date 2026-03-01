# ──────────────────────────────────────────────────────────────
# S3 — Static website hosting for the EPL Predictor frontend
# ──────────────────────────────────────────────────────────────

resource "aws_s3_bucket" "frontend" {
  bucket        = "${var.project_name}-frontend-${local.account_id}"
  force_destroy = true # Portfolio project — easy teardown

  tags = { Name = "${local.name_prefix}-frontend" }
}

# Serve index.html as both the index and error document so that
# any path that doesn't resolve to a real S3 key (e.g. /teams)
# still returns the SPA entry point for client-side routing.
resource "aws_s3_bucket_website_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "index.html"
  }
}

# All four public-access block settings default to true on bucket creation.
# They must all be explicitly set to false for static website hosting to work.
resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

# depends_on is required: Terraform must disable the public access block
# before applying a public bucket policy, or the S3 API returns
# "BlockPublicPolicy" and the apply fails.
resource "aws_s3_bucket_policy" "frontend" {
  bucket     = aws_s3_bucket.frontend.id
  depends_on = [aws_s3_bucket_public_access_block.frontend]

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "PublicReadGetObject"
      Effect    = "Allow"
      Principal = "*"
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.frontend.arn}/*"
    }]
  })
}

# Bucket-level CORS so the browser can load assets served from this bucket.
resource "aws_s3_bucket_cors_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET"]
    allowed_origins = ["*"]
    max_age_seconds = 3000
  }
}
