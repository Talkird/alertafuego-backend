# Holds the model checkpoint (model_best.pt, norm_stats.json). model/checkpoints/
# is gitignored, so it never reaches GitHub Actions via checkout - CI instead
# downloads it from here before building the Docker image. Upload once manually:
#
#   aws s3 cp model/checkpoints/model_best.pt s3://<bucket>/model_best.pt
#   aws s3 cp model/checkpoints/norm_stats.json s3://<bucket>/norm_stats.json

resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.project_name}-model-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket                  = aws_s3_bucket.model_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
