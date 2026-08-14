output "function_url" {
  value = aws_lambda_function_url.backend.function_url
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "model_artifacts_bucket" {
  value = aws_s3_bucket.model_artifacts.bucket
}

output "ee_service_account_secret_arn" {
  value = aws_secretsmanager_secret.ee_service_account.arn
}
