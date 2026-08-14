output "tfstate_bucket" {
  value = aws_s3_bucket.tfstate.bucket
}

output "tflock_table" {
  value = aws_dynamodb_table.tflock.name
}

output "github_actions_deploy_role_arn" {
  description = "Set this as the AWS_DEPLOY_ROLE_ARN GitHub Actions secret."
  value       = aws_iam_role.github_actions_deploy.arn
}
