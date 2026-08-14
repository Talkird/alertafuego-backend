resource "aws_secretsmanager_secret" "ee_service_account" {
  name = "${var.project_name}/ee-service-account-key"
}

resource "aws_secretsmanager_secret_version" "ee_service_account" {
  secret_id = aws_secretsmanager_secret.ee_service_account.id
  # Placeholder until ee_service_account_key is supplied - Earth Engine calls will
  # fail with an auth error until this is a real GCP service-account JSON key.
  secret_string = var.ee_service_account_key != "" ? var.ee_service_account_key : "REPLACE_ME_WITH_GCP_SERVICE_ACCOUNT_JSON_KEY"
}
