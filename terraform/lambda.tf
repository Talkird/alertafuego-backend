resource "aws_cloudwatch_log_group" "backend" {
  name              = "/aws/lambda/${var.project_name}-backend"
  retention_in_days = 14
}

resource "aws_lambda_function" "backend" {
  function_name = "${var.project_name}-backend"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"

  memory_size = var.lambda_memory_size
  timeout     = var.lambda_timeout

  environment {
    variables = {
      DATABASE_URL                  = var.database_url
      SUPABASE_URL                  = var.supabase_url
      EARTH_ENGINE_PROJECT_ID       = var.earth_engine_project_id
      EE_SERVICE_ACCOUNT_SECRET_ARN = aws_secretsmanager_secret.ee_service_account.arn
      FRONTEND_ORIGINS              = join(",", var.frontend_origins)
    }
  }

  depends_on = [aws_cloudwatch_log_group.backend]
}

resource "aws_lambda_function_url" "backend" {
  function_name      = aws_lambda_function.backend.function_name
  authorization_type = "NONE"

  cors {
    allow_origins = var.frontend_origins
    allow_methods = ["*"]
    allow_headers = ["*"]
    max_age       = 300
  }
}

# Function URLs with authorization_type = NONE still require an explicit resource
# policy allowing unauthenticated invocation.
resource "aws_lambda_permission" "public_function_url" {
  statement_id           = "AllowPublicFunctionUrlInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.backend.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}
