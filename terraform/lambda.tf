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
      FRONTEND_ORIGINS              = var.frontend_origins
    }
  }

  depends_on = [aws_cloudwatch_log_group.backend]
}

# No `cors` block here on purpose: the app's own CORSMiddleware (main.py, driven by
# the same FRONTEND_ORIGINS) already sets Access-Control-Allow-Origin. Configuring
# CORS at both this layer and the app layer means every response carries two
# Access-Control-Allow-Origin headers, which browsers reject outright.
resource "aws_lambda_function_url" "backend" {
  function_name      = aws_lambda_function.backend.function_name
  authorization_type = "NONE"
}

# Function URLs with authorization_type = NONE still require an explicit resource
# policy allowing unauthenticated invocation. As of Oct 2025, AWS requires both of
# these grants - InvokeFunctionUrl alone gets a silent 403 before the function runs.
resource "aws_lambda_permission" "public_function_url" {
  statement_id           = "AllowPublicFunctionUrlInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.backend.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

# The second required grant (see comment above) needs the invoked_via_function_url
# argument on aws_lambda_permission, which isn't in the provider version pinned here
# yet (added upstream after 5.100.0). Shelling out to the same AddPermission API the
# resource would otherwise call avoids forcing a major provider-version bump just for
# this; `|| true` makes it a no-op once the statement already exists.
resource "terraform_data" "public_function_invoke" {
  triggers_replace = [aws_lambda_function.backend.function_name]

  provisioner "local-exec" {
    command = <<-EOT
      aws lambda add-permission \
        --function-name ${aws_lambda_function.backend.function_name} \
        --statement-id AllowPublicFunctionInvoke \
        --action lambda:InvokeFunction \
        --principal '*' \
        --invoked-via-function-url \
        --region ${var.aws_region} \
        || true
    EOT
  }
}
