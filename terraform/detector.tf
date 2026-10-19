# Scheduled detection: a second function from the same image runs the detection over
# Argentina every few minutes, so the API function serving the app never runs a
# ~50 s detection or loads the model for one. EventBridge Scheduler invokes it; the
# Lambda Web Adapter forwards that (non-HTTP) event as a POST to
# AWS_LWA_PASS_THROUGH_PATH, a route only this function mounts (DETECTION_WORKER).
#
# No reserved concurrency: this account's limit is 10 concurrent executions, and AWS
# keeps at least 10 unreserved. Overlapping runs are prevented in the database
# instead (backend/app/routers/scheduled.py).

resource "aws_cloudwatch_log_group" "detector" {
  name              = "/aws/lambda/${var.project_name}-detector"
  retention_in_days = 14
}

resource "aws_lambda_function" "detector" {
  function_name = "${var.project_name}-detector"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"

  memory_size = var.lambda_memory_size
  # Under the schedule's interval, so a run that hangs is cut off before the next.
  timeout = var.detector_timeout

  environment {
    variables = merge(local.app_environment, {
      DETECTION_WORKER          = "1"
      AWS_LWA_PASS_THROUGH_PATH = "/internal/scheduled-detection"
    })
  }

  depends_on = [aws_cloudwatch_log_group.detector]
}

data "aws_iam_policy_document" "scheduler_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "detection_scheduler" {
  name               = "${var.project_name}-detection-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_trust.json
}

data "aws_iam_policy_document" "invoke_detector" {
  statement {
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.detector.arn]
  }
}

resource "aws_iam_role_policy" "detection_scheduler" {
  name   = "${var.project_name}-invoke-detector"
  role   = aws_iam_role.detection_scheduler.id
  policy = data.aws_iam_policy_document.invoke_detector.json
}

resource "aws_scheduler_schedule" "detection" {
  name                = "${var.project_name}-detection"
  schedule_expression = var.detection_schedule
  state               = var.detection_schedule_enabled ? "ENABLED" : "DISABLED"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_lambda_function.detector.arn
    role_arn = aws_iam_role.detection_scheduler.arn

    # A missed or failed run isn't retried: the next one, minutes later, picks up
    # the latest image anyway, and retries would only stack runs.
    retry_policy {
      maximum_retry_attempts       = 0
      maximum_event_age_in_seconds = 600
    }
  }
}
