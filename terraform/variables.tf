variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short name prefixing all resources. Must match terraform/bootstrap's project_name."
  type        = string
  default     = "alertafuego"
}

variable "image_tag" {
  description = "ECR image tag to deploy (CI passes the git SHA)."
  type        = string
  default     = "latest"
}

variable "lambda_memory_size" {
  description = "Lambda memory in MB. Also scales allotted CPU; torch inference needs headroom."
  type        = number
  default     = 3008
}

variable "lambda_timeout" {
  description = "Lambda timeout in seconds. Detection runs call Earth Engine + run inference synchronously."
  type        = number
  default     = 300
}

variable "frontend_origins" {
  description = "Comma-separated allowed CORS origins (e.g. \"https://a.example,https://b.example\"), mirrors the app's FRONTEND_ORIGINS env var. Plain string, not a list, so it can be passed via TF_VAR_frontend_origins without needing HCL/JSON quoting."
  type        = string
  default     = "http://localhost:3000"
}

variable "database_url" {
  description = "Postgres (Supabase) connection string."
  type        = string
  sensitive   = true
}

variable "detect_api_key" {
  description = "Secret required in the X-API-Key header to run detections (POST /detect/*). Without it those endpoints answer 503."
  type        = string
  sensitive   = true
}

variable "supabase_url" {
  description = "Supabase project URL (used to fetch the JWKS for auth)."
  type        = string
  sensitive   = true
}

variable "earth_engine_project_id" {
  description = "Google Earth Engine cloud project id."
  type        = string
}

variable "ee_service_account_key" {
  description = "GCP service-account JSON key with Earth Engine access. Leave empty until you've created it in GCP; the app can't call Earth Engine until this is set to a real key."
  type        = string
  sensitive   = true
  default     = ""
}

variable "frontend_url" {
  description = "Public URL of the frontend, linked from alert emails (e.g. \"https://your-frontend.example\")."
  type        = string
  default     = ""
}

variable "alert_email_from" {
  description = "Sender address for alert emails. Must be a verified identity in Amazon SES (see README). Leave empty to disable alert emails - alerts are still recorded and shown in the app."
  type        = string
  default     = ""
}

variable "telegram_bot_username" {
  description = "Username of the Telegram alerts bot, without the @ (from @BotFather). Leave empty to disable Telegram alerts."
  type        = string
  default     = ""
}

variable "telegram_bot_token" {
  description = "Telegram bot token from @BotFather."
  type        = string
  sensitive   = true
  default     = ""
}

variable "telegram_webhook_secret" {
  description = "Random string Telegram sends with every webhook call, so the backend can reject calls that aren't from Telegram."
  type        = string
  sensitive   = true
  default     = ""
}

variable "detection_schedule" {
  description = "How often the scheduled detection runs (EventBridge Scheduler expression). GOES-19 captures a full disk every 10 minutes, but Earth Engine publishes them ~40 minutes late, so 20 minutes barely delays alerts at half the runs."
  type        = string
  default     = "rate(20 minutes)"
}

variable "detection_schedule_enabled" {
  description = "Whether the scheduled detection runs."
  type        = bool
  default     = false
}

variable "detector_timeout" {
  description = "Detector Lambda timeout in seconds: under the schedule's interval, so a hung run ends before the next starts. A full-country run takes ~50 s."
  type        = number
  default     = 540
}
