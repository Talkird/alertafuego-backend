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
