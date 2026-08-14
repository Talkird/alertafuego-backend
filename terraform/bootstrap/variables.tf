variable "aws_region" {
  description = "AWS region for the state bucket, lock table, and OIDC role."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short name prefixing all resources (state bucket, lock table, IAM role)."
  type        = string
  default     = "alertafuego"
}

variable "github_repo" {
  description = "GitHub repo allowed to assume the deploy role, as \"owner/name\"."
  type        = string
  default     = "Talkird/alertafuego-backend"
}

variable "github_branch" {
  description = "Branch allowed to assume the deploy role (workflow only runs on pushes to this branch)."
  type        = string
  default     = "main"
}
