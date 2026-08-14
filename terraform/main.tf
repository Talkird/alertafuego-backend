terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Filled in via `terraform init -backend-config=backend.hcl` (or
  # -backend-config=... flags in CI). Values come from terraform/bootstrap's outputs.
  backend "s3" {}
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}
