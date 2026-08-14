# Deploying to AWS Lambda

The backend runs as a container-image Lambda (torch + earthengine-api + numpy don't
fit in the 250MB zip-package limit), fronted by a Lambda Function URL, built and
deployed by `.github/workflows/terraform-deploy.yml` on every push to `main`.

## Architecture

- **ECR repository** holds the Docker image (FastAPI app + torch, run via the
  [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) so
  `backend/app/main.py` needed no Lambda-specific handler).
- **Lambda function** (`package_type = Image`) runs the container, exposed via a
  **Function URL** (`authorization_type = NONE` — the app enforces its own Supabase
  JWT auth per-route, same as today).
- **Secrets Manager** holds the Google service-account key Earth Engine needs
  (no interactive `earthengine authenticate` login exists inside Lambda).
- **S3 artifacts bucket** holds `model_best.pt` / `norm_stats.json`, since
  `model/checkpoints/` is gitignored and never reaches CI via checkout.
- `DATABASE_URL` / `SUPABASE_URL` / `EARTH_ENGINE_PROJECT_ID` are passed straight
  through as (encrypted-at-rest) Lambda environment variables.

## One-time setup

### 1. Bootstrap (run locally, once, with your own admin AWS credentials)

Creates the Terraform state backend (S3 + DynamoDB) and the IAM role GitHub
Actions will assume for every subsequent deploy. Nothing here is created by CI,
since CI has no credentials to create its own permissions with.

```bash
cd terraform/bootstrap
terraform init
terraform apply
```

Note the outputs — you'll need them below:
- `tfstate_bucket`
- `tflock_table`
- `github_actions_deploy_role_arn`

### 2. GitHub repo configuration

**Settings → Secrets and variables → Actions → Variables** (non-sensitive):
| Name | Value |
|---|---|
| `AWS_REGION` | e.g. `us-east-1` |
| `TFSTATE_BUCKET` | bootstrap output `tfstate_bucket` |
| `TFLOCK_TABLE` | bootstrap output `tflock_table` |
| `EARTH_ENGINE_PROJECT_ID` | your GEE cloud project id |
| `FRONTEND_ORIGINS` | JSON array, e.g. `["https://your-frontend.example"]` |

**Settings → Secrets and variables → Actions → Secrets** (sensitive):
| Name | Value |
|---|---|
| `AWS_DEPLOY_ROLE_ARN` | bootstrap output `github_actions_deploy_role_arn` |
| `DATABASE_URL` | Supabase Postgres connection string |
| `SUPABASE_URL` | Supabase project URL |
| `EE_SERVICE_ACCOUNT_KEY` | GCP service-account JSON key (see step 3) — leave unset for now if you don't have it yet; Earth Engine calls will 503 until it's added and the workflow re-run |

### 3. Google Earth Engine service account (for the Lambda runtime)

Local dev uses your personal `earthengine authenticate` login; Lambda can't do
that. Create a dedicated service account instead:

1. In your GCP project, create a service account (IAM & Admin → Service Accounts).
2. Register it for Earth Engine access: https://developers.google.com/earth-engine/guides/service_account
3. Create a JSON key for it, and paste its full contents into the `EE_SERVICE_ACCOUNT_KEY` GitHub secret.

### 4. Upload the model checkpoint

The first workflow run creates the artifacts bucket but the checkpoint download
step will fail until you've uploaded the files once:

```bash
aws s3 cp model/checkpoints/model_best.pt s3://<model_artifacts_bucket_output>/model_best.pt
aws s3 cp model/checkpoints/norm_stats.json s3://<model_artifacts_bucket_output>/norm_stats.json
```

(`model_artifacts_bucket` is a terraform output — run `terraform output` in
`terraform/` after the first apply, or check the AWS Console.)

## Everyday use

Push to `main` with changes under `backend/`, `model/`, `Dockerfile`, or
`terraform/`, and the workflow builds the image, pushes it to ECR, and applies
Terraform. Check the run's "Show function URL" step for the current endpoint.

To change the API's memory/timeout, region, or CORS origins, edit
`terraform/variables.tf` defaults or the corresponding GitHub Actions variable.

## Local Terraform usage

```bash
cd terraform
cp backend.hcl.example backend.hcl   # fill in bootstrap's outputs
terraform init -backend-config=backend.hcl
terraform plan   # requires -var values or a terraform.tfvars for database_url, supabase_url, earth_engine_project_id
```
