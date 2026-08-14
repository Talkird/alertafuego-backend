"""Earth Engine initialization.

Local dev prerequisite: the developer must have run `earthengine authenticate`
(or `python -c "import ee; ee.Authenticate()"`) to grant this machine access
to Earth Engine via application-default credentials.

In deployed environments (e.g. AWS Lambda) there is no interactive login, so
if EE_SERVICE_ACCOUNT_SECRET_ARN is set, a service-account key is instead
fetched from AWS Secrets Manager and used to authenticate.
"""

import json
import logging
import os

import ee
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _service_account_credentials(secret_arn: str) -> ee.ServiceAccountCredentials:
    import boto3

    secret_value = boto3.client("secretsmanager").get_secret_value(SecretId=secret_arn)
    key_data = secret_value["SecretString"]
    email = json.loads(key_data)["client_email"]
    return ee.ServiceAccountCredentials(email, key_data=key_data)


def init_earth_engine() -> None:
    """Load EARTH_ENGINE_PROJECT_ID from .env and initialize the ee client."""
    load_dotenv()
    project_id = os.getenv("EARTH_ENGINE_PROJECT_ID")
    if not project_id:
        raise RuntimeError(
            "EARTH_ENGINE_PROJECT_ID is not set in .env. "
            "Set it to your Google Earth Engine cloud project id."
        )
    secret_arn = os.getenv("EE_SERVICE_ACCOUNT_SECRET_ARN")
    if secret_arn:
        ee.Initialize(credentials=_service_account_credentials(secret_arn), project=project_id)
    else:
        ee.Initialize(project=project_id)
    logger.info("Earth Engine initialized for project %s", project_id)
