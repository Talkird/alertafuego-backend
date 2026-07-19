"""Earth Engine initialization.

One-time manual prerequisite before this works: the developer must have run
`earthengine authenticate` (or `python -c "import ee; ee.Authenticate()"`)
locally to grant this machine access to Earth Engine. That step is not
automated here.
"""

import logging
import os

import ee
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def init_earth_engine() -> None:
    """Load EARTH_ENGINE_PROJECT_ID from .env and initialize the ee client."""
    load_dotenv()
    project_id = os.getenv("EARTH_ENGINE_PROJECT_ID")
    if not project_id:
        raise RuntimeError(
            "EARTH_ENGINE_PROJECT_ID is not set in .env. "
            "Set it to your Google Earth Engine cloud project id."
        )
    ee.Initialize(project=project_id)
    logger.info("Earth Engine initialized for project %s", project_id)
