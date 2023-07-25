from mlflow.server import get_app_client
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'

tracking_uri = "http://localhost:5001/"

auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)
try:
    auth_client.create_user(username="username", password="password")
except:
    logger.info("User already exists")
auth_client.update_user_admin(username="username", is_admin=True)

