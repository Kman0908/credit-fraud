import os
import logging
from datetime import datetime

LOG_FILE = f'{datetime.strftime('%Y_%m_%d_%H_%M_%S.log')}.log'

logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok = True)

LOG_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename = LOG_PATH,
    format =  '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    level = logging.INFO
)

logger = logging.getLogger(__name__)