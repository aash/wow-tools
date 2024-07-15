import logging
import sys
import os

def pytest_configure(config):
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/tests.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        encoding='utf-8'
    )
    logging.info("Pytest configure...")

# def pytest_generate_tests(metafunc):
#     os.environ['AHK_PATH'] = 'D:/tools/AutoHotkey_2.0.12/AutoHotkey64.exe'