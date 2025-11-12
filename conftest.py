import logging
import sys
import os

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Convert milliseconds to seconds
        record.relativeCreatedInSeconds = record.relativeCreated / 1000.0
        record.levelnameChar = record.levelname[0]
        # Call the original format method
        return super().format(record)


def pytest_configure(config):
    if not os.path.exists('logs'):
        os.mkdir('logs')
    formatter = CustomFormatter('%(levelnameChar)s %(relativeCreatedInSeconds)6.2f %(name)s %(message)s')


    file_handler = logging.FileHandler("logs/tests.log", encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)


    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            file_handler,
            console_handler
        ],
        encoding='utf-8',
        
    )

    logging.info("Pytest configure...")
    
# def pytest_generate_tests(metafunc):
#     os.environ['AHK_PATH'] = 'D:/tools/AutoHotkey_2.0.12/AutoHotkey64.exe'