import sys
import json
import logging
import time
import asyncio
import struct
import numpy as np
import cv2 as cv
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

from asyncio.streams import StreamWriter, StreamReader
from asyncio.exceptions import IncompleteReadError
import multiprocessing.shared_memory as shm

from qasync import QEventLoop, QApplication, asyncClose
from PySide6.QtGui import QPainter, QColor, QImage
from PySide6.QtWidgets import QMainWindow, QWidget, QGridLayout, QLabel
from PySide6.QtCore import QRect, Qt, Signal

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.WARN
)
logger = logging.getLogger('overlay_server')

OVERLAY_IMAGE_BUFFER = 'overlay_image_buffer'


class Commands(Enum):
    SAVE = 0xDEADBEEF
    STOP = 0xFEADBABE
    OK = 0xCAFFEECA


class ClientHandler:

    def __init__(self, stop_event, window):
        self.sev = stop_event
        self.win = window

    async def __call__(self, reader: StreamReader, writer: StreamWriter):
        while not self.sev.is_set():
            try:
                cmd = await reader.readexactly(4)
                t = struct.unpack('I', cmd)
                if t[0] == Commands.SAVE.value:
                    logger.info('Received save command')
                    fmt = 'iiiI'
                    sz = struct.calcsize(fmt)
                    data = await reader.readexactly(sz)
                    t = struct.unpack(fmt, data)
                    logger.info(f'Received data: {t}')
                    imsz = t[0] * t[1] * t[2]
                    logger.info(f'img size: {imsz} bytes')
                    nme_data = await reader.readexactly(t[3])
                    name = nme_data.decode('utf-8')
                    logger.info(f'img name: {name}')
                    shma = shm.SharedMemory(name=OVERLAY_IMAGE_BUFFER)
                    im_shm = np.ndarray(t[:3], dtype=np.uint8, buffer=shma.buf)
                    im = cv.cvtColor(im_shm, cv.COLOR_BGR2RGB)
                    # imdata = await reader.readexactly(imsz)
                    # print(f'img data: {len(imdata)} bytes')
                    # im = np.frombuffer(imdata, dtype=np.uint8).reshape(t[:3])
                    # cv.putText(im, name, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    h, w, d = im.shape
                    self.win.img = QImage(im.data, w, h, w*3, QImage.Format.Format_RGB888)
                    # cv.imwrite(name, im)

                    writer.write(struct.pack('I', Commands.OK.value))
                    await writer.drain()
                elif t[0] == Commands.STOP.value:
                    self.sev.set()
                    logger.info('Received stop command')
                    writer.write(struct.pack('I', Commands.OK.value))
                    await writer.drain()
                else:
                    logger.info(f'Received unknown command: {t[0]}')
            except IncompleteReadError:
                asyncio.sleep(0.03)

        logger.info('Closing the connection')
        writer.close()


def millis_now():
    return int(time.time()*1000)

@dataclass
class Marker:
    marker_type: str
    geometry: tuple
    color: QColor
    data: dict


def json_to_marker(json_string):
    data = defaultdict(lambda: None)
    data.update(json.loads(json_string).items())
    # print(data)
    if data['action'] == 'test':
        return Marker(data={"action": "test"}, marker_type='', geometry=(), color=QColor(0, 0, 0, 0))
    return Marker(
        marker_type=data['marker_type'],
        geometry=tuple(data['geometry']),
        color=QColor(*data['color']),
        data=data['data']
    )

class MainWindow(QMainWindow):
    update_signal = Signal()
    new_marker_signal = Signal()
    # new_image_signal = Signal(QImage)

    def __init__(self, stop_event: asyncio.Event):
        super().__init__()
        self.sev = stop_event
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowTransparentForInput |
                            Qt.WindowType.WindowStaysOnTopHint
                            #| Qt.WindowType.Tool
                            )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, 1920*2, 1080*2)

        self.w = QWidget(self)
        #self.w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.w.setFixedSize(1920*2, 1080*2)
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.w.setLayout(layout)
        self.setCentralWidget(self.w)

        self.w.setStyleSheet("border: 2px dashed green")

        # self.setWindowOpacity(0.75)

        self.t0 = millis_now()

        self.label = QLabel(self)
        self.label.setStyleSheet("")
        self.label.setAlignment(Qt.AlignRight)
        self.label.setStyleSheet("font-family: 'JetBrainsMono Nerd Font Mono', 'Consolas'; color: white; font-size: 20px; ")
        self.label.move(0, 0)
        # self.label.setTextMask("00:00.000")
        self.label.setText("00:00.000")
        # self.label.setOutlineThickness(10)
        self.label.setGeometry(QRect(0, 0, 100, 24))

        self.update_signal.connect(self.update)
        # self.update_timer_thread = threading.Thread(target=self.update_timer)
        # self.update_timer_thread.start()

        # self.hotkey_thread = threading.Thread(target=self.start_hotkey_listener)
        # self.hotkey_thread.start()

        self.markers = {
            # 'rect1': Marker("rectangle", (10, 10, 100, 100), QColor(255, 0, 255, 255), {"name": "rect1"}),
        }
        # self.new_marker_signal.connect(self.add_marker)
        # self.loop.create_task(self.timer())
        #threading.Timer(3, self.close).start()
        self.sev = stop_event
        self.img = None

    @classmethod
    def create(cls, loop: QEventLoop, stop_event: asyncio.Event):
        cc = cls(stop_event)
        tsk = loop.create_task(cc.update_timer(), name='timer')
        tsk.set_name('update timer task')
        return cc

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.img is not None:
            painter.drawImage(0, 0, self.img)
        event.accept()


    @asyncClose
    async def closeEvent(self, event):
        logger.info('close event')
        self.sev.set()
        return super().closeEvent(event)
        
    async def update_timer(self):
        while not self.sev.is_set():
            millis = millis_now() - self.t0
            seconds = millis // 1000
            minutes = seconds // 60
            self.label.setText("{:02d}:{:02d}.{:03d}".format(minutes, seconds % 60, millis % 1000))
            self.update_signal.emit()
            await asyncio.sleep(0.01)

# async def main():
if __name__ == '__main__':
    
    overlay_server_address = '127.0.0.1:5123'
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    loop.set_debug(True)
    asyncio.set_event_loop(loop)

    stop_event = asyncio.Event()
    app.aboutToQuit.connect(stop_event.set)
    # window = MainWindow(stop_event)

    window = MainWindow.create(loop, stop_event)
    # timer_update_task = loop.create_task(window.update_timer(), name='timer')
    window.setWindowTitle('aions')
    window.show()
    logger.info('overlay window created')
    

    # asyncio.run(app_close_event.wait(), loop_factory=QEventLoop)

    try:
        host, port = overlay_server_address.split(':')
        port = int(port)
        asyncio.run_coroutine_threadsafe(
        loop.create_server(lambda: asyncio.StreamReaderProtocol(asyncio.StreamReader(),
            ClientHandler(stop_event, window), loop=loop), host, port), loop)
        loop.run_until_complete(stop_event.wait())
            # timer_update_task.cancel()
    finally:
        # TODO fix deinitialization, event loop needs revisiting
        # server.close()
        # await server.wait_closed()
        pass

    # asyncio.run(main())