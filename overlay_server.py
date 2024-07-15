import asyncio
import enum
from enum import Enum
import struct
import numpy as np
import cv2 as cv
from asyncio.streams import StreamWriter, StreamReader
from asyncio.exceptions import IncompleteReadError
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import sys

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
                    print('Received save command')
                    fmt = 'iiiI'
                    sz = struct.calcsize(fmt)
                    data = await reader.readexactly(sz)
                    t = struct.unpack(fmt, data)
                    print(f'Received data: {t}')
                    imsz = t[0] * t[1] * t[2]
                    print(f'img size: {imsz} bytes')
                    nme_data = await reader.readexactly(t[3])
                    name = nme_data.decode('utf-8')
                    print(f'img name: {name}')
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
                    print('Received stop command')
                    writer.write(struct.pack('I', Commands.OK.value))
                    await writer.drain()
                else:
                    print(f'Received unknown command: {t[0]}')
            except IncompleteReadError as e:
                asyncio.sleep(0.03)

        print('Closing the connection')
        writer.close()


import sys
from dataclasses import dataclass

from PySide6.QtGui import QCloseEvent, QPainter, QColor, QPen, QBrush, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QGraphicsLayout, QBoxLayout, QSizePolicy, QLabel
from PySide6.QtCore import QRect, Qt, QThread, QEvent, Signal
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QBrush, QPen, QFontMetrics, QPainterPath, QPainter

import numpy as np
from collections import defaultdict
import logging
import math, time

def millis_now():
    return int(time.time()*1000)

@dataclass
class Marker:
    marker_type: str
    geometry: tuple
    color: QColor
    data: dict
import json


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

from PySide6.QtWidgets import QMainWindow, QPlainTextEdit
from PySide6.QtCore import Qt
from qasync import QEventLoop, QApplication
# from asyncqt import QEventLoop
class MainWindow(QMainWindow):
    update_signal = Signal()
    new_marker_signal = Signal()
    new_image_signal = Signal(QImage)

    def __init__(self, stop_event: asyncio.Event):
        super().__init__()
        self.sev = stop_event
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowTransparentForInput |
                            Qt.WindowType.WindowStaysOnTopHint
                            #| Qt.WindowType.Tool
                            )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, 1920, 1200)

        self.w = QWidget(self)
        #self.w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.w.setFixedSize(1920, 1200)
        l = QGridLayout()
        l.setContentsMargins(0, 0, 0, 0)
        self.w.setLayout(l)
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
    async def create(cls, loop, stop_event):
        cc = cls(stop_event)
        loop.create_task(cc.update_timer(), name='timer')
        return cc

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.img is not None:
            painter.drawImage(0, 0, self.img)
        event.accept()
        
    async def update_timer(self):
        while not self.sev.is_set():
            millis = millis_now() - self.t0
            seconds = millis // 1000
            minutes = seconds // 60
            self.label.setText("{:02d}:{:02d}.{:03d}".format(minutes, seconds % 60, millis % 1000))
            self.update_signal.emit()
            await asyncio.sleep(0.01)

async def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    stop_event = asyncio.Event()
    app.aboutToQuit.connect(stop_event.set)
    window = await MainWindow.create(loop, stop_event)
    window.setWindowTitle('aions')
    window.show()
    print('window created')

    try:
        host, port = '127.0.0.1:5123'.split(':')
        port = int(port)
        server = await loop.create_server(lambda: asyncio.StreamReaderProtocol(asyncio.StreamReader(),
            ClientHandler(stop_event, window), loop=loop), host, port)
        with loop:
            loop.run_until_complete(stop_event.wait())
            server.close()
            await server.wait_closed()
    finally:
        pass

if __name__ == '__main__':
    asyncio.run(main())