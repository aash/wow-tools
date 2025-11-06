import logging
from numpy import (
    ndarray as np_ndarray,
    uint8 as np_uint8,
    copyto as np_copyto)
# import numpy as np
# import cv2 as cv
import multiprocessing.shared_memory as shm
from asyncio.streams import StreamWriter, StreamReader
from asyncio import open_connection, sleep, run
from asyncio.subprocess import create_subprocess_exec
import struct
import threading
import queue
import contextlib
import enum

SERVER_MODULE_NAME = 'overlay_server'
MAX_IMG_SZ = 1024 * 1024 * 10
SERVER_ADDRESS = '127.0.0.1:5123'
SHMEM_NAME = 'overlay_image_buffer'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger('overlay_client')


class Commands(enum.Enum):
    SAVE = 0xDEADBEEF
    STOP = 0xFEADBABE
    OK = 0xCAFFEECA

async def send_img(reader: StreamReader, writer: StreamWriter, im: np_ndarray, nme: str):
    # logger.info(f'command: 0x{Commands.SAVE.value:X}, {im.shape}, size: {im.size}')
    fmt = 'Iiii'
    assert len(im.shape) == 3
    data = struct.pack(fmt, Commands.SAVE.value, *im.shape)
    str_enc = nme.encode('utf-8')
    data += struct.pack('I', len(str_enc))
    data += str_enc
    # data += im.tobytes()
    writer.write(data)
    await writer.drain()
    d = await reader.readexactly(4)
    t = struct.unpack('I', d)
    assert t[0] == Commands.OK.value

async def send_stop(reader: StreamReader, writer: StreamWriter):
    writer.write(struct.pack('I', Commands.STOP.value))
    await writer.drain()
    d = await reader.readexactly(4)
    t = struct.unpack('I', d)
    assert t[0] == Commands.OK.value


async def overlay_client_async(command_queue: queue.Queue):
    proc = await create_subprocess_exec('uv', 'run', 'python', f'{SERVER_MODULE_NAME}.py')
    shma = shm.SharedMemory(name=SHMEM_NAME, create=True, size=MAX_IMG_SZ)
    host, port = SERVER_ADDRESS.split(':')
    port = int(port)
    reader, writer = await open_connection(host, port)
    im_buf = np_ndarray((MAX_IMG_SZ), dtype=np_uint8, buffer=shma.buf)
    
    # TODO: wait till server is ready

    logger.info('overlay async client: process started')
    while True:
        call, data = command_queue.get()
        if call == 'exit':
            break
        elif call == 'send_img':
            assert isinstance(data[0], np_ndarray)
            assert isinstance(data[1], tuple)
            tmp = data[0].flatten()
            np_copyto(im_buf[:tmp.shape[0]], tmp)
            await send_img(reader, writer, data[0], "")
        await sleep(0.005)
    await send_stop(reader, writer)
    await sleep(1)
    logger.info('overlay async client: closing the connection')
    writer.close()
    await writer.wait_closed()
    await proc.wait()
    logger.info('overlay async client: process terminated')

@contextlib.contextmanager
def overlay_client():
    command_queue = queue.Queue()
    try:
        th = threading.Thread(target=run, args=(overlay_client_async(command_queue),))
        th.start()
        def send_img(img):
            command_queue.put(['send_img', [img, img.shape]])
        yield send_img
    finally:
        command_queue.put(('exit', ()))
        th.join()
