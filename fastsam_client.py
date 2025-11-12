import socket
import struct
from multiprocessing import shared_memory
import numpy as np
from common import (
    SERVER_IP,
    SERVER_PORT,
    SHMEM_NAME,
    SHMEM_DATA_SIZE,
    OPCODE_RUN_SAM
    )
import logging
import cv2 as cv
# import select
from contextlib import contextmanager
import time
import os

@contextmanager
def client_context():
    shmem = shared_memory.SharedMemory(name=SHMEM_NAME)
    
    try:
        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_IP, SERVER_PORT))
        # Yield the socket and shared memory to the block inside the `with` statement
        yield client_socket, shmem
    finally:
        # Ensure to close resources
        shmem.close()
        client_socket.close()

def send_run_sam_request(client_socket, shmem, img: np.ndarray, img_sz: int = 1024, conf: float = 0.4, iou: float = 0.9):
    
    assert len(img.shape) == 3

    im_buf = np.ndarray((SHMEM_DATA_SIZE), dtype=np.uint8, buffer=shmem.buf)
    tmp = img.flatten()
    np.copyto(im_buf[:tmp.shape[0]], tmp)

    message = struct.pack('!Biiiiff', OPCODE_RUN_SAM, *img.shape, img_sz, conf, iou)
    client_socket.sendall(message)

    response = client_socket.recv(1024)
    msg = response.decode()
    parts = msg.split()
    if len(parts) == 2 and parts[1].isdigit():
        number = int(parts[1])
        # print(number)
    else:
        raise RuntimeError("could not process server response data (number of masks)")
    # print(f'Server response: {msg}')

    h, w, d = img.shape
    im_buf = np.ndarray((h, w*number,), dtype=np.uint8, buffer=shmem.buf)
    im = im_buf.copy()
    masks = [im[:, i * w:(i + 1) * w] for i in range(number-3)]
    os.system('del st*.png')
    logging.info(f'got {number} masks from the fastsam call')
    for i, m in enumerate(masks):
        cv.imwrite(f'st{i:02d}.png', m)
    return masks

def main():
    with client_context() as (client_socket, shm):
        img = cv.imread('tmp/1/char_img.png')
        t0 = time.time()
        for i in range(1):
            m = send_run_sam_request(client_socket, shm, img, 512, 0.2)
        dt = time.time() - t0
        print(f'time: {dt:.3f}s, {20/dt:.2f}op/s')

if __name__ == '__main__':
    main()