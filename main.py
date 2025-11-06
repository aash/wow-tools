from overlay_client import overlay_client
import time
import numpy as np


def main():
    with overlay_client() as ovl_clt:
        ovl_clt(np.array([[(0,0,0)]], dtype=np.uint8))
        time.sleep(10)


if __name__ == "__main__":
    main()
