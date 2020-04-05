import tempfile
import urllib
from urllib.request import urlopen

import cv2
import numpy as np


def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image


def download_with_user_agent(url: str, file_location: tempfile._TemporaryFileWrapper):
    user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) "
        + "Chrome/35.0.1916.47 Safari/537.36"
    )

    # Create a request with a custom user-agent (Contivio won't let you download if it's not a "real" user-agent)
    req = urllib.request.Request(url, data=None, headers={"User-Agent": user_agent})
    f = urllib.request.urlopen(req).read()
    file_location.write(f)
    return
