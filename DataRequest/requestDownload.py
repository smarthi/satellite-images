"""
Script for downloading data (updated version of the one from sentinelhub library)
"""
from __future__ import print_function

import os
import requestTime
import requests
import json
import cv2
import numpy as np
import tifffile as tiff
from io import BytesIO

import requestThreading

REDOWNLOAD = False
THREADED_DOWNLOAD = False

REQUEST_TYPES = ['GET', 'POST']
IMAGE_FORMATS = ['tif', 'tiff', 'tiff;depth=32f', 'png', 'jpeg', 'jpg', 'jp2']

MAX_NUMBER_OF_DOWNLOAD_TRIES = 2 # Due to bad connection some requests might fail and need to be repeated
SLEEP_TIME = 5

SUCCESS_STATUS_CODE_INTERVAL = (200, 299)

VERBOSE = True

class DownloadRequest():
    def __init__(self, url=None, filename=None, headers=None, request_type='GET', post_values=None, save_response=True, return_data=True, data_type='raw', verbose=True):
        self.url = url
        self.filename = filename
        self.headers = headers
        self.request_type = request_type.upper()
        self.post_values = post_values
        self.save_response = save_response
        self.return_data = return_data
        self.data_type = data_type.lower()
        self.verbose = verbose

        self.will_download = True

        if self.request_type not in REQUEST_TYPES:
            print('Error: unknown download request type')
            raise

    def __str__(self):
        info = []
        for method in self.__dict__.keys():
            info.append(method + ': ' + str(self.__dict__[method]))
        return '\n'.join(info)

def download_data(request_list, redownload=REDOWNLOAD, threaded_download=THREADED_DOWNLOAD):
    if not isinstance(request_list, list): # in case only one request would be given
        return download_data([request_list], redownload=redownload, threaded_download=threaded_download)
    for request in request_list: # just checking that content of list is OK
        if not isinstance(request, DownloadRequest):
            new_request_list = [transform_request(r) for r in request_list]
            download_data(new_request_list, redownload=redownload, threaded_download=threaded_download)

    for request in request_list:
        request.will_download = (request.save_response or request.return_data) and redownload

    if threaded_download:
        threaded_process = requestThreading.ThreadedProcess(request_list, make_request)
        return threaded_process.get_output()
    return [make_request(request) for request in request_list]

def transform_request(request):
    if isinstance(request, DownloadRequest):
        return request
    if len(request) == 2:
        return DownloadRequest(url=request[0], filename=request[1])
    return DownloadRequest(url=request[0], filename=request[1], headers=request[2])

def make_request(request):  # request is an instance of DownloadRequest class
    if not request.will_download:
        return None
    try_num = MAX_NUMBER_OF_DOWNLOAD_TRIES
    response = None
    request.verbose=True
    while try_num > 0:
        try:
            if request.request_type == 'GET':
                response = requests.get(request.url, headers=request.headers)
            if request.request_type == 'POST':
                response = requests.post(request.url, data=json.dumps(request.post_values), headers=request.headers)
            response.raise_for_status()
            try_num = 0
            if request.verbose:
                print('Downloaded from %s' % request.url)
        except:
            try_num -= 1
            if try_num > 0:
                if request.verbose:
                    print('Unsuccessful download from %s ... will retry in %ds' % (request.url, SLEEP_TIME))
                requestTime.sleep(SLEEP_TIME)
            else:
                if request.verbose:
                    print('Failed to download from %s' % request.url)
                return None

    if request.return_data:
        return decode_data(response, request.data_type)

def decode_data(response, data_type):
    if data_type == 'json':
        return response.json()
    if data_type in IMAGE_FORMATS:
        return decode_image(response.content, data_type)

    print('Warning: unknown response data type: {}'.format(data_type))
    return response.content

def decode_image(data, image_type):
    if image_type == 'tiff' or image_type == 'tif' or image_type == 'tiff;depth=32f':
        return tiff.imread(BytesIO(data))

    img_array = np.asarray(bytearray(data), dtype="uint8")
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    if img is None:
        print('Unable to decode image')
    return img

def get_json(url, post_values=None, headers=None):
    if post_values is None:
        request_type = 'GET'
        json_headers = None
    else:
        request_type = 'POST'
        json_headers = {}
        if headers is not None:
            json_headers = headers.copy()
        json_headers['Content-Type'] = 'application/json'
    request = DownloadRequest(url=url, headers=json_headers, request_type=request_type, post_values=post_values, save_response=False, return_data=True, data_type='json', verbose=False)
    try:
        return make_request(request)
    except:
        print('Error obtaining json from {}'.format(url))
        raise
