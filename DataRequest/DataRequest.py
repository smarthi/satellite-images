import wmsRequest
import geopediaRequest
import requestDownload

class S2Request():
    """
    Class for managing Sentinel 2 requests using Sentinel Hub's WMS service
    """
    def __init__(self, instance_id, crs=4326, data_format=None, bbox=None, layers=None, time=None,
            width=None, height=None):

        self.instance_id = instance_id
        self.source = 'wms'
        self.data_format = data_format
        self.bbox = bbox
        self.layers = layers
        self.time = time
        self.crs = crs
        self.width = width
        self.height = height
        self.maxcc = 100.

        self.request_is_valid = False
        self.download_list = []

        self.create_wms_request()

    def create_wms_request(self):
        self.download_list = wmsRequest.get_wms_requests(self)

    def get_download_list(self):
        return self.download_list

    def get_data(self):
        data_list = requestDownload.download_data(self.download_list,redownload=True,threaded_download=True)

        return data_list

class TulipFieldRequest():
    """
    Class for managing Tulip Field requests using Geopedia

    Note: date argument is ignored at the moment
    """
    def __init__(self, bbox, width, height, layer, crs=4326):

        self.source = 'geopedia'
        self.bbox = bbox
        self.layer = layer
        self.crs = crs
        self.width = width
        self.height = height

        self.request_is_valid = False
        self.download_list = []

        self.create_geopedia_request()

    def create_geopedia_request(self):
        self.download_list = geopediaRequest.get_geopedia_request(self)

    def get_download_list(self):
        return self.download_list

    def get_data(self):
        data_list = requestDownload.download_data(self.download_list,redownload=True,threaded_download=True)

        return data_list
