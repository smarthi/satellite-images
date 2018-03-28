import wmsRequest
import geopediaRequest
import requestDownload

class Request():
    def get_download_list(self):
        return self.download_list

    def get_data(self):
        data_list = requestDownload.download_data(self.download_list,redownload=True,threaded_download=True)
        return data_list


class S2Request(Request):
    """
    Class for managing Sentinel 2 requests using Sentinel Hub's WMS service
    """
    source = 'wms'

    def __init__(self, instance_id, crs=4326, data_format=None, bbox=None, layers=None, time=None,
            width=None, height=None, maxcc=100.):

        self.instance_id = instance_id
        self.data_format = data_format
        self.bbox = bbox
        self.layers = layers
        self.time = time
        self.crs = crs
        self.width = width
        self.height = height
        self.maxcc = maxcc
        self.request_is_valid = False
        self.download_list = wmsRequest.get_wms_requests(self)


class TulipFieldRequest(Request):
    """
    Class for managing Tulip Field requests using Geopedia

    Note: date argument is ignored at the moment
    """
    source = 'geopedia'

    def __init__(self, bbox, width, height, layer, crs=4326):
        self.bbox = bbox
        self.layer = layer
        self.crs = crs
        self.width = width
        self.height = height
        self.request_is_valid = False
        self.download_list = geopediaRequest.get_geopedia_request(self)
