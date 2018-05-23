import logging
import pyproj
import os
import http.client as http_client
import skimage.io as skio
from PIL import Image
from dateutil import parser
from urllib import parse #from urlparse import urlparse
from shapely import geometry
import requestThreading
from requestDownload import make_request, DownloadRequest
import struct
import binascii
import json
import datetime
import glob
import cv2
import numpy as np

from DataRequest import TulipFieldRequest, S2Request

http_client.HTTPConnection.debuglevel = 1
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.INFO)
requests_log.propagate = True


# Utility functions for dealing with crs and saving data
def to_epsg3857(latlong_wgs84):
    epsg3857 = pyproj.Proj(init='epsg:3857')
    wgs84 = pyproj.Proj(init='EPSG:4326')
    return pyproj.transform(wgs84,epsg3857,latlong_wgs84[1],latlong_wgs84[0])


def to_wgs84(latlong_epsg3857):
    wgs84 = pyproj.Proj(init='EPSG:4326')
    epsg3857 = pyproj.Proj(init='epsg:3857')
    res = pyproj.transform(epsg3857,wgs84,latlong_epsg3857[0],latlong_epsg3857[1])
    return [res[1], res[0]]


WMS_INSTANCE = '71513b0b-264d-494a-b8c4-c3c36433db28'
geopedia_layers = {'tulip_field_2016':'ttl1904', 'tulip_field_2017':'ttl1905', 'arable_land_2017':'ttl1917'}
s2_wms_layers = ['TRUE_COLOR', 'ALL_BANDS']


class Patch(object):
    def __init__(self, bbox, shapely_polygon, active_set, xy_index, poly_sw=None):
        self.polygon = shapely_polygon
        self.xy_index = xy_index # (i,j) coordinates of patch within polygon bounding-box
        self.set = active_set
        self.bbox = bbox
        self.p1 = to_wgs84(bbox[0])
        self.p2 = to_wgs84(bbox[1])
        self.box = (self.p1, self.p2)
        self.bounds = self.box
        self.shapely = geometry.box(self.p1[1], self.p1[0], self.p2[1], self.p2[0])
        self.contained = self.polygon.contains(self.shapely)
        self.poly_sw = poly_sw
        self.id = binascii.hexlify(struct.pack('ffff', self.p1[1], self.p1[0], self.p2[1], self.p2[0])).decode()


class GeoJsonSaver:
    """
    Class to save geo json files so they can be used in future downloads. 
    Files are stored in the specified dir, with the name 'polygon_hh_mm_ss.json,
    where hh mm and ss refer to the time when the polygon was drawn'
    """
    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            logging.info('Directory {} does not exist, creating it'.format(self.path))
            os.makedirs(self.path)

    def __call__(self, controller, action, geo_json):
        now = datetime.datetime.now()
        filename = 'polygon_{}_{}_{}.json'.format(now.hour, now.minute, now.second)
        with open(os.path.join(self.path, filename), 'w') as fp:
            json.dump(geo_json, fp)


class PolygonSlidingWindow(object):
    """
    Class to download satellite imagery from an ipyleaflet polygon
    """

    def __init__(self, box_width=2000, box_height=2000, stride_x=2000, stride_y=2000, callbacks=None):
        self.box_width = box_width
        self.box_height = box_height
        self.stride_x = stride_x
        self.stride_y = stride_y
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks
        self.active_set = None
        self.geo_jsons = []
        self.patches = {}

    def set_mode(self, set):
        self.active_set = set

    def get_active_set(self):
        return self.active_set

    def list_sets(self):
        return self.patches.keys()

    def register_callback(self, cb):
        self.callbacks.append(cb)

    def _apply_functions_on_patches_of_set(self, _set, *funs):
        for patches in self.patches.get(_set, []):
            for patch in patches:
                for fun in funs:
                    fun(patch)

    def apply_functions_on_patches(self, *funs):
        """provided function has same expected input as callbacks."""
        active_set = self.get_active_set()
        if active_set is not None:
            self._apply_functions_on_patches_of_set(active_set, *funs)
        else:
            for active_set in self.list_sets():
                self._apply_functions_on_patches_of_set(active_set, *funs)

    def add_polygon(self, geo_json_polygon_coordinates):
        active_set = self.get_active_set()
        shapely_polygon = geometry.Polygon(geo_json_polygon_coordinates)
        bounds = shapely_polygon.bounds
        bottom_left = to_epsg3857((bounds[1], bounds[0]))
        top_right = to_epsg3857((bounds[3], bounds[2]))
        starting_top_right = (bottom_left[0] + self.box_height, bottom_left[1] + self.box_width)
        bbox = [bottom_left, starting_top_right]
        patches = []
        n_cols = int(abs((bottom_left[0] - top_right[0]) / self.stride_y)) + 1
        for i in range(0, n_cols):
            temp_width = [bbox[0][1], bbox[1][1]]
            n_rows = int(abs((bottom_left[1] - top_right[1]) / self.stride_x)) + 1
            for v in range(0, n_rows):
                patch = Patch(bbox, shapely_polygon, active_set, (i,v), self)
                patches.append(patch)
                for cb in self.callbacks:
                    cb(self, patch)  # bbox: in epsg3857/tuple, box: in wgs84/shapely
                bbox = [(bbox[0][0], bbox[0][1] + self.stride_x), (bbox[1][0], bbox[1][1] + self.stride_x)]
            bbox = [(bbox[0][0] + self.stride_y, temp_width[0]), (bbox[1][0] + self.stride_y, temp_width[1])]
        if active_set not in self.patches:
            self.patches[active_set] = []
        self.patches[active_set].append(patches)
        return

    def load_polygons_from_folder(self, folder):
        """Adds all polygons described in .json files stored in the given folder"""
        geojsons = glob.glob(folder + '*.json')
        logging.debug('Found {} json files'.format(len(geojsons)))
        for idx, fn in enumerate(geojsons):
            with open(fn) as data_file:
                geo_json = json.load(data_file)
                if geo_json['geometry']['type'] != 'Polygon':
                    logging.debug('File {} is not a valid geojson')
                    continue
                self.geo_jsons.append(geo_json)
                self.add_polygon(geo_json['geometry']['coordinates'][0])

    def __call__(self, draw_controller, action, geo_json):
        if geo_json['geometry']['type'] != 'Polygon':
            return
        self.geo_jsons.append(geo_json)
        self.add_polygon(geo_json['geometry']['coordinates'][0])


class BatchDownloader(object):

    def __init__(self, root_dir, polygon_sw, requester, requester_args, requester_kwargs, callbacks_queue=None, callbacks_download=None):
        self.root_dir = root_dir
        self.requester = requester
        self.requester_args = requester_args
        self.requester_kwargs = requester_kwargs
        self.polygon_sw = polygon_sw
        if callbacks_queue is None:
            callbacks_queue = []
        self.callbacks_queue = callbacks_queue
        if callbacks_download is None:
            callbacks_download = callbacks_download
        self.callbacks_download = callbacks_download
        self.dl_list = None
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

    def _queue_contained_patches_for_download(self, patch):
        if not patch.contained:
            return
        p1, p2 = patch.p1, patch.p2
        logging.debug('Adding patch {},{} --> {},{} to queue ({})'.format(p1[0], p1[1], p2[0], p2[1], patch.bbox))
        try:
            self.requester_kwargs['bbox'] = patch.bbox
            req = self.requester(*self.requester_args, **self.requester_kwargs)
            self.dl_list.append((patch, req.download_list))
        finally:
            del self.requester_kwargs['bbox']
        logging.debug('There are {} urls queued for download'.format(len(req.download_list)))

    def collect_download_list(self):
        if self.dl_list is None:
            self.dl_list = []
            self.polygon_sw.apply_functions_on_patches(self._queue_contained_patches_for_download, *self.callbacks_queue)

    def download_list(self):
        self.collect_download_list()
        return [ (p, req) for (p, dl) in self.dl_list for req in dl ]

    def imsave(self, data, patch, request):
        bbox = patch.bbox
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        if self.requester.source == 'geopedia': # tulip data
            layer = self.requester_kwargs['layer']
            outfilename = '{}tulip_{}_geopedia_{}.png'.format(self.root_dir, patch.id, layer)
            gray = Image.fromarray(data).convert('L')
            im_bw = cv2.bitwise_not(cv2.threshold(np.asarray(gray), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
            # bw = gray.point(lambda x: 0 if x*255<128 else 255, '1')
            skio.imsave(outfilename, im_bw)
        if self.requester.source == 'wms' and self.requester_kwargs.get('layers') == 'TRUE_COLOR': # rgb image from S2
            layer = self.requester_kwargs['layers']
            logging.debug((patch, request))
            date = parser.parse(parse.parse_qs(parse.urlparse(request.url).query)['time'][0].split('/')[0])
            if date:
                skio.imsave('{}tulip_{}_wms_{}_{}.png'.format(self.root_dir, patch.id, date.strftime("%Y%m%d-%H%M%S"), layer), data)
        if self.requester.source == 'wms' and self.requester_kwargs.get('layers') == 'ALL_BANDS': # multispectral from S2
            layer = self.requester_kwargs['layers']
            print("layer {} not yet supported".format(layer))


    def _download_request_and_save(self, patch_request):
        data = make_request(patch_request[1])
        self.imsave(data, *patch_request)
        return True


    def download_data(self):
        threaded_process = requestThreading.ThreadedProcess(self.download_list(), self._download_request_and_save)
        return all(threaded_process.get_output())
