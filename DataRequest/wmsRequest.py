"""
Script for creating requests for WMS service
http://services.sentinel-hub.com/v1/wms
"""

import requestTime
import indexService
import requestDownload

BASE_URL = 'http://services.sentinel-hub.com/v1/'

TRUE_BAND = 'TRUE_COLOR'
ALL_BANDS = 'ALL_BANDS' # important - this is only for eo-research instance
BAND_LIST = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']

# Possible formats: jpeg, png, tiff, jp2

TIFF_FORMAT = 'tiff;depth=32f'
PNG_FORMAT = 'png'

DEFAULT_IMG_SIZE = (512, 512)
START_DATE = '2015-01-01'
MAX_CLOUD_COVERAGE = 1.0
WGS84 = 4326

def get_wms_requests(request):

    bbox = get_bbox(request.bbox)

    if bbox is None:
        print('invalid bbox')
        return None

    bands = get_bands(request.layers)
    if bands is None:
        print('invalid layers')
        return None

    if isinstance(bands, list):
        bands = ','.join(bands)

    img_format = get_img_format(request.data_format, bands)

    maxcc = get_maxcc(request.maxcc)
    crs = get_crs(request.crs)

    if request.width and request.height:
        img_size = (int(request.width), int(request.height))

    else:
        print('request width or height is missing')
        return None


    dates = get_dates(bbox, request.time, maxcc=maxcc, crs=crs)

    bbox = ','.join(map(str, bbox))
    download_list = []

    for date in dates:
        url = get_wms_url(request.instance_id, bbox, date, bands, img_format, maxcc, crs, img_size)
        download_list.append(requestDownload.DownloadRequest(url=url, return_data=True, data_type=img_format, verbose=False))
    return download_list


def get_wms_url(instance_id, bbox, date, bands, img_format, maxcc=MAX_CLOUD_COVERAGE, crs=WGS84, img_size=DEFAULT_IMG_SIZE):
    url = BASE_URL + 'wms/' + instance_id + '?showLogo=false&service=WMS&request=GetMap' \
          + '&layers=' + bands + '&format=image/' + img_format + '&version=1.3.0&height=' + str(img_size[1]) \
          + '&width=' + str(img_size[0]) + '&crs=EPSG:' + str(crs) + '&maxcc=' + str(int(100 * maxcc)) \
          + '&time=' + date + '/' + date + '&bbox=' + bbox
    if img_format == TIFF_FORMAT:
        url += '&styles=REFLECTANCE'
    return url

def get_dates(bbox, time, maxcc=MAX_CLOUD_COVERAGE, crs=WGS84):
    if time:
        if isinstance(time, str):
            if len(time.split('T')) == 1:
                date_interval = (time + 'T00:00:00', time + 'T23:59:59')
            else:
                date_interval = (time, time)
        else:
            start_time, end_time = time
            if len(start_time.split('T')) == 1:
                start_time += 'T00:00:00'
            if len(end_time.split('T')) == 1:
                end_time += 'T23:59:59'
            date_interval = (start_time, end_time)
    else:
        date_interval = (START_DATE, time.get_current_date())

    return indexService.get_area_dates(bbox, date_interval, maxcc, crs)

def get_bbox(data):
    if data is None:
        print('wms request must have bbox')
        return None
    if isinstance(data, str):
        data = data.split(',')
    if isinstance(data, list):
        if len(data) == 2 and (isinstance(data[0], list) or isinstance(data[0], tuple)) and (isinstance(data[0], list) or isinstance(data[0], tuple)):
            data = [coord for point in data for coord in point]
        if len(data) == 4:
            return list(map(float, data))
    return None

def get_bands(layers):
    '''if layers is None:
        return ','.join(BAND_LIST)'''
    if isinstance(layers, str) and ',' not in layers:
        return layers
    '''if isinstance(layers, list):
        return ','.join(layers)'''
    return None

def get_maxcc(data):
    if data:
        if isinstance(data, str):
            data = float(data)
        if data > 1: # we assume we have percents
            return data / 100
        return data
    else:
        return MAX_CLOUD_COVERAGE

def get_crs(data):
    if data:
        return int(data)
    else:
        return WGS84

def get_img_format(img_format, bands):
    if img_format:
        if img_format == 'tif' or img_format == 'tiff':
            return TIFF_FORMAT

        return img_format

    if bands == TRUE_BAND:
        return PNG_FORMAT

    return TIFF_FORMAT
