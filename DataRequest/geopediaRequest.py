import requestDownload

BASE_URL = 'http://service.geopedia.world/wms/ml_aws?service=WMS&request=GetMap&styles=&format=image%2Fpng&transparent=false&version=1.1.1'
WGS84 = 3857
PNG_FORMAT = 'png'
DEFAULT_IMG_SIZE = (512, 512)

def get_geopedia_request(request):

    bbox = get_bbox(request.bbox)

    if bbox is None:
        print('invalid bbox')
        return None

    img_format = PNG_FORMAT

    crs = get_crs(request.crs)

    if request.width and request.height:
        img_size = (int(request.width), int(request.height))
    else:
        print('request width or height is missing')
        return None

    download_list = []

    bbox = ','.join(map(str, bbox))
    url = get_geopedia_url(bbox, request.layer, crs=crs, img_size=img_size)
    download_list.append(requestDownload.DownloadRequest(url=url, return_data=True, data_type=img_format, verbose=False))

    return download_list


def get_geopedia_url(bbox, layer, crs, img_size=DEFAULT_IMG_SIZE):

    url = BASE_URL+ '&layers='+str(layer) + '&height=' + str(img_size[1]) \
          + '&width=' + str(img_size[0]) + '&srs=EPSG:' + str(crs) \
          + '&bbox=' + bbox

    return url

def get_bbox(data):
    if data is None:
        print('Geopedia request must have bbox')
        return None
    if isinstance(data, str):
        data = data.split(',')
    if isinstance(data, list):
        if len(data) == 2 and (isinstance(data[0], list) or isinstance(data[0], tuple)) and (isinstance(data[0], list) or isinstance(data[0], tuple)):
            data = [coord for point in data for coord in point]
        if len(data) == 4:
            return list(map(float, data))
    return None

def get_crs(data):
    if data:
        return int(data)
    else:
        return WGS84

