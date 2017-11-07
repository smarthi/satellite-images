'''
Scripts that call Sentinel Index service
'''

import requestDownload
import requestTime

BASE_URL = 'http://services.sentinel-hub.com/index/v2/'

MAX_CLOUD_COVERAGE = 1.0
WGS84 = 4326

def get_tile_info(tile, date):
    if len(date.split('T')) == 1:
        next_date = requestTime.next_date(date)
    else:
        next_date = date
    url = BASE_URL + 'search?maxcc=1.0&maxcount=10000&timefrom=' + date + '&timeto=' + next_date
    values = {"type": "Polygon",
              "crs": {"type" : "name", "properties": {"name" : "urn:ogc:def:crs:EPSG::4326"}},
              "coordinates": [[[-180.0, -90.0], [-180.0, 90.0], [180.0, 90.0], [180.0, -90.0], [-180.0, -90.0]]]
              }
    info = requestDownload.get_json(url, post_values=values)
    if info['hasMore']:
        print('Error: increase maxcount for index service')
        raise
    candidates = []
    for tile_info in info['tiles']:
        path = tile_info['pathFragment']
        tile_name = ''.join(path.split('/')[1:4])
        if tile_name == tile.lstrip('T').lstrip('0'):
            candidates.append(tile_info)
    if len(candidates) == 1:
        return candidates[0]
    else:
        print('Warning: there are', len(candidates), 'candidates with', tile, 'and', date, '.')
        if len(candidates) > 0:
            timestamps = []
            for tile_info in candidates:
                timestamps.append(tile_info['sensingTime'])
            print('Available timestamps:', timestamps)
            return candidates[-1]
        #raise

def get_area_dates(bbox, date_interval, maxcc=MAX_CLOUD_COVERAGE, crs=WGS84):
    geometry = [[[bbox[0], bbox[1]],
                  [bbox[0], bbox[3]],
                  [bbox[2], bbox[3]],
                  [bbox[2], bbox[1]],
                  [bbox[0], bbox[1]]]]
    if int(crs) == WGS84:  # coordinates get reversed only with wgs84 !
        reverse_coords(geometry)
    values={"type": "Polygon", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::" + str(crs)} },
    "coordinates": geometry}
    url = BASE_URL + "search?&maxcount=10000&maxcc=" + str(maxcc) + "&timefrom=" + date_interval[0] + "&timeto=" + date_interval[1]
    info = requestDownload.get_json(url, post_values=values)
    if info['hasMore']:
        print('Error: increase maxcount for index service')
        raise
    dates = [tile_info['sensingTime'].split('.')[0] for tile_info in info['tiles']]
    return sorted(set(dates)) #in case some dates are doubled

def reverse_coords(geometry):
    for j in range(len(geometry)):
        for i in range(len(geometry[j])):
            geometry[j][i] = [geometry[j][i][1], geometry[j][i][0]]

if __name__ == '__main__':
    pass
    # Examples:
