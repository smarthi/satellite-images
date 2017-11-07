'''
Script with useful time/date functions
'''

import datetime

# returns list of dates, start_date and end_date included
def get_dates_in_range(start_date, end_date):
    start_dt = iso_to_datetime(start_date)
    end_dt = iso_to_datetime(end_date)
    num_days = int((end_dt - start_dt).days)
    return [datetime_to_iso(start_dt + datetime.timedelta(i)) for i in range(num_days + 1)]

def next_date(date):
    dt = iso_to_datetime(date)
    return datetime_to_iso(dt + datetime.timedelta(1))

def prev_date(date):
    dt = iso_to_datetime(date)
    return datetime_to_iso(dt - datetime.timedelta(1))

def iso_to_datetime(date):
    chunks = list(map(int, date.split('T')[0].split('-')))
    return datetime.datetime(chunks[0], chunks[1], chunks[2])

def datetime_to_iso(date, only_date=True):
    if only_date:
        return date.isoformat().split('T')[0]
    else:
        return date.isoformat()

def get_current_date():
    date = datetime.datetime.now()
    return datetime_to_iso(date)
