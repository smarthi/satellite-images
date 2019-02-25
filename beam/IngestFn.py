import apache_beam

import os

class IngestFn(apache_beam.DoFn):

    def __init__(self, url, id):
        super(IngestFn, self).__init__()

    def process(self, element):
       """
       Returns clear images after filtering the cloudy ones
       :param element:
       :return:
       """

