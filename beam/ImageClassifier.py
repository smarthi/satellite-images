from __future__ import absolute_import

import argparse
import logging
import glob
import FilterCloudyFn
import UNetInference

import apache_beam as beam
from apache_beam.metrics import Metrics
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import StandardOptions


def run(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--input',
                      dest='input',
                      help='Input folder to process.')
  parser.add_argument('--output',
                      dest='output',
                      required=True,
                      help='Output folder to write results to.')
  parser.add_argument('--models',
                      dest='models',
                      help='Input folder to read model parameters.')
  parser.add_argument('--batchsize',
                      dest='batchsize',
                      help='Batch size for processing')
  known_args, pipeline_args = parser.parse_known_args(argv)

  # We use the save_main_session option because one or more DoFn's in this
  # workflow rely on global context (e.g., a module imported at module level).
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True
  pipeline_options.view_as(StandardOptions).streaming = True

  with beam.Pipeline(options=pipeline_options) as p:
      filtered_images = (p | "Read Images" >> beam.Create(glob.glob(known_args.input + '*wms*' + '.png'))
                         | "Batch elements" >> beam.BatchElements(20, known_args.batchsize)
                         | "Filter Cloudy images" >> beam.ParDo(FilterCloudyFn.FilterCloudyFn(known_args.models)))

      filtered_images | "Segment for Land use" >> beam.ParDo(UNetInference.UNetInferenceFn(known_args.models, known_args.output))

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()