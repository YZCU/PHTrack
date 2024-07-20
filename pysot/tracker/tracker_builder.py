from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.phtrack_tracker import phtrackTracker

TRACKS = {
          'phtrackTracker': phtrackTracker
         }


def build_tracker(model, cfg):
    return TRACKS[model, cfg]