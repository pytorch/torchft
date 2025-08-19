# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from hta.trace_analysis import TraceAnalysis

_PROFILES_DIR = "output/replica-0/profiles/step-120"


def main():
    analyzer = TraceAnalysis(trace_dir=_PROFILES_DIR)
    cp_graph, success = analyzer.critical_path_analysis(
        rank=0, annotation="", instance_id=None
    )
    if not success:
        print("Critical path analysis failed")
        return
    analyzer.overlay_critical_path_analysis(0, cp_graph, output_dir=_PROFILES_DIR)


if __name__ == "__main__":
    main()
