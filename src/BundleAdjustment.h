#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core.hpp>
#include "Map.h"

namespace slam {

void bundle_adjustment(Map& map, int window, bool optimize_points);

};