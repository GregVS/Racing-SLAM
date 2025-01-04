#pragma once

#include "Map.h"

namespace slam {

void bundle_adjustment(Map& map, int window, bool optimize_points);

};