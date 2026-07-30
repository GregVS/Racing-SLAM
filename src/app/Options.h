#pragma once

#include <optional>
#include <string>

namespace slam::app {

struct Options {
    std::string config_path;
    bool headless = false;
    std::string output_dir; // Optional
    std::string run_id;     // Defaults to a UTC timestamp
    std::string sequence;   // Optional
};

std::optional<Options> parse_options(int argc, char** argv);

} // namespace slam::app
