#include "Options.h"

#include <ctime>
#include <filesystem>
#include <iostream>

namespace slam::app {

namespace {

void print_usage(const char* program)
{
    std::cerr << "Usage: " << program
              << " <config.yaml> [--headless] [--output-dir <dir>] [--run-id <id>]"
                 " [--sequence <name>] [--last-frame <index>]"
              << '\n';
}

std::string utc_timestamp()
{
    std::time_t now = std::time(nullptr);
    char buffer[16];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", std::gmtime(&now));
    return buffer;
}

} // namespace

std::optional<Options> parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--headless") {
            options.headless = true;
        } else if (arg == "--last-frame") {
            if (i + 1 >= argc) {
                std::cerr << arg << " requires a value" << '\n';
                print_usage(argv[0]);
                return std::nullopt;
            }
            options.last_frame = std::stoul(argv[++i]);
        } else if (arg == "--output-dir" || arg == "--run-id" || arg == "--sequence") {
            if (i + 1 >= argc) {
                std::cerr << arg << " requires a value" << '\n';
                print_usage(argv[0]);
                return std::nullopt;
            }
            std::string value = argv[++i];
            if (arg == "--output-dir") {
                options.output_dir = value;
            } else if (arg == "--run-id") {
                options.run_id = value;
            } else {
                options.sequence = value;
            }
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << '\n';
            print_usage(argv[0]);
            return std::nullopt;
        } else if (options.config_path.empty()) {
            options.config_path = arg;
        } else {
            std::cerr << "Unexpected argument: " << arg << '\n';
            print_usage(argv[0]);
            return std::nullopt;
        }
    }

    if (options.config_path.empty()) {
        print_usage(argv[0]);
        return std::nullopt;
    }
    if (options.run_id.empty()) {
        options.run_id = utc_timestamp();
    }
    if (options.sequence.empty()) {
        options.sequence = std::filesystem::path(options.config_path).stem().string();
    }
    return options;
}

} // namespace slam::app
