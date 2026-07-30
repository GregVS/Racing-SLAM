#include "app/Options.h"
#include "app/Runner.h"

int main(int argc, char** argv)
{
    auto options = slam::app::parse_options(argc, argv);
    if (!options) {
        return 1;
    }
    return slam::app::run(*options);
}
