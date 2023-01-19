
#include "app.hpp"

namespace App {
    int use_amirstan_plugin() {
        INFO("using amirstan_plugin.");
        return initLibAmirstanInferPlugins();
    }
}; // namespace App