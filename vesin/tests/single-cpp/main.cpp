#include <vesin.h>

int main() {
    auto neighbors = VesinNeighborList();

    auto options = VesinOptions();
    options.cutoff = 5.0;
    options.full = true;
    options.return_shifts = true;

    const char* error_message = nullptr;
    vesin_neighbors(nullptr, 0, nullptr, nullptr, VesinDevice{VesinCPU, 0}, options, &neighbors, &error_message);

    vesin_free(&neighbors);
}
