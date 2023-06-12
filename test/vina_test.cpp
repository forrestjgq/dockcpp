#include <stdlib.h>
namespace dock {
extern bool create_vina_server(int device, int nrinst);
}
extern int run_vina(int argc, const char* argv[]);
int main(int argc, const char* argv[]) {
    auto device = atoi(argv[1]);
    dock::create_vina_server(device, 1);
    run_vina(argc-1, argv+1);
    return 0;
}