
namespace dock {
extern bool create_vina_server(int device, int nrinst);
}
extern int run_vina(int argc, const char* argv[]);
int main(int argc, const char* argv[]) {
    dock::create_vina_server(1, 1);
    run_vina(argc, argv);
    return 0;
}