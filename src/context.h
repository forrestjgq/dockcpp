#include <memory>
#include "utils.h"

#ifndef DOCK_CONTEXT_H_
# define DOCK_CONTEXT_H_
namespace dock {

class Context;
class Request {
public:
    Request()          = default;
    virtual ~Request() = default;
    virtual void run(Context *) = 0;
    void wait() {
        evt_.wait();
        evt_.reset();
    }
    void signal() {
        evt_.set();
    }
    virtual std::string getProp(const std::string &key) {
        return "";
    }

private:
    Event evt_;
};

using RequestSP = std::shared_ptr<Request>;
class Context {
public:
    Context() = default;
    virtual ~Context() = default;

    virtual void commit(RequestSP req) {}

};

extern std::shared_ptr<Context> createCudaContext(int device);
};  // namespace dock
#endif