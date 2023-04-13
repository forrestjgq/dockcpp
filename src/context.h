#include <memory>
#include "utils.h"

#ifndef DOCK_CONTEXT_H_
# define DOCK_CONTEXT_H_
namespace dock {

class Context;

enum class RequestResult {
    Success,
    Fail,
    Retry,
};
class Request {
public:
    Request()          = default;
    virtual ~Request() = default;
    virtual void run(Context *) {}
    void wait() {
        if(!callback_) {
            evt_.wait();
            evt_.reset();
        }
    }
    void signal() {
        if (callback_) {
            callback_(this);
        } else {
            evt_.set();
        }
    }
    virtual std::string getProp(const std::string &key) {
        return "";
    }
    RequestResult result() {
        return result_;
    }
    bool ok() {
        return result_ == RequestResult::Success;
    }
    void setCallback(std::function<void(Request *)> callback) {
        callback_ = std::move(callback);
    }

protected:
    RequestResult result_ = RequestResult::Success;
private:
    Event evt_;
    std::function<void(Request *)> callback_;
};

using RequestSP = std::shared_ptr<Request>;

class CallableRequest : public Request {
public:
    CallableRequest(std::function<void(Context *)> callback) : callback_(std::move(callback)) {
    }
    ~CallableRequest() override = default;
    void run(Context *ctx) override {
        callback_(ctx);
    }

private:
    std::function<void(Context *)> callback_;
};
class Context {
public:
    Context() = default;
    virtual ~Context() = default;

    virtual void commit(RequestSP req) {}

};

extern std::shared_ptr<Context> createCudaContext(int device);
};  // namespace dock
#endif