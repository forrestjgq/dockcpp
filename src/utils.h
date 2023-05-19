#ifndef DOCK_UTILS_H
#define DOCK_UTILS_H

# include <mutex>
# include <queue>
# include <thread>
#include <iostream>
#include <functional>

# include <condition_variable>
namespace dock {

template <typename T>
class BlockQueue {
public:
    const int MIN_WAIT_MS = 5;
    T pop() {
        std::unique_lock<std::mutex> mlock(mutex_);
        while (queue_.empty()) {
            cond_.wait(mlock);
        }
        auto val = queue_.front();
        queue_.pop();
        return val;
    }

    void pop(T& item) {
        std::unique_lock<std::mutex> mlock(mutex_);
        while (queue_.empty()) {
            cond_.wait(mlock);
        }
        item = queue_.front();
        queue_.pop();
    }

    T popWait(int tx = 1) {
        std::unique_lock<std::mutex> mlock(mutex_);
        if (queue_.empty()) {
            tx = tx > MIN_WAIT_MS ? tx : MIN_WAIT_MS;
            if (!cond_.wait_for(mlock, std::chrono::milliseconds(tx), [this]() {
                    return !this->queue_.empty();
                })) {
                return NULL;  // timeout
            }
        }

        auto val = queue_.front();
        queue_.pop();
        return val;
    }

    void push(const T& item) {
        std::unique_lock<std::mutex> mlock(mutex_);
        queue_.push(item);
        mlock.unlock();
        cond_.notify_one();
    }
    bool empty() {
        return queue_.empty();
    }
    inline size_t size() {
        return queue_.size();
    }
    BlockQueue()                             = default;
    BlockQueue(const BlockQueue&)            = delete;  // disable copying
    BlockQueue& operator=(const BlockQueue&) = delete;  // disable assignment

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
};
enum { ZFZ_EVENT_FAIL = (-1), ZFZ_EVENT_SUCCESS = 0, ZFZ_EVENT_TIME_OUT = 1 };
class Event {
public:
    Event(bool init_signal = false, bool manual_reset = true)
        : signal_(init_signal), manual_reset_(manual_reset), blocked_(0) {
    }

    ~Event() {
    }

private:
    Event(const Event&)            = delete;
    Event(Event&&)                 = delete;
    Event& operator=(const Event&) = delete;

public:
    int wait(const int time_out_ms = (-1)) {
        std::unique_lock<std::mutex> ul(lock_);

        if (signal_) {
            if (!manual_reset_) {
                signal_ = false;
            }

            return ZFZ_EVENT_SUCCESS;
        } else {
            if (time_out_ms == 0) {
                return ZFZ_EVENT_TIME_OUT;
            } else {
                ++blocked_;
            }
        }

        if (time_out_ms >= 0) {
            std::chrono::milliseconds wait_time_ms(time_out_ms);
            auto result = cv_.wait_for(ul, wait_time_ms, [&] { return signal_; });
            --blocked_;
            if (result) {
                if (!manual_reset_) {
                    signal_ = false;
                }
                return ZFZ_EVENT_SUCCESS;
            } else {
                return ZFZ_EVENT_TIME_OUT;
            }
        } else {
            cv_.wait(ul, [&] { return signal_; });
            --blocked_;
            if (!manual_reset_) {
                signal_ = false;
            }
            return ZFZ_EVENT_SUCCESS;
        }
    }

    void set() {
        std::lock_guard<std::mutex> lg(lock_);

        signal_ = true;
        if (blocked_ > 0) {
            if (manual_reset_) {
                cv_.notify_all();
            } else {
                cv_.notify_one();
            }
        }
    }

    void reset() {
        lock_.lock();
        signal_ = false;
        lock_.unlock();
    }

private:
    std::mutex lock_;
    std::condition_variable cv_;
    bool signal_       = false;
    bool manual_reset_ = true;
    int blocked_       = 0;
};  // class Event

class Clock {
public:
    std::uint64_t ts_ = 0;
    std::uint64_t mark() {
        auto cur = now();
        auto du  = cur - ts_;
        ts_      = cur;
        return du;
    }

private:
    std::uint64_t now() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    }
};
};
#endif