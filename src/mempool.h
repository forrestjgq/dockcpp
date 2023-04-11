#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H
#include <functional>
#include <assert.h>
#include <iostream>
namespace dock {

class MemPool {
private:
    struct MemBlock {
        explicit MemBlock(void *p, int sz) {
            ptr_    = (uint8_t *)p;
            size_   = sz;
            offset_ = 0;
        }
        uint8_t *ptr_ = nullptr;
        int size_     = 0;
        int offset_   = 0;
        void reset() {
            offset_ = 0;
        }
        void *crop(int sz, int align) {
            sz = (sz + align - 1) / align * align;
            if (left() >= sz) {
                auto p = ptr_ + offset_;
                offset_ += sz;
                return p;
            }
            return nullptr;
        }
        int left() {
            return size_ - offset_;
        }
        bool newBlock() {
            return offset_ = 0;
        }
    };

public:
    using f_alloc_  = std::function<void *(int)>;
    using f_dealloc = std::function<void(void *)>;
    explicit MemPool(f_alloc_ alloc, f_dealloc deaclloc, int blksz, int align = sizeof(double))
        : alloc_(std::move(alloc)), dealloc_(deaclloc), blksize_(blksz), align_(align) {
        blksize_ = (blksize_ + align - 1) / align * align;
    }
    ~MemPool() {
        for (auto &b : blks_) {
            if (b.ptr_) {
                dealloc_((void *)(b.ptr_));
            }
        }
    }
    void reset() {
        for (auto &b : blks_) {
            b.reset();
        }
    }
    void *alloc(int sz) {
        for (auto &b : blks_) {
            auto p = b.crop(sz, align_);
            if (p != nullptr) {
                return p;
            }
        }
        if(!grow(sz)) {
            return nullptr;
        }
        return blks_.back().crop(sz, align_);
    }

protected:
    bool grow(int sz) {
        sz     = (sz + align_ - 1) / align_ * align_;
        sz     = (sz + blksize_ - 1) / blksize_ * blksize_;
        auto p = alloc_(sz);
        if (!p) {
            return false;
        }
        uint8_t *p1 = (uint8_t *)p;
        p1 += sz;
        std::cout << "grow start " << p << " end " << (void *)p1 << " sz " << sz << std::endl;
        blks_.emplace_back(p, sz);
        return true;
    }

private:
    f_alloc_ alloc_;
    f_dealloc dealloc_;
    int blksize_ = 0;
    int align_;
    std::vector<MemBlock> blks_;
};
};
#endif