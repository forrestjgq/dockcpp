#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H
#include <functional>
#include <assert.h>
#include <iostream>
#include <vector>
namespace dock {

// memory pool manages memory blocks, it will grow automatically as required.
class MemPool {
private:
    struct MemBlock {
        explicit MemBlock(void *p, void *dp, int sz) {
            ptr_    = (uint8_t *)p;
            dptr_ = (uint8_t *)dp;
            size_   = sz;
            offset_ = 0;
        }
        uint8_t *ptr_ = nullptr;
        uint8_t *dptr_ = nullptr;
        int size_     = 0;
        int offset_   = 0;
        void reset() {
            offset_ = 0;
        }
        void *crop(int sz, int align, void **dptr) {
            sz = (sz + align - 1) / align * align;
            if (left() >= sz) {
                auto p = ptr_ + offset_;
                auto dp = dptr_ + offset_;
                offset_ += sz;
                if (dptr) {
                    *dptr = dp;
                }
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
    // f_alloc_ is used to allocate a memory, it returns the memory address at least has sz bytes.
    // for device mapped memory, the pptr will be filled with device memory address(while returning
    // host address), for other memories, pptr will be filled with returning address.
    // pptr will NOT be filled unless its not NULL
    using f_alloc_  = std::function<void *(int sz, void **pptr)>;
    // free memory allocated by f_alloc_
    using f_dealloc = std::function<void(void *, void *)>;

    explicit MemPool(f_alloc_ alloc, f_dealloc deaclloc, int blksz, int align = sizeof(double))
        : alloc_(std::move(alloc)), dealloc_(deaclloc), blksize_(blksz), align_(align) {
        blksize_ = (blksize_ + align - 1) / align * align;
    }
    ~MemPool() {
        for (auto &b : blks_) {
            if (b.ptr_) {
                dealloc_((void *)(b.ptr_), (void *)b.dptr_);
            }
        }
    }
    void reset() {
        for (auto &b : blks_) {
            b.reset();
        }
    }
    void *alloc(int sz, void **dptr = nullptr) {
        for (auto &b : blks_) {
            auto p = b.crop(sz, align_, dptr);
            if (p != nullptr) {
                return p;
            }
        }
        if(!grow(sz)) {
            return nullptr;
        }
        return blks_.back().crop(sz, align_, dptr);
    }

protected:
    bool grow(int sz) {
        sz     = (sz + align_ - 1) / align_ * align_;
        sz     = (sz + blksize_ - 1) / blksize_ * blksize_;
        void *dp = nullptr;
        auto p = alloc_(sz, &dp);
        if (!p) {
            return false;
        }
        // uint8_t *p1 = (uint8_t *)p;
        // p1 += sz;
        // std::cout << "grow start " << p << " end " << (void *)p1 << " sz " << sz << std::endl;
        blks_.emplace_back(p, dp, sz);
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