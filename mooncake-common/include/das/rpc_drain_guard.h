// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace mooncake {

class RpcDrainGuard {
   private:
    struct State {
        std::atomic<bool> stopping{false};
        std::atomic<int> inflight{0};
        std::mutex mutex;
        std::condition_variable cv;
    };

   public:
    class ScopedCall {
       public:
        explicit ScopedCall(const RpcDrainGuard& guard)
            : state_(guard.state_), ok_(try_enter(*state_)) {}
        ~ScopedCall() {
            if (ok_) {
                leave(*state_);
            }
        }
        ScopedCall(const ScopedCall&) = delete;
        ScopedCall& operator=(const ScopedCall&) = delete;
        bool ok() const { return ok_; }

       private:
        static bool try_enter(State& state) {
            if (state.stopping.load(std::memory_order_acquire)) {
                return false;
            }
            state.inflight.fetch_add(1, std::memory_order_acq_rel);
            if (state.stopping.load(std::memory_order_acquire)) {
                leave(state);
                return false;
            }
            return true;
        }

        static void leave(State& state) {
            if (state.inflight.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.cv.notify_all();
            }
        }

        std::shared_ptr<State> state_;
        bool ok_;
    };

    bool drain_for(std::chrono::milliseconds timeout) {
        state_->stopping.store(true, std::memory_order_release);
        std::unique_lock<std::mutex> lock(state_->mutex);
        return state_->cv.wait_for(lock, timeout, [&] {
            return state_->inflight.load(std::memory_order_acquire) == 0;
        });
    }

   private:
    std::shared_ptr<State> state_ = std::make_shared<State>();
};

}  // namespace mooncake
