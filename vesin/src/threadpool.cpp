#include "threadpool.hpp"

#include <algorithm>

using namespace vesin::cpu;

ThreadPool::ThreadPool() {
    n_threads_ = std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()));
}


ThreadPool::~ThreadPool() {
    {
        auto lock = std::lock_guard<std::mutex>(mutex_);
        stopping_ = true;
        generation_ += 1;
    }
    start_cv_.notify_all();

    for (auto& worker_thread : workers_) {
        worker_thread.join();
    }
}


ThreadPool& ThreadPool::global() {
    static auto pool = ThreadPool();
    return pool;
}


void ThreadPool::ensure_workers() {
    if (!workers_.empty() || n_threads_ <= 1) {
        return;
    }

    {
        auto lock = std::lock_guard<std::mutex>(mutex_);
        if (stopping_) {
            return;
        }
    }

    workers_.reserve(n_threads_ - 1);
    for (size_t thread_id = 1; thread_id < n_threads_; thread_id++) {
        workers_.emplace_back([this, thread_id]() {
            this->worker(thread_id);
        });
    }
}


void ThreadPool::prepare_for_fork() {
    auto run_lock = std::unique_lock<std::mutex>(run_mutex_);
    {
        auto lock = std::lock_guard<std::mutex>(mutex_);
        stopping_ = true;
        generation_ += 1;
    }
    start_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    workers_.clear();
}


void ThreadPool::reinit_after_fork() {
    after_fork_ = false;
    auto lock = std::lock_guard<std::mutex>(mutex_);
    stopping_ = false;
    generation_ = 0;
    running_workers_ = 0;
    active_threads_ = 1;
    n_tasks_ = 0;
    task_data_ = nullptr;
    task_function_ = nullptr;
    first_exception_ = nullptr;
    has_exception_ = false;
}


void ThreadPool::worker(size_t thread_id) {
    size_t seen_generation = 0;

    auto lock = std::unique_lock<std::mutex>(mutex_);
    while (true) {
        start_cv_.wait(lock, [this, seen_generation]() {
            return stopping_ || generation_ != seen_generation;
        });

        if (stopping_) {
            return;
        }

        seen_generation = generation_;
        auto is_active = thread_id < active_threads_;
        auto active_threads = active_threads_;
        lock.unlock();
        if (is_active) {
            this->execute_assigned_tasks(thread_id, active_threads);
        }
        lock.lock();

        if (is_active) {
            running_workers_ -= 1;
            if (running_workers_ == 0) {
                done_cv_.notify_one();
            }
        }
    }
}

void ThreadPool::execute_assigned_tasks(size_t thread_id, size_t active_threads) {
    auto begin = (thread_id * n_tasks_) / active_threads;
    auto end = ((thread_id + 1) * n_tasks_) / active_threads;

    for (size_t task_i = begin; task_i < end; task_i++) {
        if (has_exception_.load()) {
            return;
        }

        try {
            task_function_(task_data_, task_i, thread_id);
        } catch (...) {
            auto lock = std::lock_guard<std::mutex>(mutex_);
            if (first_exception_ == nullptr) {
                first_exception_ = std::current_exception();
                has_exception_.store(true);
            }
            return;
        }
    }
}


#ifndef _WIN32

#include <pthread.h>

// ---------------------------------------------------------------------------
// pthread_atfork handlers — tear down the pool before fork so that no
// mutexes / condition variables are held across the copy, then let both
// parent and child lazily recreate workers on their next run().
// ---------------------------------------------------------------------------

void vesin::cpu::details::fork_prepare() {
    ThreadPool::global().prepare_for_fork();
}

void vesin::cpu::details::fork_parent() {
    ThreadPool::global().reinit_after_fork();
}

void vesin::cpu::details::fork_child() {
    ThreadPool::global().after_fork_ = true;
}

struct ForkHandlerRegistrar {
    ForkHandlerRegistrar() {
        pthread_atfork(
            vesin::cpu::details::fork_prepare,
            vesin::cpu::details::fork_parent,
            vesin::cpu::details::fork_child
        );
    }
} fork_handler_registrar_;

#endif // _WIN32
