#ifndef VESIN_CPU_THREADPOOL_HPP
#define VESIN_CPU_THREADPOOL_HPP

#include <algorithm>
#include <exception>
#include <vector>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace vesin {
namespace cpu {

/// Small reusable thread pool for CPU parallel regions.
///
/// The pool always creates one worker slot per available CPU core (according to
/// `std::thread::hardware_concurrency()`), and reuses worker threads across
/// multiple `run` calls. Thread 0 is always the caller of `run`, while workers
/// handle thread IDs 1..max_threads()-1.
///
/// A `run` call is identified by a monotonically increasing `generation_`:
/// workers remember the last generation they have executed and only wake when a
/// new one appears.
class ThreadPool {
public:
    ThreadPool():
        n_threads_(std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()))) {
        workers_.reserve(n_threads_ - 1);
        for (size_t thread_id = 1; thread_id < n_threads_; thread_id++) {
            workers_.emplace_back([this, thread_id]() {
                this->worker(thread_id);
            });
        }
    }

    ~ThreadPool() {
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

    /// Maximum number of threads available in this pool.
    ///
    /// This is equal to the number of hardware threads detected at
    /// construction time, clamped to at least 1.
    size_t max_threads() const {
        return n_threads_;
    }

    /// Execute `n_tasks` invocations of `task(task_i, thread_id)`.
    ///
    /// `active_threads` selects how many threads from the pool should
    /// participate in this run. It is clamped to `[1, min(max_threads(), n_tasks)]`.
    ///
    /// Tasks are split deterministically into contiguous chunks based on
    /// `thread_id` among active threads; this keeps ordering stable across
    /// runs.
    ///
    /// Exceptions are captured and rethrown on the caller thread after workers
    /// finish/abort.
    template <typename Function>
    void run(size_t active_threads, size_t n_tasks, Function task) {
        if (n_tasks == 0) {
            return;
        }

        auto run_lock = std::unique_lock<std::mutex>(run_mutex_);

        active_threads = std::max<size_t>(1, active_threads);
        active_threads = std::min(active_threads, n_threads_);
        active_threads = std::min(active_threads, n_tasks);

        if (active_threads <= 1 || n_tasks <= 1) {
            for (size_t task_i = 0; task_i < n_tasks; task_i++) {
                task(task_i, 0);
            }
            return;
        }

        {
            auto lock = std::lock_guard<std::mutex>(mutex_);
            n_tasks_ = n_tasks;
            task_data_ = &task;
            task_function_ = [](void* data, size_t task_i, size_t thread_id) {
                auto* function = static_cast<Function*>(data);
                (*function)(task_i, thread_id);
            };
            first_exception_ = nullptr;
            has_exception_.store(false);
            active_threads_ = active_threads;
            running_workers_ = active_threads_ - 1;
            generation_ += 1;
        }
        start_cv_.notify_all();

        /// The caller also participates in the work, so we execute its chunk
        /// directly here while workers are running their chunks in the background.
        this->execute_assigned_tasks(0, active_threads);

        auto lock = std::unique_lock<std::mutex>(mutex_);
        done_cv_.wait(lock, [this]() {
            return running_workers_ == 0;
        });

        if (first_exception_ != nullptr) {
            std::rethrow_exception(first_exception_);
        }
    }

private:
    /// Main loop for background workers.
    ///
    /// Workers block on `start_cv_` while idle, wake for each new generation,
    /// execute their chunk, then decrement `running_workers_`.
    void worker(size_t thread_id) {
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

    /// Execute this thread's deterministic chunk for the current generation.
    void execute_assigned_tasks(size_t thread_id, size_t active_threads) {
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

    size_t n_threads_;
    std::vector<std::thread> workers_;

    /// Serialize `run` calls on this pool.
    ///
    /// The current implementation stores one shared run context; allowing
    /// concurrent `run` calls would race on that state.
    std::mutex run_mutex_;

    /// Protects shared run state (`stopping_`, `generation_`, `running_workers_`,
    /// `task_`, `n_tasks_`, and `first_exception_`).
    std::mutex mutex_;
    /// Signaled when a new run generation starts or shutdown begins.
    std::condition_variable start_cv_;
    /// Signaled when all background workers completed the current run.
    std::condition_variable done_cv_;

    /// Are we trying to stop the pool? If true, workers should exit as soon as
    /// possible
    bool stopping_ = false;
    /// Incremented for each new `run` and for shutdown.
    ///
    /// Together with each worker's local `seen_generation`, this separates real
    /// work starts from spurious wakeups and prevents re-running the same batch.
    size_t generation_ = 0;
    /// Number of background workers (thread IDs >= 1) still active in the
    /// current generation.
    size_t running_workers_ = 0;
    /// Number of threads participating in the current generation.
    size_t active_threads_ = 1;

    using task_function_t = void (*)(void*, size_t, size_t);

    size_t n_tasks_ = 0;
    void* task_data_ = nullptr;
    task_function_t task_function_ = nullptr;

    /// Store the first exception thrown by any worker, if any, to rethrow on
    /// the caller thread after workers finish/abort.
    std::exception_ptr first_exception_;
    std::atomic<bool> has_exception_ = false;
};

} // namespace cpu
} // namespace vesin

#endif // VESIN_CPU_THREADPOOL_HPP
