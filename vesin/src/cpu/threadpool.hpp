#ifndef VESIN_CPU_THREADPOOL_HPP
#define VESIN_CPU_THREADPOOL_HPP

#include <cstddef>
#include <exception>
#include <vector>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>


namespace vesin::cpu {

namespace details {
    // Functions for `pthread_atfork` handlers.
    void fork_prepare();
    void fork_parent();
    void fork_child();
}

/// Task scheduling strategy for `ThreadPool::run`.
///
/// `Static` splits the tasks into contiguous, deterministic chunks (one per
/// thread); `Dynamic` dispatches tasks in fixed-size chunks to whichever
/// thread becomes free first, which improves load balancing when the per-task
/// cost varies (at the cost of non-deterministic ordering of the output).
struct Scheduling {
    enum Kind { Static, Dynamic };

    /// Scheduling strategy in use.
    Kind kind = Static;
    /// Number of tasks per chunk when `kind == Dynamic`. Ignored for `Static`.
    size_t chunk_size = 256;

    static Scheduling static_() { return {Static, 0}; }
    static Scheduling dynamic(size_t chunk = 256) { return {Dynamic, chunk}; }
};

/// Decide whether running an operation on the thread pool is worth its
/// dispatch overhead.
///
/// `serial_cost_us` is an estimate, in microseconds, of how long the whole
/// operation takes on a single thread, and `n_run_calls` the number of
/// `ThreadPool::run` calls it is split into. Returns `thread_count` when
/// running in parallel should be faster, and 1 otherwise.
size_t pick_n_threads(double serial_cost_us, size_t n_run_calls, size_t thread_count);

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
    ThreadPool();

    ~ThreadPool();

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
    /// `scheduling` selects how tasks are assigned to threads:
    ///  - `Scheduling::static_()` (the default) splits tasks into contiguous,
    ///    deterministic chunks — one per thread. This keeps ordering stable
    ///    across runs.
    ///  - `Scheduling::dynamic(chunk)` dispatches tasks in fixed-size chunks
    ///    to whichever thread is free. This improves load balancing when the
    ///    per-task cost varies, at the cost of non-deterministic ordering of
    ///    the output (the set of results is still correct).
    ///
    /// Exceptions are captured and rethrown on the caller thread after workers
    /// finish/abort.
    template <typename Function>
    void run(size_t active_threads, size_t n_tasks, Scheduling scheduling, Function task) {
        if (n_tasks == 0) {
            return;
        }

        auto run_lock = std::unique_lock<std::mutex>(run_mutex_);

        if (after_fork_) {
            this->reinit_after_fork();
        }

        this->ensure_workers();

        active_threads = std::max<size_t>(1, active_threads);
        active_threads = std::min(active_threads, n_threads_);
        active_threads = std::min(active_threads, n_tasks);

        if (active_threads <= 1 || n_tasks <= 1 || workers_.empty()) {
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
            has_exception_ = false;
            active_threads_ = active_threads;
            running_workers_ = active_threads_ - 1;
            scheduling_ = scheduling;
            next_chunk_.store(0, std::memory_order_relaxed);
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

    /// Access the global thread pool (lazily created on first use).
    static ThreadPool& global();

private:
    /// Create worker threads if they don't exist yet.
    ///
    /// Called at the top of every `run()` — workers are created lazily rather
    /// than in the constructor so that `pthread_atfork` can tear them down
    /// before a fork and let `run()` recreate them afterwards.
    void ensure_workers();

    /// Prepare the pool for fork: stop all workers, join them, and clear the
    /// worker vector.  Afterwards no mutexes or condition variables represent
    /// any waiter state.
    ///
    /// This acquires `run_mutex_` to wait for any in-flight `run()` call to
    /// finish.  Once it returns, the window before the actual `fork()` is
    /// tiny, but a concurrent call to `run()` may still slip in.  Such a call
    /// will see `stopping_ == true`, skip worker creation, and fall back to
    /// single-threaded execution — guaranteeing that `run_mutex_` is released
    /// before the kernel `fork()`.
    void prepare_for_fork();

    /// Reset bookkeeping after fork.  Workers are NOT created here — that
    /// happens lazily on the first `run()` call.
    void reinit_after_fork();

    /// Main loop for background workers.
    ///
    /// Workers block on `start_cv_` while idle, wake for each new generation,
    /// execute their chunk, then decrement `running_workers_`.
    void worker(size_t thread_id);

    /// Execute this thread's tasks for the current generation.
    ///
    /// Dispatches according to `sched_`: contiguous chunks for `Static`, or
    /// atomically-stealing fixed-size chunks for `Dynamic`.
    void execute_assigned_tasks(size_t thread_id, size_t active_threads);

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
    /// Scheduling strategy for the current generation.
    Scheduling scheduling_;
    /// Next task index to be dispatched in `Dynamic` mode.
    std::atomic<size_t> next_chunk_{0};

    using task_function_t = void (*)(void*, size_t, size_t);

    size_t n_tasks_ = 0;
    void* task_data_ = nullptr;
    task_function_t task_function_ = nullptr;

    /// Store the first exception thrown by any worker, if any, to rethrow on
    /// the caller thread after workers finish/abort.
    std::exception_ptr first_exception_;
    std::atomic<bool> has_exception_ = false;

    /// Did we just fork?  If true, the next `run()` call will reinitialize
    /// bookkeeping and recreate worker threads.
    std::atomic<bool> after_fork_ = false;

    friend void details::fork_prepare();
    friend void details::fork_parent();
    friend void details::fork_child();
};

} // namespace vesin::cpu


#endif // VESIN_CPU_THREADPOOL_HPP
