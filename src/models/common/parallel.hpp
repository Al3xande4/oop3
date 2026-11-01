#pragma once

#include <concepts>
#include <exception>
#include <stop_token>
#include <thread>
#include <vector>

template <typename Func>
concept WorkerFunc = requires(Func func, size_t worker_id, const std::stop_source &stop) {
    { func(worker_id, stop) } -> std::same_as<void>;
};

class ParallelWorker
{
private:
    size_t m_num_workers;

public:
    ParallelWorker(size_t num_workers) : m_num_workers(num_workers) {}

    template <WorkerFunc Func>
    void perform(Func func)
    {
        std::stop_source global_stop;

        if (m_num_workers <= 1)
        {
            func(0, global_stop);
            return;
        }

        std::vector<std::exception_ptr> errors(m_num_workers, nullptr);

        {
            std::vector<std::jthread> workers;
            workers.reserve(m_num_workers);

            for (size_t worker_id = 0; worker_id < m_num_workers; worker_id++)
            {
                workers.emplace_back(
                    [&, worker_id]()
                    {
                        try
                        {
                            func(worker_id, global_stop);
                        }
                        catch (...)
                        {
                            errors[worker_id] = std::current_exception();
                            global_stop.request_stop();
                        }
                    }
                );
            }
        }

        for (const auto &error : errors)
            if (error)
                std::rethrow_exception(error);
    }
};
