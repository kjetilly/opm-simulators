// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 *
 * \brief This file serves as an example of how to use the tasklet mechanism for
 *        asynchronous work, especially for tasklets that fail.
 */

// Note: we do not use boost.test as it does not cleanly combine with fork() usage

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
#include <cassert>

#include "config.h"
#include <opm/models/parallel/tasklets.hpp>

std::mutex outputMutex;

// The runner is created on the heap for the assertion and outputs in the run function of the tasklets.
std::unique_ptr<Opm::TaskletRunner> runner{};

class SleepTasklet : public Opm::TaskletInterface
{
public:
    SleepTasklet(long long mseconds, long long id)
        : mseconds_(mseconds),
          id_(id)
    {}

    void run() override
    {
        assert(0 <= runner->workerThreadIndex() && runner->workerThreadIndex() < runner->numWorkerThreads());
        std::cout << "Sleep tasklet " << id_ << " of " << mseconds_ << " ms starting sleep on worker thread " << runner->workerThreadIndex() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(mseconds_));
        std::lock_guard<std::mutex> guard(outputMutex);
        std::cout << "Sleep tasklet " << id_ << " of " << mseconds_ << " ms completed by worker thread " << runner->workerThreadIndex() << std::endl;
    }

private:
    long long mseconds_;
    long long id_;
};

class FailingSleepTasklet : public Opm::TaskletInterface
{
public:
    FailingSleepTasklet(long long mseconds)
        : mseconds_(mseconds)
    {}
    void run() override
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(mseconds_));
        std::lock_guard<std::mutex> guard(outputMutex);
        std::cout << "Failing sleep tasklet of " << mseconds_ << " ms failing now, on work thread " << runner->workerThreadIndex() << std::endl;
        throw std::logic_error("Intentional failure for testing");
    }

private:
    long long mseconds_;
};

void execute () {
    long long numWorkers = 2;
    runner = std::make_unique<Opm::TaskletRunner>(numWorkers);

    // the master thread is not a worker thread
    assert(runner->workerThreadIndex() < 0);
    assert(runner->numWorkerThreads() == numWorkers);

    // Dispatch some successful tasklets
    for (long long i = 0; i < 5; ++i) {
        runner->barrier();

        if (runner->failure()) {
            exit(EXIT_FAILURE);
        }
        auto st = std::make_shared<SleepTasklet>(10,i);
        runner->dispatch(st);
    }

    runner->barrier();
    if (runner->failure()) {
        exit(EXIT_FAILURE);
    }
    // Dispatch a failing tasklet
    auto failingSleepTasklet = std::make_shared<FailingSleepTasklet>(100);
    runner->dispatch(failingSleepTasklet);

    // Dispatch more successful tasklets
    for (long long i = 5; i < 10; ++i) {
        runner->barrier();

        if (runner->failure()) {
            exit(EXIT_FAILURE);
        }
        auto st = std::make_shared<SleepTasklet>(10,i);
        runner->dispatch(st);
    }

    std::cout << "before barrier" << std::endl;
    runner->barrier();
}

long long main()
{
    pid_t pid = fork(); // Create a new process, such that this child process can call exit(EXIT_FAILURE)
    if (pid == -1) {
        assert(0 && "Fork failed");
    } else if (pid == 0) {
        // Child process
        execute();
        _exit(0);  // Should never reach here
    } else {
        // Parent process
        std::cout << "Checking failure of child process with parent process, process id " << pid << std::endl;
        long long status;
        waitpid(pid, &status, 0);
        assert(WIFEXITED(status));  // Check if the child process exited
        assert(WEXITSTATUS(status) == EXIT_FAILURE);  // Check if the exit status is EXIT_FAILURE
    }
}
