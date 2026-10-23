#ifndef GPUTEST_H
#define GPUTEST_H

#include <boost/test/unit_test.hpp>
#include <cstdlib> // std::system
#include <string>  // std::string

// Runs the test unit testPath (e.g. "WithoutGpu") of the current test binary
// in a child process in which no CUDA device is visible. CUDA reads the
// visible devices once per process, so this cannot be tested in-process.
// Test units meant for this are disabled by default (decorator
// boost::unit_test::disabled()) and only run when named explicitly, as here.
// Returns the exit status of the child, 0 if its tests passed.
inline int runWithoutGpu(std::string const& testPath)
{
    std::string const self =
        boost::unit_test::framework::master_test_suite().argv[0];
    std::string const command = "CUDA_VISIBLE_DEVICES= " + self
                              + " --run_test=" + testPath
                              + " --log_level=error";
    return std::system(command.c_str());
}

#endif
