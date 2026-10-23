#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GpuDevice
#include <boost/test/unit_test.hpp>

#include "GpuDevice.h"
#include <cstdlib>   // setenv, unsetenv, getenv
#include <map>       // std::map
#include <stdexcept> // std::runtime_error
#include <string>    // std::string

using namespace std;
using namespace nnp;

// Environment variables read by gpuLocalRank(), in order of precedence.
char const* const localRankVariables[] = {"OMPI_COMM_WORLD_LOCAL_RANK",
                                          "MPI_LOCALRANKID",
                                          "MV2_COMM_WORLD_LOCAL_RANK",
                                          "SLURM_LOCALID"};

// Clears the local rank variables for one test and restores the values set
// by the launcher (e.g. SLURM_LOCALID in a batch job) afterwards.
struct FixtureCleanEnvironment
{
    map<string, string> saved;

    FixtureCleanEnvironment()
    {
        for (char const* name : localRankVariables)
        {
            char const* value = getenv(name);
            if (value != nullptr) saved[name] = value;
            unsetenv(name);
        }
    }

    ~FixtureCleanEnvironment()
    {
        for (char const* name : localRankVariables) unsetenv(name);
        for (auto const& s : saved) setenv(s.first.c_str(),
                                           s.second.c_str(),
                                           1);
    }
};

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(DeviceCount_GpuNode_AtLeastOne)
{
    BOOST_REQUIRE_GT(gpuDeviceCount(), 0);
}

BOOST_FIXTURE_TEST_CASE(LocalRank_NoLauncherVariables_Zero,
                        FixtureCleanEnvironment)
{
    BOOST_REQUIRE_EQUAL(gpuLocalRank(), 0);
}

BOOST_FIXTURE_TEST_CASE(LocalRank_EachLauncherVariable_Read,
                        FixtureCleanEnvironment)
{
    for (char const* name : localRankVariables)
    {
        BOOST_TEST_INFO(name);
        setenv(name, "3", 1);
        BOOST_REQUIRE_EQUAL(gpuLocalRank(), 3);
        unsetenv(name);
    }
}

BOOST_FIXTURE_TEST_CASE(LocalRank_SeveralVariables_MpiBeforeSlurm,
                        FixtureCleanEnvironment)
{
    setenv("SLURM_LOCALID", "0", 1);
    setenv("OMPI_COMM_WORLD_LOCAL_RANK", "2", 1);
    BOOST_REQUIRE_EQUAL(gpuLocalRank(), 2);
}

BOOST_FIXTURE_TEST_CASE(SelectDevice_LocalRanks_SpreadOverDevices,
                        FixtureCleanEnvironment)
{
    int const count = gpuDeviceCount();
    BOOST_REQUIRE_GT(count, 0);
    for (int rank = 0; rank < 2 * count + 1; ++rank)
    {
        BOOST_TEST_INFO("local rank " << rank);
        setenv("OMPI_COMM_WORLD_LOCAL_RANK", to_string(rank).c_str(), 1);
        BOOST_REQUIRE_EQUAL(gpuSelectDevice(), rank % count);
        BOOST_REQUIRE_EQUAL(gpuCurrentDevice(), rank % count);
    }
}

BOOST_AUTO_TEST_CASE(DeviceName_SelectedDevice_NotEmpty)
{
    BOOST_REQUIRE(!gpuDeviceName(gpuCurrentDevice()).empty());
}

BOOST_AUTO_TEST_CASE(DeviceName_InvalidDevice_Throws)
{
    // A failed CUDA call must surface as an exception, not end the process.
    BOOST_REQUIRE_THROW(gpuDeviceName(gpuDeviceCount()), runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
