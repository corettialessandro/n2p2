#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-train
#include "Example_nnp_train.h"
#include "nnp_test.h"
#include "tolerance.h"
#include "utility.h"

#include <cstddef> // std::size_t
#include <fstream> // std::ifstream
#include <string>  // std::string
#include <vector>  // std::vector

using namespace std;
using namespace nnp;

BoostDataContainer<Example_nnp_train> container;

// GPU builds only: allowed relative deviation per epoch class. The untrained
// network (epoch 0) is a pure evaluation, so it must agree with the CPU to
// the printed digits. After one epoch the trajectories still agree closely.
// By the last epoch, summation-order differences have been amplified by the
// selection of update candidates (see Example_nnp_train.h; for Cu2S_PBE by
// about a factor of ten per epoch), so only a sanity bound is checked there:
// an RMSE within a factor of ten of the CPU result.
double const gpuRelativeEpoch0 = gpuRelativePrinted;
double const gpuRelativeEpoch1 = 1.0E-6;
double const gpuRelativeLastEpoch = 9.0;

// Compares one learning-curve.out row against reference RMSEs.
void checkRow(vector<string> const&   columns,
              bool                    charges,
              LearningCurveRow const& reference,
              Tolerance const&        tolerance)
{
    BOOST_TEST_INFO("epoch " << reference.epoch);
    REQUIRE_CLOSE_TO_REFERENCE(reference.rmseTrain, stod(columns.at(1)),
                               tolerance);
    REQUIRE_CLOSE_TO_REFERENCE(reference.rmseTest, stod(columns.at(2)),
                               tolerance);
    if (charges) return;
    REQUIRE_CLOSE_TO_REFERENCE(reference.rmseForcesTrain, stod(columns.at(9)),
                               tolerance);
    REQUIRE_CLOSE_TO_REFERENCE(reference.rmseForcesTest, stod(columns.at(10)),
                               tolerance);
}

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_train const example)
{
    bool epochFound = false;
    string line;
    ifstream file;

    // Grep for "CI-CHECK" for code changes if you want to review output files
    // from non-interactive CI run.
    //ifstream source("nnp-train.log.0000", ios::binary);
    //ofstream dest("../nnp-train.log.0000." + example.name, ios::binary);
    //dest << source.rdbuf();
    //source.close();
    //dest.close();
    //ifstream source2("learning-curve.out", ios::binary);
    //ofstream dest2("../learning-curve.out." + example.name, ios::binary);
    //dest2 << source2.rdbuf();
    //source2.close();
    //dest2.close();

    string filename{"learning-curve.out"};
    if (example.args != "")
        filename += ".stage-" + example.args;

    bool const charges = (example.args == "1");
    LearningCurveRow last;
    last.epoch = example.lastEpoch;
    last.rmseTrain = charges ? example.rmseChargesTrain : example.rmseEnergyTrain;
    last.rmseTest = charges ? example.rmseChargesTest : example.rmseEnergyTest;
    last.rmseForcesTrain = charges ? 0.0 : example.rmseForcesTrain;
    last.rmseForcesTest = charges ? 0.0 : example.rmseForcesTest;

    Tolerance const toleranceLast =
        buildTolerance(Tolerance(example.accuracy,
                                 example.accuracyLastEpochRelative),
                       gpuRelativeLastEpoch);

    size_t earlyFound = 0;
    file.open(filename);
    BOOST_REQUIRE(file.is_open());
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));
        if (columns.at(0) == to_string(example.lastEpoch))
        {
            epochFound = true;
            checkRow(columns, charges, last, toleranceLast);
        }
        for (auto const& early : example.earlyEpochs)
        {
            if (columns.at(0) != to_string(early.epoch)) continue;
            earlyFound++;
            double const relativeGpu = early.epoch == 0 ? gpuRelativeEpoch0
                                                        : gpuRelativeEpoch1;
            checkRow(columns,
                     charges,
                     early,
                     buildTolerance(Tolerance(example.accuracy),
                                    relativeGpu,
                                    0.0));
        }
    }
    file.close();
    BOOST_REQUIRE_MESSAGE(epochFound,
                          string("ERROR: Epoch information was not "
                                 "found in file."));
    BOOST_REQUIRE_EQUAL(earlyFound, example.earlyEpochs.size());

    return;
}
