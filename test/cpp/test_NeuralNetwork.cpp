#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE NeuralNetwork
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"
#include "tolerance.h"
#ifdef N2P2_GPU
#include "gpuTest.h"
#endif

#include "NeuralNetwork.h"
#include <algorithm> // std::max
#include <cmath>     // std::sqrt, std::pow, std::tanh, std::fabs
#include <cstddef>   // std::size_t
#include <fstream>   // std::ofstream, std::ifstream
#include <limits>    // std::numeric_limits
#include <memory>    // std::unique_ptr
#include <stdexcept> // std::runtime_error
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();
double const accuracyNumeric = 1E-6;

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_Architecture_CorrectCounts)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};

    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    BOOST_REQUIRE_EQUAL(nn.getNumNeurons(), 2 + 3 + 1);
    BOOST_REQUIRE_EQUAL(nn.getNumWeights(), 2 * 3 + 3 * 1);
    BOOST_REQUIRE_EQUAL(nn.getNumBiases(), 3 + 1);
    BOOST_REQUIRE_EQUAL(nn.getNumConnections(),
                        nn.getNumWeights() + nn.getNumBiases());
}

BOOST_AUTO_TEST_CASE(SetGetConnections_RoundTrip)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> in(n);
    for (int i = 0; i < n; ++i) in.at(i) = 0.1 * (i + 1);
    nn.setConnections(in.data());

    vector<double> out(n, 0.0);
    nn.getConnections(out.data());
    for (int i = 0; i < n; ++i)
    {
        BOOST_REQUIRE_SMALL(in.at(i) - out.at(i), accuracy);
    }
}

BOOST_AUTO_TEST_CASE(Propagate_LinearNetwork_CorrectOutput)
{
    // All-identity 2-2-1 network: output is a plain linear map, computed
    // here independently and compared against propagate()'s result.
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 2, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    // Connection order (see NeuralNetwork::setConnections()): for each
    // layer, weights ordered by [source neuron][dest neuron], then biases
    // ordered by [dest neuron].
    vector<double> connections = {
        1.0, 2.0,   // x0 -> h0, x0 -> h1
        3.0, 4.0,   // x1 -> h0, x1 -> h1
        0.5, -0.5,  // bias h0, bias h1
        2.0, -1.0,  // h0 -> E, h1 -> E
        0.25        // bias E
    };
    nn.setConnections(connections.data());

    double const x0 = 1.5;
    double const x1 = -2.0;
    double input[2] = {x0, x1};
    nn.setInput(input);
    nn.propagate();

    double const h0 = 1.0 * x0 + 3.0 * x1 + 0.5;
    double const h1 = 2.0 * x0 + 4.0 * x1 + (-0.5);
    double const E  = 2.0 * h0 + (-1.0) * h1 + 0.25;

    double output = 0.0;
    nn.getOutput(&output);
    BOOST_REQUIRE_SMALL(output - E, accuracy);
}

BOOST_AUTO_TEST_CASE(CalculateDEdG_MatchesNumericDerivative_OneHiddenLayer)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 4, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
    nn.initializeConnectionsRandomUniform(42);

    double input[2] = {0.3, -0.7};
    nn.setInput(input);
    nn.propagate();

    vector<double> dEdG(2);
    nn.calculateDEdG(dEdG.data());

    double const h = 1.0E-6;
    for (int i = 0; i < 2; ++i)
    {
        double const orig = input[i];

        input[i] = orig + h;
        nn.setInput(input);
        nn.propagate();
        double outHi = 0.0;
        nn.getOutput(&outHi);

        input[i] = orig - h;
        nn.setInput(input);
        nn.propagate();
        double outLo = 0.0;
        nn.getOutput(&outLo);

        input[i] = orig;

        double const numDeriv = (outHi - outLo) / (2.0 * h);
        BOOST_REQUIRE_SMALL(dEdG.at(i) - numDeriv, accuracyNumeric);
    }
}

BOOST_AUTO_TEST_CASE(CalculateDEdG_MatchesNumericDerivative_TwoHiddenLayers)
{
    int const numLayers = 4;
    int const numNeuronsPerLayer[4] = {3, 4, 3, 1};
    NeuralNetwork::ActivationFunction const af[4] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
    nn.initializeConnectionsRandomUniform(7);

    double input[3] = {0.2, -0.4, 0.6};
    nn.setInput(input);
    nn.propagate();

    vector<double> dEdG(3);
    nn.calculateDEdG(dEdG.data());

    double const h = 1.0E-6;
    for (int i = 0; i < 3; ++i)
    {
        double const orig = input[i];

        input[i] = orig + h;
        nn.setInput(input);
        nn.propagate();
        double outHi = 0.0;
        nn.getOutput(&outHi);

        input[i] = orig - h;
        nn.setInput(input);
        nn.propagate();
        double outLo = 0.0;
        nn.getOutput(&outLo);

        input[i] = orig;

        double const numDeriv = (outHi - outLo) / (2.0 * h);
        BOOST_REQUIRE_SMALL(dEdG.at(i) - numDeriv, accuracyNumeric);
    }
}

BOOST_AUTO_TEST_CASE(CalculateDEdG_MatchesNumericDerivative_DeepNormalized)
{
    // Three hidden layers, different activation functions, normalized
    // neurons: all factors of the backward pass.
    int const numLayers = 5;
    int const numNeuronsPerLayer[5] = {4, 6, 5, 3, 1};
    NeuralNetwork::ActivationFunction const af[5] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_SOFTPLUS,
         NeuralNetwork::AF_LOGISTIC,
         NeuralNetwork::AF_TANH};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
    nn.setNormalizeNeurons(true);
    nn.initializeConnectionsRandomUniform(11);

    vector<double> input = {0.3, -0.7, 1.2, 0.4};
    nn.setInput(input.data());
    nn.propagate();
    vector<double> dEdG(4);
    nn.calculateDEdG(dEdG.data());

    double const h = 1.0E-5;
    for (size_t k = 0; k < input.size(); ++k)
    {
        vector<double> in = input;
        double eHi = 0.0;
        double eLo = 0.0;
        in.at(k) = input.at(k) + h;
        nn.setInput(in.data());
        nn.propagate();
        nn.getOutput(&eHi);
        in.at(k) = input.at(k) - h;
        nn.setInput(in.data());
        nn.propagate();
        nn.getOutput(&eLo);
        BOOST_TEST_INFO("input " << k);
        BOOST_REQUIRE_SMALL(dEdG.at(k) - (eHi - eLo) / (2.0 * h), 1.0E-9);
    }
}

BOOST_AUTO_TEST_CASE(CalculateDEdc_MatchesNumericDerivative)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 4, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
    nn.initializeConnectionsRandomUniform(123);

    double input[2] = {0.3, -0.7};
    nn.setInput(input);
    nn.propagate();

    int const numConnections = nn.getNumConnections();
    vector<double> dEdc(numConnections);
    nn.calculateDEdc(dEdc.data());

    vector<double> conn(numConnections);
    nn.getConnections(conn.data());

    double const h = 1.0E-6;
    for (int i = 0; i < numConnections; ++i)
    {
        vector<double> connHi = conn;
        connHi.at(i) += h;
        nn.setConnections(connHi.data());
        nn.setInput(input);
        nn.propagate();
        double outHi = 0.0;
        nn.getOutput(&outHi);

        vector<double> connLo = conn;
        connLo.at(i) -= h;
        nn.setConnections(connLo.data());
        nn.setInput(input);
        nn.propagate();
        double outLo = 0.0;
        nn.getOutput(&outLo);

        nn.setConnections(conn.data());

        double const numDeriv = (outHi - outLo) / (2.0 * h);
        BOOST_REQUIRE_SMALL(dEdc.at(i) - numDeriv, accuracyNumeric);
    }
}

BOOST_AUTO_TEST_CASE(CalculateDFdc_MatchesNumericDerivative)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 4, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
    nn.initializeConnectionsRandomUniform(99);

    double input[2] = {0.3, -0.7};
    nn.setInput(input);
    nn.propagate();

    vector<double> const dGdxyz = {1.0, 0.5};
    int const numConnections = nn.getNumConnections();
    vector<double> dFdc(numConnections);
    nn.calculateDFdc(dFdc.data(), dGdxyz.data());

    vector<double> conn(numConnections);
    nn.getConnections(conn.data());

    // Q(c) = dE/dG(c) . dGdxyz ; calculateDFdc() returns -dQ/dc (see "Note:
    // F = -dE/dx" in NeuralNetwork.cpp).
    auto computeQ = [&](vector<double> const& c) -> double
    {
        nn.setConnections(c.data());
        nn.setInput(input);
        nn.propagate();
        vector<double> dEdG(2);
        nn.calculateDEdG(dEdG.data());
        double Q = 0.0;
        for (size_t i = 0; i < dGdxyz.size(); ++i) Q += dEdG.at(i) * dGdxyz.at(i);
        return Q;
    };

    double const h = 1.0E-6;
    for (int i = 0; i < numConnections; ++i)
    {
        vector<double> cHi = conn;
        cHi.at(i) += h;
        vector<double> cLo = conn;
        cLo.at(i) -= h;
        double const numDeriv = (computeQ(cHi) - computeQ(cLo)) / (2.0 * h);
        BOOST_REQUIRE_SMALL(dFdc.at(i) - (-numDeriv), accuracyNumeric);
    }

    nn.setConnections(conn.data());
}

// calculateDFdc() is linear in the derivatives of the inputs: for each unit
// vector e_i it evaluates exactly the per-input term of the previous
// implementation, sum_i dGdxyz[i] * calculateDFdc(e_i), so this also
// compares the one-pass calculation against it. Deep network with
// normalized neurons and different activation functions.
BOOST_AUTO_TEST_CASE(CalculateDFdc_LinearInDerivativesOfInputs)
{
    int const numLayers = 5;
    int const numNeuronsPerLayer[5] = {6, 7, 5, 4, 1};
    NeuralNetwork::ActivationFunction const af[5] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_SOFTPLUS,
         NeuralNetwork::AF_LOGISTIC,
         NeuralNetwork::AF_IDENTITY};
    for (bool normalize : {false, true})
    {
        BOOST_TEST_INFO("normalized neurons " << normalize);
        NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);
        nn.setNormalizeNeurons(normalize);
        nn.initializeConnectionsRandomUniform(7);

        double const input[6] = {0.3, -0.7, 1.1, 0.05, -0.4, 0.9};
        nn.setInput(input);
        nn.propagate();

        int const numConnections = nn.getNumConnections();
        vector<double> const dGdxyz = {1.0, -0.5, 0.25, 2.0, -1.5, 0.75};
        vector<double> dFdc(numConnections);
        nn.calculateDFdc(dFdc.data(), dGdxyz.data());

        vector<double> sum(numConnections, 0.0);
        vector<double> unit(6, 0.0);
        vector<double> dFdcUnit(numConnections);
        for (size_t i = 0; i < unit.size(); ++i)
        {
            unit.assign(6, 0.0);
            unit.at(i) = 1.0;
            nn.calculateDFdc(dFdcUnit.data(), unit.data());
            for (int j = 0; j < numConnections; ++j)
            {
                sum.at(j) += dGdxyz.at(i) * dFdcUnit.at(j);
            }
        }
        for (int j = 0; j < numConnections; ++j)
        {
            BOOST_TEST_INFO("connection " << j);
            BOOST_REQUIRE_SMALL(dFdc.at(j) - sum.at(j), accuracy);
        }
    }
}

BOOST_AUTO_TEST_CASE(ModifyConnections_ZeroBias)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    nn.modifyConnections(NeuralNetwork::MS_ZEROBIAS);

    vector<double> out(n);
    nn.getConnections(out.data());
    // Layer 2->3 weights: 2*3=6 (idx 0-5), biases: 3 (idx 6-8).
    // Layer 3->4(out) weights: 3*1=3 (idx 9-11), bias: 1 (idx 12).
    int const biasIdx[] = {6, 7, 8, 12};
    for (int idx : biasIdx)
    {
        BOOST_REQUIRE_SMALL(out.at(idx), accuracy);
    }
    for (int i = 0; i < n; ++i)
    {
        bool const isBias = (i == 6 || i == 7 || i == 8 || i == 12);
        if (!isBias)
        {
            BOOST_REQUIRE_SMALL(out.at(i) - 1.0, accuracy);
        }
    }
}

BOOST_AUTO_TEST_CASE(ModifyConnections_ZeroOutputWeights)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    nn.modifyConnections(NeuralNetwork::MS_ZEROOUTPUTWEIGHTS);

    vector<double> out(n);
    nn.getConnections(out.data());
    // Output layer weights are idx 9-11 (see layout above).
    int const outWeightIdx[] = {9, 10, 11};
    for (int idx : outWeightIdx)
    {
        BOOST_REQUIRE_SMALL(out.at(idx), accuracy);
    }
    for (int i = 0; i < n; ++i)
    {
        bool const isOutWeight = (i == 9 || i == 10 || i == 11);
        if (!isOutWeight)
        {
            BOOST_REQUIRE_SMALL(out.at(i) - 1.0, accuracy);
        }
    }
}

BOOST_AUTO_TEST_CASE(ModifyConnections_FanIn_OnlyScalesTanhLayers)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    nn.modifyConnections(NeuralNetwork::MS_FANIN);

    vector<double> out(n);
    nn.getConnections(out.data());
    // Hidden layer (TANH) weights idx 0-5: divided by sqrt(numNeuronsPrevLayer=2).
    double const expectedHidden = 1.0 / sqrt(2.0);
    for (int i = 0; i < 6; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - expectedHidden, accuracy);
    }
    // Output layer (IDENTITY) weights idx 9-11: unaffected.
    for (int i = 9; i < 12; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - 1.0, accuracy);
    }
}

BOOST_AUTO_TEST_CASE(ModifyConnections_GlorotBengio_OnlyScalesTanhLayers)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    nn.modifyConnections(NeuralNetwork::MS_GLOROTBENGIO);

    vector<double> out(n);
    nn.getConnections(out.data());
    // Hidden layer (TANH) weights idx 0-5: *= sqrt(6/(nIn=2 + nOut=3)).
    double const expectedHidden = sqrt(6.0 / 5.0);
    for (int i = 0; i < 6; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - expectedHidden, accuracy);
    }
    // Output layer (IDENTITY) weights idx 9-11: unaffected.
    for (int i = 9; i < 12; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - 1.0, accuracy);
    }
}

BOOST_AUTO_TEST_CASE(ModifyConnections_PreconditionOutput)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    double const mean = 2.0;
    double const sigma = 3.0;
    nn.modifyConnections(NeuralNetwork::MS_PRECONDITIONOUTPUT, mean, sigma);

    vector<double> out(n);
    nn.getConnections(out.data());
    // Output weights idx 9-11: *= sigma.
    for (int i = 9; i < 12; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - sigma, accuracy);
    }
    // Output bias idx 12: += mean.
    BOOST_REQUIRE_SMALL(out.at(12) - (1.0 + mean), accuracy);
    // Hidden layer untouched.
    for (int i = 0; i < 9; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - 1.0, accuracy);
    }
}

BOOST_AUTO_TEST_CASE(ModifyConnections_NguyenWidrow_AllOutputWeightsHalved)
{
    // Nguyen-Widrow initialization is documented (and, by its formula,
    // intended) to halve every weight feeding into the output neuron(s).
    // With more than one neuron in the last hidden layer, this currently
    // only holds for the very first incoming weight -- see the
    // corresponding fix in NeuralNetwork.cpp (loop bound
    // outputLayer->numNeurons should be outputLayer->numNeuronsPrevLayer).
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    nn.modifyConnections(NeuralNetwork::MS_NGUYENWIDROW);

    vector<double> out(n);
    nn.getConnections(out.data());

    // Hidden layer (idx 0-5, TANH, numNeuronsPrevLayer=2, numNeurons=3):
    // beta = 0.7 * 3^(1/2); each neuron's 2 weights start at 1.0, so
    // ||w|| = sqrt(2) before scaling.
    double const beta = 0.7 * pow(3.0, 1.0 / 2.0);
    double const expectedHiddenWeight = 1.0 * (beta / sqrt(2.0)) * 2.0;
    double const expectedHiddenBias   = 1.0 * beta * 2.0;
    for (int i = 0; i < 6; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - expectedHiddenWeight, accuracyNumeric);
    }
    for (int i = 6; i < 9; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - expectedHiddenBias, accuracyNumeric);
    }

    // Output layer weights idx 9-11 (3 hidden neurons -> 1 output neuron):
    // every one of them should be halved, not just the first.
    for (int i = 9; i < 12; ++i)
    {
        BOOST_REQUIRE_SMALL(out.at(i) - 0.5, accuracy);
    }
}

BOOST_AUTO_TEST_CASE(InitializeConnectionsRandomUniform_RangeAndReproducibility)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};

    NeuralNetwork nn1(numLayers, numNeuronsPerLayer, af);
    NeuralNetwork nn2(numLayers, numNeuronsPerLayer, af);

    nn1.initializeConnectionsRandomUniform(2024);
    nn2.initializeConnectionsRandomUniform(2024);

    int const n = nn1.getNumConnections();
    vector<double> c1(n);
    vector<double> c2(n);
    nn1.getConnections(c1.data());
    nn2.getConnections(c2.data());

    for (int i = 0; i < n; ++i)
    {
        BOOST_REQUIRE(c1.at(i) >= -1.0 && c1.at(i) <= 1.0);
        // Same seed must reproduce the same sequence.
        BOOST_REQUIRE_SMALL(c1.at(i) - c2.at(i), accuracy);
    }
}

BOOST_AUTO_TEST_CASE(NeuronStatistics_AccumulateAndReset)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    double input1[2] = {1.0, 3.0};
    nn.setInput(input1);
    double input2[2] = {-2.0, 5.0};
    nn.setInput(input2);

    int const numNeurons = nn.getNumNeurons();
    vector<long> count(numNeurons);
    vector<double> minV(numNeurons);
    vector<double> maxV(numNeurons);
    vector<double> sumV(numNeurons);
    vector<double> sum2V(numNeurons);
    nn.getNeuronStatistics(count.data(), minV.data(), maxV.data(),
                           sumV.data(), sum2V.data());

    // Input neuron 0 saw values 1.0 and -2.0.
    BOOST_REQUIRE_EQUAL(count.at(0), 2);
    BOOST_REQUIRE_SMALL(minV.at(0) - (-2.0), accuracy);
    BOOST_REQUIRE_SMALL(maxV.at(0) - 1.0, accuracy);
    BOOST_REQUIRE_SMALL(sumV.at(0) - (1.0 + (-2.0)), accuracy);
    BOOST_REQUIRE_SMALL(sum2V.at(0) - (1.0 * 1.0 + 2.0 * 2.0), accuracy);
    // Input neuron 1 saw values 3.0 and 5.0.
    BOOST_REQUIRE_EQUAL(count.at(1), 2);
    BOOST_REQUIRE_SMALL(minV.at(1) - 3.0, accuracy);
    BOOST_REQUIRE_SMALL(maxV.at(1) - 5.0, accuracy);

    nn.resetNeuronStatistics();
    nn.getNeuronStatistics(count.data(), minV.data(), maxV.data(),
                           sumV.data(), sum2V.data());
    for (int i = 0; i < numNeurons; ++i)
    {
        BOOST_REQUIRE_EQUAL(count.at(i), 0);
        BOOST_REQUIRE_SMALL(sumV.at(i), accuracy);
        BOOST_REQUIRE_SMALL(sum2V.at(i), accuracy);
    }
}

BOOST_AUTO_TEST_CASE(ActivationFromString_KnownLetters_CorrectValues)
{
    BOOST_REQUIRE_EQUAL(activationFromString("l"), NeuralNetwork::AF_IDENTITY);
    BOOST_REQUIRE_EQUAL(activationFromString("t"), NeuralNetwork::AF_TANH);
    BOOST_REQUIRE_EQUAL(activationFromString("s"), NeuralNetwork::AF_LOGISTIC);
    BOOST_REQUIRE_EQUAL(activationFromString("p"), NeuralNetwork::AF_SOFTPLUS);
    BOOST_REQUIRE_EQUAL(activationFromString("r"), NeuralNetwork::AF_RELU);
    BOOST_REQUIRE_EQUAL(activationFromString("g"), NeuralNetwork::AF_GAUSSIAN);
    BOOST_REQUIRE_EQUAL(activationFromString("c"), NeuralNetwork::AF_COS);
    BOOST_REQUIRE_EQUAL(activationFromString("S"), NeuralNetwork::AF_REVLOGISTIC);
    BOOST_REQUIRE_EQUAL(activationFromString("e"), NeuralNetwork::AF_EXP);
    BOOST_REQUIRE_EQUAL(activationFromString("h"), NeuralNetwork::AF_HARMONIC);
}

BOOST_AUTO_TEST_CASE(ActivationFromString_UnknownLetter_Throws)
{
    BOOST_REQUIRE_THROW(activationFromString("z"), runtime_error);
}

BOOST_AUTO_TEST_CASE(Info_NonEmpty)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    vector<string> const info = nn.info();
    BOOST_REQUIRE(!info.empty());

    BOOST_REQUIRE(nn.getMemoryUsage() > 0);
}

BOOST_FIXTURE_TEST_CASE(WriteConnections_ProducesNonEmptyFile, FixtureRepairDir)
{
    int const numLayers = 3;
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(numLayers, numNeuronsPerLayer, af);

    int const n = nn.getNumConnections();
    vector<double> ones(n, 1.0);
    nn.setConnections(ones.data());

    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    ofstream file("weights.dat");
    nn.writeConnections(file);
    file.close();

    BOOST_REQUIRE(bfs::exists("weights.dat"));
    BOOST_REQUIRE(bfs::file_size("weights.dat") > 0);

    bfs::current_path("..");
    bfs::remove_all("test");
}

#ifndef N2P2_GPU
BOOST_AUTO_TEST_CASE(Gpu_CpuBuild_NotAvailable)
{
    int const numNeuronsPerLayer[3] = {2, 3, 1};
    NeuralNetwork::ActivationFunction const af[3] =
        {NeuralNetwork::AF_IDENTITY,
         NeuralNetwork::AF_TANH,
         NeuralNetwork::AF_IDENTITY};
    NeuralNetwork nn(3, numNeuronsPerLayer, af);
    BOOST_REQUIRE(!nn.isGpuEnabled());
    BOOST_REQUIRE_NO_THROW(nn.setGpuEnabled(false));
    BOOST_REQUIRE_THROW(nn.setGpuEnabled(true), runtime_error);
    double input[2] = {0.1, 0.2};
    double output = 0.0;
    BOOST_REQUIRE_THROW(nn.propagateBatchGpu(1, input, &output, nullptr),
                        runtime_error);
}
#endif

BOOST_AUTO_TEST_SUITE_END()

#ifdef N2P2_GPU
// Tests of the GPU twin of NeuralNetwork against the CPU implementation, in
// the same binary.

// Network architecture for the GPU tests.
struct Architecture
{
    vector<int>                               numNeurons;
    vector<NeuralNetwork::ActivationFunction> activations;
    bool                                      normalize;
};

ostream& operator<<(ostream& os, Architecture const& a)
{
    for (size_t l = 0; l < a.numNeurons.size(); ++l)
    {
        os << a.numNeurons.at(l) << "(" << a.activations.at(l) << ") ";
    }
    return os << (a.normalize ? "normalized" : "");
}

// Small, medium and deep networks, every activation function in a hidden
// layer and as output activation, with and without normalized neurons.
vector<Architecture> gpuArchitectures()
{
    typedef NeuralNetwork NN;
    vector<Architecture> v;
    v.push_back({{5, 8, 1}, {NN::AF_IDENTITY, NN::AF_TANH, NN::AF_IDENTITY},
                 false});
    for (int f = NN::AF_IDENTITY; f <= NN::AF_HARMONIC; ++f)
    {
        NN::ActivationFunction const af =
            static_cast<NN::ActivationFunction>(f);
        v.push_back({{6, 10, 7, 1},
                     {NN::AF_IDENTITY, af, NN::AF_TANH, NN::AF_IDENTITY},
                     false});
        v.push_back({{6, 9, 1}, {NN::AF_IDENTITY, NN::AF_SOFTPLUS, af},
                     true});
    }
    v.push_back({{4, 6, 5, 3, 1},
                 {NN::AF_IDENTITY, NN::AF_TANH, NN::AF_LOGISTIC,
                  NN::AF_SOFTPLUS, NN::AF_IDENTITY},
                 true});
    return v;
}

// Network with deterministic, irregular connections.
unique_ptr<NeuralNetwork> makeNetwork(Architecture const& a)
{
    unique_ptr<NeuralNetwork> nn(new NeuralNetwork(a.numNeurons.size(),
                                                   a.numNeurons.data(),
                                                   a.activations.data()));
    nn->setNormalizeNeurons(a.normalize);
    vector<double> c(nn->getNumConnections());
    for (size_t i = 0; i < c.size(); ++i) c.at(i) = 0.8 * sin(1.3 * i + 0.4);
    nn->setConnections(c.data());
    return nn;
}

// Inputs (numIn x numInputs), one input scaled up to reach the overflow
// branches of the exponential activation functions.
vector<double> makeInputs(size_t numIn, size_t numInputs, size_t seed)
{
    vector<double> in(numIn * numInputs);
    for (size_t i = 0; i < in.size(); ++i)
    {
        in.at(i) = 1.5 * sin(0.7 * i + 0.3 * seed);
    }
    for (size_t k = 0; k < numIn; ++k) in.at(k) *= 40.0;
    return in;
}

// Propagates the inputs through cpu (one by one) and gpu (as a batch) and
// requires the same outputs and derivatives. Round-off scales with the
// largest magnitude of an input's results, so the absolute tolerance does
// too (near-zero derivatives are differences of large terms).
void requireSamePropagation(NeuralNetwork& cpu,
                            NeuralNetwork& gpu,
                            size_t         numIn,
                            size_t         numInputs,
                            size_t         seed)
{
    vector<double> const in = makeInputs(numIn, numInputs, seed);
    vector<double> outGpu(numInputs);
    vector<double> dEdGGpu(numIn * numInputs);
    gpu.propagateBatchGpu(numInputs, in.data(), outGpu.data(), dEdGGpu.data());
    vector<double> dEdG(numIn);
    for (size_t t = 0; t < numInputs; ++t)
    {
        double out = 0.0;
        cpu.setInput(&(in.at(t * numIn)));
        cpu.propagate();
        cpu.getOutput(&out);
        cpu.calculateDEdG(dEdG.data());
        double scale = max(1.0, fabs(out));
        for (double d : dEdG) scale = max(scale, fabs(d));
        Tolerance const tolerance(gpuRelativeEvaluation * scale,
                                  gpuRelativeEvaluation);
        BOOST_TEST_INFO("input " << t);
        REQUIRE_CLOSE_TO_REFERENCE(out, outGpu.at(t), tolerance);
        for (size_t k = 0; k < numIn; ++k)
        {
            BOOST_TEST_INFO("input " << t << ", dEdG " << k);
            REQUIRE_CLOSE_TO_REFERENCE(dEdG.at(k),
                                       dEdGGpu.at(t * numIn + k),
                                       tolerance);
        }
    }
}

// Requires the same neuron statistics of both networks.
void requireSameStatistics(NeuralNetwork const& cpu, NeuralNetwork const& gpu)
{
    size_t const n = cpu.getNumNeurons();
    vector<long> countCpu(n), countGpu(n);
    vector<double> minCpu(n), maxCpu(n), sumCpu(n), sum2Cpu(n);
    vector<double> minGpu(n), maxGpu(n), sumGpu(n), sum2Gpu(n);
    cpu.getNeuronStatistics(countCpu.data(), minCpu.data(), maxCpu.data(),
                            sumCpu.data(), sum2Cpu.data());
    gpu.getNeuronStatistics(countGpu.data(), minGpu.data(), maxGpu.data(),
                            sumGpu.data(), sum2Gpu.data());
    for (size_t i = 0; i < n; ++i)
    {
        // The sum can cancel: round-off scales with count * largest value.
        double const scale = max(1.0, max(fabs(minCpu.at(i)),
                                          fabs(maxCpu.at(i))));
        Tolerance const tolerance(gpuRelativeEvaluation * scale,
                                  gpuRelativeEvaluation);
        Tolerance const toleranceSum(gpuRelativeEvaluation * scale
                                     * max(1L, countCpu.at(i)),
                                     gpuRelativeEvaluation);
        BOOST_TEST_INFO("neuron " << i);
        BOOST_REQUIRE_EQUAL(countCpu.at(i), countGpu.at(i));
        REQUIRE_CLOSE_TO_REFERENCE(minCpu.at(i), minGpu.at(i), tolerance);
        REQUIRE_CLOSE_TO_REFERENCE(maxCpu.at(i), maxGpu.at(i), tolerance);
        REQUIRE_CLOSE_TO_REFERENCE(sumCpu.at(i), sumGpu.at(i), toleranceSum);
        REQUIRE_CLOSE_TO_REFERENCE(sum2Cpu.at(i), sum2Gpu.at(i), tolerance);
    }
}

// Requires equal vectors, with round-off scaled by their largest entry.
void requireCloseScaled(vector<double> const& cpu,
                        vector<double> const& gpu,
                        string const&         what)
{
    BOOST_REQUIRE_EQUAL(cpu.size(), gpu.size());
    double scale = 1.0;
    for (double v : cpu) scale = max(scale, fabs(v));
    Tolerance const tolerance(gpuRelativeEvaluation * scale,
                              gpuRelativeEvaluation);
    for (size_t i = 0; i < cpu.size(); ++i)
    {
        BOOST_TEST_INFO(what << " " << i);
        REQUIRE_CLOSE_TO_REFERENCE(cpu.at(i), gpu.at(i), tolerance);
    }
}

BOOST_AUTO_TEST_SUITE(GpuTests)

BOOST_DATA_TEST_CASE(CalculateDEdcBatch_Architectures_MatchesCpuSum,
                     bdata::make(gpuArchitectures()),
                     architecture)
{
    unique_ptr<NeuralNetwork> cpu = makeNetwork(architecture);
    unique_ptr<NeuralNetwork> gpu = makeNetwork(architecture);
    gpu->setGpuEnabled(true);
    size_t const numIn = architecture.numNeurons.front();
    size_t const numConnections = cpu->getNumConnections();
    size_t const numInputs = 23;
    vector<double> const in = makeInputs(numIn, numInputs, 5);

    vector<double> dEdcSum(numConnections, 0.0);
    vector<double> dEdc(numConnections);
    vector<double> out(numInputs);
    for (size_t t = 0; t < numInputs; ++t)
    {
        cpu->setInput(&(in.at(t * numIn)));
        cpu->propagate();
        cpu->getOutput(&(out.at(t)));
        cpu->calculateDEdc(dEdc.data());
        for (size_t j = 0; j < numConnections; ++j) dEdcSum.at(j) += dEdc.at(j);
    }

    vector<double> outGpu(numInputs);
    vector<double> dEdcSumGpu(numConnections);
    gpu->calculateDEdcBatchGpu(numInputs, in.data(), outGpu.data(),
                               dEdcSumGpu.data());
    requireCloseScaled(out, outGpu, "output");
    requireCloseScaled(dEdcSum, dEdcSumGpu, "dEdc");
    requireSameStatistics(*cpu, *gpu);
}

BOOST_DATA_TEST_CASE(CalculateDFdcBatch_Architectures_MatchesCpuSum,
                     bdata::make(gpuArchitectures()),
                     architecture)
{
    unique_ptr<NeuralNetwork> cpu = makeNetwork(architecture);
    unique_ptr<NeuralNetwork> gpu = makeNetwork(architecture);
    gpu->setGpuEnabled(true);
    size_t const numIn = architecture.numNeurons.front();
    size_t const numConnections = cpu->getNumConnections();
    size_t const numInputs = 17;
    vector<double> const in = makeInputs(numIn, numInputs, 7);
    // Derivatives of the inputs with respect to a coordinate.
    vector<double> dGdx(numIn * numInputs);
    for (size_t i = 0; i < dGdx.size(); ++i) dGdx.at(i) = cos(0.45 * i + 1.0);

    vector<double> dFdcSum(numConnections, 0.0);
    vector<double> dFdc(numConnections);
    vector<double> out(numInputs);
    vector<double> dEdG(numIn * numInputs);
    for (size_t t = 0; t < numInputs; ++t)
    {
        cpu->setInput(&(in.at(t * numIn)));
        cpu->propagate();
        cpu->getOutput(&(out.at(t)));
        cpu->calculateDEdG(&(dEdG.at(t * numIn)));
        cpu->calculateDFdc(dFdc.data(), &(dGdx.at(t * numIn)));
        for (size_t j = 0; j < numConnections; ++j) dFdcSum.at(j) += dFdc.at(j);
    }

    vector<double> outGpu(numInputs);
    vector<double> dEdGGpu(numIn * numInputs);
    vector<double> dFdcSumGpu(numConnections);
    gpu->calculateDFdcBatchGpu(numInputs, in.data(), dGdx.data(),
                               outGpu.data(), dEdGGpu.data(),
                               dFdcSumGpu.data());
    requireCloseScaled(out, outGpu, "output");
    requireCloseScaled(dEdG, dEdGGpu, "dEdG");
    requireCloseScaled(dFdcSum, dFdcSumGpu, "dFdc");
    requireSameStatistics(*cpu, *gpu);

    // Without dEdG output.
    BOOST_REQUIRE_NO_THROW(gpu->calculateDFdcBatchGpu(numInputs,
                                                      in.data(),
                                                      dGdx.data(),
                                                      outGpu.data(),
                                                      nullptr,
                                                      dFdcSumGpu.data()));
    requireCloseScaled(dFdcSum, dFdcSumGpu, "dFdc without dEdG");
}

BOOST_AUTO_TEST_CASE(CalculateDEdcBatch_Weighted_MatchesCpuWeightedSum)
{
    // Weighted sums (charge Jacobian) without updating the statistics.
    Architecture const a = gpuArchitectures().back();
    unique_ptr<NeuralNetwork> cpu = makeNetwork(a);
    unique_ptr<NeuralNetwork> gpu = makeNetwork(a);
    gpu->setGpuEnabled(true);
    size_t const numIn = a.numNeurons.front();
    size_t const numConnections = cpu->getNumConnections();
    size_t const numInputs = 31;
    vector<double> const in = makeInputs(numIn, numInputs, 9);
    vector<double> weights(numInputs);
    for (size_t t = 0; t < numInputs; ++t) weights.at(t) = sin(2.1 * t + 0.2);

    vector<double> sum(numConnections, 0.0);
    vector<double> dEdc(numConnections);
    for (size_t t = 0; t < numInputs; ++t)
    {
        cpu->setInput(&(in.at(t * numIn)));
        cpu->propagate();
        cpu->calculateDEdc(dEdc.data());
        for (size_t j = 0; j < numConnections; ++j)
        {
            sum.at(j) += weights.at(t) * dEdc.at(j);
        }
    }
    // Statistics of the CPU network as after one propagation of the batch.
    vector<double> out(numInputs);
    gpu->propagateBatchGpu(numInputs, in.data(), out.data(), nullptr);

    vector<double> sumGpu(numConnections);
    gpu->calculateDEdcBatchGpu(numInputs, in.data(), nullptr, sumGpu.data(),
                               weights.data(), false);
    requireCloseScaled(sum, sumGpu, "weighted dEdc");
    requireSameStatistics(*cpu, *gpu);
}

BOOST_AUTO_TEST_CASE(CalculateJacobianBatch_NoInputs_Zero)
{
    unique_ptr<NeuralNetwork> gpu = makeNetwork(gpuArchitectures().front());
    gpu->setGpuEnabled(true);
    vector<double> d(gpu->getNumConnections(), 1.0);
    gpu->calculateDEdcBatchGpu(0, nullptr, nullptr, d.data());
    for (double v : d) BOOST_REQUIRE_EQUAL(v, 0.0);
    d.assign(d.size(), 1.0);
    gpu->calculateDFdcBatchGpu(0, nullptr, nullptr, nullptr, nullptr,
                               d.data());
    for (double v : d) BOOST_REQUIRE_EQUAL(v, 0.0);
}

BOOST_DATA_TEST_CASE(PropagateBatch_Architectures_MatchesCpu,
                     bdata::make(gpuArchitectures()),
                     architecture)
{
    unique_ptr<NeuralNetwork> cpu = makeNetwork(architecture);
    unique_ptr<NeuralNetwork> gpu = makeNetwork(architecture);
    gpu->setGpuEnabled(true);
    BOOST_REQUIRE(gpu->isGpuEnabled());
    size_t const numIn = architecture.numNeurons.front();
    // Growing and shrinking batches reuse and grow the device buffers.
    size_t seed = 0;
    for (size_t numInputs : {3, 40, 1, 97})
    {
        BOOST_TEST_INFO("batch of " << numInputs);
        requireSamePropagation(*cpu, *gpu, numIn, numInputs, seed++);
    }
    requireSameStatistics(*cpu, *gpu);
}

BOOST_AUTO_TEST_CASE(PropagateBatch_ConnectionsChanged_FollowsCpu)
{
    // The GPU twin must follow every change of the connections.
    Architecture const a = gpuArchitectures().front();
    unique_ptr<NeuralNetwork> cpu = makeNetwork(a);
    unique_ptr<NeuralNetwork> gpu = makeNetwork(a);
    gpu->setGpuEnabled(true);
    size_t const numIn = a.numNeurons.front();

    vector<double> c(cpu->getNumConnections());
    for (size_t i = 0; i < c.size(); ++i) c.at(i) = 0.5 * cos(0.9 * i);
    cpu->setConnections(c.data());
    gpu->setConnections(c.data());
    requireSamePropagation(*cpu, *gpu, numIn, 10, 1);

    cpu->modifyConnections(NeuralNetwork::MS_ZEROBIAS);
    gpu->modifyConnections(NeuralNetwork::MS_ZEROBIAS);
    requireSamePropagation(*cpu, *gpu, numIn, 10, 2);

    cpu->modifyConnections(NeuralNetwork::MS_PRECONDITIONOUTPUT, 0.3, 2.0);
    gpu->modifyConnections(NeuralNetwork::MS_PRECONDITIONOUTPUT, 0.3, 2.0);
    requireSamePropagation(*cpu, *gpu, numIn, 10, 3);

    cpu->setNormalizeNeurons(true);
    gpu->setNormalizeNeurons(true);
    BOOST_REQUIRE(gpu->isGpuEnabled());
    requireSamePropagation(*cpu, *gpu, numIn, 10, 4);
}

BOOST_AUTO_TEST_CASE(PropagateBatch_NoInputs_NothingChanges)
{
    Architecture const a = gpuArchitectures().front();
    unique_ptr<NeuralNetwork> cpu = makeNetwork(a);
    unique_ptr<NeuralNetwork> gpu = makeNetwork(a);
    gpu->setGpuEnabled(true);
    BOOST_REQUIRE_NO_THROW(gpu->propagateBatchGpu(0, nullptr, nullptr,
                                                  nullptr));
    requireSameStatistics(*cpu, *gpu);
}

BOOST_AUTO_TEST_CASE(PropagateBatch_GpuNotEnabled_Throws)
{
    unique_ptr<NeuralNetwork> nn = makeNetwork(gpuArchitectures().front());
    double input[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    double output = 0.0;
    BOOST_REQUIRE_THROW(nn->propagateBatchGpu(1, input, &output, nullptr),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(SetGpuEnabled_NoVisibleDevice_Throws)
{
    BOOST_REQUIRE_EQUAL(runWithoutGpu("WithoutGpu"), 0);
}

BOOST_AUTO_TEST_SUITE_END()

// Run only by runWithoutGpu() above, in a process without visible devices.
BOOST_AUTO_TEST_SUITE(WithoutGpu, *boost::unit_test::disabled())

BOOST_AUTO_TEST_CASE(SetGpuEnabled_NoVisibleDevice_Throws)
{
    unique_ptr<NeuralNetwork> nn = makeNetwork(gpuArchitectures().front());
    BOOST_REQUIRE_THROW(nn->setGpuEnabled(true), runtime_error);
    BOOST_REQUIRE(!nn->isGpuEnabled());
}

BOOST_AUTO_TEST_SUITE_END()
#endif
