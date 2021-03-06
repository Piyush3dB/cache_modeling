#include <iostream>
#include "woof.h"
#include "rdtsc.h"

const uint32_t ARRAY_SIZE = 1500;
uint32_t CACHE_SIZE = 550;
const uint32_t ASSOCIATIVITY = 16;
const uint32_t NUM_SPARSE_POINTS = 128; // age coarsening

void solve(
    const woof::vec& rdd,               // reuse distance distribution
    woof::Rank::Factory rankingFunction) { // ie, replacement policy

    uint64_t start, end;                // for timing
    
    // all cache lines belong to the same class (ie, no
    // classification; see tech report)
    woof::Unclassifier unclassifier;
    
    // instantiate the model
    woof::Cache model(
        CACHE_SIZE,
        ASSOCIATIVITY,
        rdd,
        unclassifier,
        rankingFunction);

    start = rdtsc();
    
    model.solve();
    
    end = rdtsc();

    // print results
    std::cout << "Model solved hit rate of " << model.hitRate()
              << " in " << (end - start) << " cycles (full solution).\n";
/*

    // instantiate the sparse model after coarsening regions
    woof::uvec sparsePoints = woof::sparse::divide(
        model.cl(0).hit + model.cl(0).evict,
        NUM_SPARSE_POINTS / 2,
        NUM_SPARSE_POINTS / 2);
    
    woof::Cache sparseModel(
        CACHE_SIZE,
        ASSOCIATIVITY,
        rdd,
        unclassifier,
        rankingFunction,
        &sparsePoints,
        &model.cl(0).age);

    start = rdtsc();
    
    sparseModel.solve();

    end = rdtsc();
    
    // print results
    std::cout << "Model solved hit rate of " << sparseModel.hitRate()
              << " in " << (end - start) << " cycles (sparse solution).\n";
*/
}

woof::vec rddRandom() {
    woof::vec rdd(ARRAY_SIZE * 8, arma::fill::zeros);

    for (uint32_t a = 1; a < rdd.n_elem; a++) {
        rdd(a) = (1. / ARRAY_SIZE) * std::pow((ARRAY_SIZE - 1.) / ARRAY_SIZE, a - 1);
    }

    return rdd;
}

woof::vec rddRead(char* filename) {
    std::ifstream rdd_file;
    int d;
    float p;
    int n = 1200;
    rdd_file.open(filename);
    woof::vec rdd(n, arma::fill::zeros);
    while (!rdd_file.eof()) {
        rdd_file >> d >> p;
        rdd(d) = p;
    }
    rdd_file.close();

    return rdd;
}

void dumpArray(woof::vec array, char* filename) {
    std::ofstream outfile;
    outfile.open(filename);
    int n = array.n_elem;
    for (int i=0;i<n;i++) {
        outfile << array(i) << std::endl;
    }
    outfile.close();

    return;
}

woof::vec rddScan() {
    woof::vec rdd(CACHE_SIZE * 4, arma::fill::zeros);

    rdd(CACHE_SIZE/2) = 0.25;
    rdd(CACHE_SIZE*1.2) = 0.25;
    rdd(CACHE_SIZE*3) = 0.5;

    return rdd;
}

woof::vec rddStack() {
    woof::vec rdd(ARRAY_SIZE * 3, arma::fill::zeros);

    for (uint32_t a = 1; a <= ARRAY_SIZE * 2; a++) {
        rdd(a) = 1. / (2 * ARRAY_SIZE);
    }

    return rdd;
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        CACHE_SIZE = atoi(argv[1]);
    }
    std::cout<< "size: " << CACHE_SIZE << std::endl;

    auto rdd = rddRead("rdd.out");
    assert(std::fabs(arma::accu(rdd) - 1.) < 1e-2);

    // solve for diff repl policies

    // note: lru converges slowly; for an LRU model, it makes sense
    // sense to specialize the model by eliminating ranks from the
    // solution (since rank = age), which speeds convergence
    std::cout << "Rank form file:\n";
    solve(rdd, woof::repl::rankFile);

    //std::cout << "ETTR:\n";
    //solve(rdd, woof::repl::ExpectedTimeToReuse);

    // std::cout << "PDP:\n";
    // solve(rdd, woof::Rank::factory<woof::repl::PDP>);

    // std::cout << "IRGD:\n";
    // solve(rdd, woof::repl::HarmonicExpectedTimeToReuse);

    return 0;
}
