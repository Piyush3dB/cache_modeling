#include <cassert>
#include <armadillo>

// warn if solver terminates at max iterations
#define WOOF_WARN_MAX_ITERATIONS (false)

// ignore associativity parameter, use faster exponentiation with
// associativity = 16
#define WOOF_FAST_ASSOCIATIVITY (true)

// smaller floats lead to small performance improvement with little
// accuracy loss
#define WOOF_USE_FLOAT32 (true)

#define info(...)                               \
    do {                                        \
        fprintf(stderr, __VA_ARGS__);           \
        fputc('\n', stderr);                    \
    } while(0)

namespace woof {

#if WOOF_USE_FLOAT32

typedef float fp_t;
typedef arma::fvec vec;
typedef arma::fmat mat;
typedef arma::uvec uvec;
typedef arma::uword uword;

#else

typedef double fp_t;
typedef arma::vec vec;
typedef arma::mat mat;
typedef arma::uvec uvec;
typedef arma::uword uword;

#endif

// Age coarsening functions.
namespace sparse {
uvec divide(const vec& vec, uword nsplit, uword nadaptive);
vec compress(const vec& vec, const uvec& sparsePoints, bool sum);
vec expand(const vec& vec, const uvec& sparsePoints, bool spread);
uvec none(uword npoints);
vec rank(const vec& fullRanks, const uvec& sparse, const vec& fullAgeDist); // compute coarsened ranks using age probabilities as weights
} // namespace sparse

using std::vector;

const int MIN_ITERATIONS = 10;
const int MAX_ITERATIONS = 50;
const fp_t TERMINATION_CONDITION = 1e-3;
const fp_t DRAG = 3.0;

class Cache;
class Class;

// A ranking function.
//
// Essentially a vector with extra methods to allow indexing by a
// floating-point value (quantized into buckets) and factory
// methods to make the ranking functions for different replacement
// policies. See woof_repl.h.
class Rank : public vec {
    public:
        Rank() : min(std::numeric_limits<fp_t>::quiet_NaN()), max(std::numeric_limits<fp_t>::quiet_NaN()) {}
        virtual ~Rank() {}

        Rank(uword _n_elem, fp_t _min, fp_t _max)
                : vec(_n_elem, arma::fill::zeros)
                , min(_min)
                , max(_max) {
        }

        Rank(const vec& _vals)
                : vec(_vals)
                , min(_vals.min())
                , max(_vals.max()) {
        }

        Rank(Rank&& that)
                : vec(std::move(that))
                , min(that.min)
                , max(that.max) {
        }

        Rank(const Rank& that)
                : vec(that)
                , min(that.min)
                , max(that.max) { }

        Rank& operator=(const Rank& that) {
            this->min = that.min;
            this->max = that.max;
            vec::operator=(that);
            return *this;
        }

        static Rank like(const Rank& orig) {
            return Rank(orig.n_elem, orig.min, orig.max);
        }

        fp_t min, max;

        inline const uword index(fp_t r) const {
            unsigned int ri = (n_elem-1) * (r - min) / (max - min);
            assert(ri < n_elem);
            return ri;
        }

        inline fp_t& at(fp_t r) {
            return vec::operator()(index(r));
        }

        inline const fp_t& at(fp_t r) const {
            return vec::operator()(index(r));
        }

        virtual void refresh(const Class& cl) {
            // Nothing by default
        }

        typedef Rank* (*Factory)(const uword n_elem, const Class& cl);

        template<class Child>
        static Rank* factory(const uword n_elem, const Class& cl) {
            auto child = new Child(n_elem);
            child->refresh(cl);
            return child;
        }
}; // class Rank

// The model breaks cache lines into Classes, eg reused vs
// non-reused lines. See tech report for details.
//
// Each class has a fraction of overall cache lines (prob), a hit
// rate for that class (hitRate), an access pattern (rd),
// replacement ranking function (rankfn), and model distributions
// for the class (age, hit, evict).
class Class {
    public:
        fp_t prob;
        fp_t hitRate;
        vec rd;
        vec age;
        vec hit;
        vec evict;
        Rank* rankfn;
        Cache* cache;
        Rank::Factory factory;

        Class(const uword n_elem, Rank::Factory _factory)
                : prob(0.)
                , hitRate(0.)
                , rd(n_elem, arma::fill::zeros)
                , age(n_elem, arma::fill::zeros)
                , hit(n_elem, arma::fill::zeros)
                , evict(n_elem, arma::fill::zeros)
                , rankfn(NULL)
                , cache(NULL)
                , factory(_factory) {
        }

        void reseed(const vec& _rd, fp_t _prob, Cache* _cache) {
            age = _prob - vec(arma::cumsum(_rd));
            age *= _prob / arma::sum(age);
            hit = 0.5 * _rd;
            evict = _rd - hit;
            hitRate = 0.5 * arma::sum(_rd);

            reset(_rd, _prob, _cache);
        }

        void reset(const vec& _rd, fp_t _prob, Cache* _cache) {
            assert((rankfn == NULL));
            cache = _cache;
            rd = _rd;
            refresh(_rd, _prob);
        }

        void refresh(const vec& _rd, fp_t _prob);

        void cleanup() {
            delete rankfn;
            rankfn = NULL;
        }

        void iterate(fp_t cacheHitRate, const Rank& cacheRankDist, const Rank& cacheMaxRankDist, const uvec& sparsePoints);
};

// Classifiers break lines into Classes.
class Classifier {
    public:
        virtual ~Classifier() {}

        virtual vector<Class> init(const vec& rd, Rank::Factory factory, Cache* cache) = 0;
        virtual void refresh(vector<Class>& classes, const vec& rd) = 0;
        virtual void cleanup(vector<Class>& classes) {
            for (auto& cl : classes) {
                cl.cleanup();
            }
        }
};

// Our cache model from "Modeling Cache Peformance Beyond LRU" by
// Nathan Beckmann and Daniel Sanchez in HPCA'16.
class Cache {
    public:
        uint32_t size;
        uint32_t assoc;

        // how to coarsen ages into regions for more efficient
        // solution
        uvec sparsePoints;

        // for coarsening ranks into sparse regions -- generally
        // the age distribution from the previous solution. the
        // weights are *not* coarsened.
        vec rankWeights;

    private:
        Classifier& classifier;
        vector<Class> classes;
        const vec& origRdPmf;
        Rank::Factory factory;
        uint32_t iterations;
        fp_t maxHitRate, minHitRate;
        const fp_t terminationCondition;

        Rank modelRank() {
            fp_t min = std::numeric_limits<fp_t>::max();
            fp_t max = std::numeric_limits<fp_t>::lowest();

            for (auto& cl : classes) {
                min = std::min(cl.rankfn->min, min);
                max = std::max(cl.rankfn->max, max);
            }

            Rank rank(sparsePoints.n_elem, min, max);
            fp_t cumrank = 0.;
            for (auto& cl : classes) {
                uword lastPoint = 0;
                for (uword i = 0; i < sparsePoints.n_elem; i++) {
                    auto r = (*cl.rankfn)(i);
                    fp_t weight = cl.age(i) * (sparsePoints(i) - lastPoint);
                    rank.at(r) += weight;
                    cumrank += weight;

                    // std::cerr << "modelRank -- i: " << i << ", rank: " << r << ", prob: " << rank.at(r) << ", cum: " << cumrank << ", from-to: " << lastPoint << "-" << sparsePoints(i) << std::endl;
                    lastPoint = sparsePoints(i);
                }
            }
            rank /= cumrank;
            // info("cumrank: %g", cumrank);
            return rank;
        }

        Rank modelMaxRank(const Rank& rank) {
            fp_t cumrank = 0.;
            fp_t cummaxrank = 0.;
            fp_t prevcummaxrank = 0.;

            auto maxrank = Rank::like(rank);
            for (uword r = 0; r < rank.n_elem; r++) {
                cumrank += rank(r);
#if WOOF_FAST_ASSOCIATIVITY
                cummaxrank = cumrank;     // r
                cummaxrank *= cummaxrank; // r^2
                cummaxrank *= cummaxrank; // r^4
                cummaxrank *= cummaxrank; // r^8
                cummaxrank *= cummaxrank; // r^16
#else
                cummaxrank = std::pow(cumrank, assoc);
#endif
                maxrank(r) = cummaxrank - prevcummaxrank;
                prevcummaxrank = cummaxrank;
            }

            assert(std::abs(cumrank - 1.) < 1e-4);
            return maxrank;
        }

    public:
        Cache(uint32_t _size,
              uint32_t _assoc,
              const vec& _rd,
              Classifier& _classifier,
              Rank::Factory _rankFactory,
              const uvec* _sparsePointsPtr = nullptr,
              const vec* _rankWeights = nullptr,
              fp_t _terminationCondition = TERMINATION_CONDITION)
                : size(_size)
                , assoc(_assoc)
                , classifier(_classifier)
                , origRdPmf(_rd)
                , factory(_rankFactory)
                , iterations(0)
                , terminationCondition(_terminationCondition) {
#if WOOF_FAST_ASSOCIATIVITY
            // When WOOF_FAST_ASSOCIATIVITY is set, the model uses a
            // faster method to raise numbers to the power of assoc
            // (eg, Eq 14 in the HPCA'16 paper). If you want to model
            // other associativities, either turn off the flag or
            // modify the code in modelMaxRank above. -nzb
            assert(assoc == 16);
#endif
            
            if (_sparsePointsPtr) {
                sparsePoints = *_sparsePointsPtr;
                assert(_rankWeights);
                rankWeights = *_rankWeights;
            } else {
                sparsePoints = sparse::none(origRdPmf.n_elem);
                rankWeights = vec(sparsePoints.n_elem, arma::fill::ones);
            }
            // std::cout << "RD PMF: " << pmfrd.rows(0,512).t() << std::endl;
            classes = classifier.init(origRdPmf, factory, this);
        }

        ~Cache() {
            cleanup();
        }

        uint32_t numClasses() const { return classes.size(); }
        const Class& cl(int index) const { return classes[index]; }
        fp_t hitRate() const { fp_t hr=0.; for (auto& cl : classes) hr += cl.hitRate; return hr; }
        uint32_t iters() const { return iterations; }

        fp_t checkCapacityConstraint(bool print=false) const {
            fp_t exphit = 0.;
            fp_t expev = 0.;

            for (uint32_t c = 0; c < numClasses(); c++) {
                for (uint32_t a = 0; a < cl(c).hit.n_elem; a++) {
                    exphit += a * cl(c).hit(a);
                    expev += a * cl(c).evict(a);
                }
            }

            if (print) {
                std::cout << "Constraints: " << exphit << " + " << expev << " = " << exphit + expev
                          << "\t" << (exphit+expev) / size << std::endl;
            }

            return (exphit + expev) / size;
        }

        void iterate() {
            classifier.refresh(classes, origRdPmf);
            auto pmfrank = modelRank();
            auto pmfmaxrank = modelMaxRank(pmfrank);

            for (auto& cl : classes) {
                cl.iterate(hitRate(), pmfrank, pmfmaxrank, sparsePoints);
            }
        }

    private:
        bool hasConverged(int iterations, fp_t& capacityConstraint, fp_t& hitRateConstraint) {
            // checkCapacityConstraint(true);
            fp_t hr = hitRate();

            if ((iterations % MIN_ITERATIONS) == MIN_ITERATIONS-1) {
                if (maxHitRate - minHitRate < terminationCondition) {
                    return true;
                } else {
                    maxHitRate = hr;
                    minHitRate = hr;
                }
            } else {
                maxHitRate = std::max(hr, maxHitRate);
                minHitRate = std::min(hr, minHitRate);
            }

            return false;
        }

    public:
        // This is the main method where the magic happens!
        void solve() {
            fp_t capacityConstraint = 0.;
            fp_t hitRateConstraint = 0.;

            for (auto& cl : classes) {
                cl.age = sparse::compress(cl.age, sparsePoints, false);
                cl.hit = sparse::compress(cl.hit, sparsePoints, true);
                cl.evict = sparse::compress(cl.evict, sparsePoints, true);
            }

            int i;
            for (i = 0; i < MAX_ITERATIONS; i++) {
                iterate();
                // std::cout << "Hit rate: " << hitRate() << " after iteration: " << i << std::endl;
                if (hasConverged(i, capacityConstraint, hitRateConstraint)) break;
            }

            if (i == MAX_ITERATIONS && WOOF_WARN_MAX_ITERATIONS) {
                std::cerr << "WARNING: Reached max iterations. May not have converged. "  << std::endl;
                // checkCapacityConstraint(true);
            }

            iterations += i;

            for (auto& cl : classes) {
                cl.age = sparse::expand(cl.age, sparsePoints, false);
                cl.hit = sparse::expand(cl.hit, sparsePoints, true);
                cl.evict = sparse::expand(cl.evict, sparsePoints, true);
            }

            // std::cout << "Complete in " << i << " iterations." << std::endl;
        }

        void cleanup() {
            classifier.cleanup(classes);
        }
}; // class Cache

} // namespace woof

#include "woof_repl.h"
#include "woof_classifier.h"
