#pragma once

namespace woof {

// the public release contains several ranking functions for LRU,
// PDP, and IRGD, as described in our HPCA 2016 paper, plus a
// ranking function that computes the expected time until reuse
// (which performs relatively poorly). -nzb
namespace repl {

// Read ranks from file.
inline Rank* rankFile(const uword n_elem) {
    char* rankFileName = "ranks.out";
    std::ifstream rank_file;
    rank_file.open(rankFileName);
    int n = 0;
    rank_file >> n;

    auto r = new Rank(n_elem, 0, n_elem);
    for (uword a = 0; a < n_elem; a++) {
        if (a<n) {
            rank_file >> (*r)(a);
        }
        else {
            (*r)(a) = n+1;
        }
    }
    rank_file.close();
    return r;
}

inline Rank* rankFile(const uword n_elem, const Class& cl) {
    return rankFile(n_elem);
}

// Most recently used.
inline Rank* MRU(const uword n_elem) {
    auto r = new Rank(n_elem, 0, n_elem);
    for (uword a = 0; a < n_elem; a++) {
        (*r)(a) = n_elem-a;
    }
    return r;
}

inline Rank* MRU(const uword n_elem, const Class& cl) {
    return MRU(n_elem);
}

// Least recently used.
inline Rank* LRU(const uword n_elem) {
    auto r = new Rank(n_elem, 0, n_elem);
    for (uword a = 0; a < n_elem; a++) {
        (*r)(a) = a;
    }
    return r;
}

inline Rank* LRU(const uword n_elem, const Class& cl) {
    return LRU(n_elem);
}

// Protecting distance policy (as published, with their model).
class PDP : public Rank {
    public:
        PDP(const uword n_elem, uint32_t pd=0)
                : Rank(n_elem, 0, std::max(pd,(uint32_t)n_elem)) {
            init(pd);
        }

        void init(uint32_t pd) {
            (*this)(0) = 0.;
            for (uword a = 1; a < n_elem; a++) {
                (*this)(a) = (a < pd)? pd-a : a;
            }
        }

        static uint32_t bestProtectingDistance(const vec& rdpmf, uint32_t modelLines) {
            fp_t cumrd = 0.0;
            fp_t cumhitocc = 0.0;
            fp_t bestmodel = 0.0;
            uint32_t bestpd = 0;
            assert(rdpmf(0) < 1e-10);
            for (uint32_t pd = 1; pd < rdpmf.n_elem; pd++) {
                cumrd += rdpmf(pd);
                cumhitocc += pd*rdpmf(pd);

                fp_t d = cumhitocc + (1 - cumrd)*(pd + modelLines);
                if (d < 1e-40) continue;

                fp_t model = cumrd/d;

                if (model > bestmodel) {
                    bestmodel = model;
                    bestpd = pd;
                }
            }

            info("Best protecting distance = %d, best E(dp) = %f | buckets = %u | modelLines = %d", bestpd, bestmodel, rdpmf.n_elem, modelLines);
            return bestpd;
        }

        virtual void refresh(const Class& cl) {
            assert(cl.prob == 1.);
            init(bestProtectingDistance(cl.rd, cl.cache->size));
        }
};

// HarmonicExpectedTimeToReuse --- IRGD
//
// There are two ways to view this metric. The first is as
// Expected utility, without considering cache
// effects. However its not at all clear what the physical
// analog of a weighted-sum of utilities is. (See discussion
// later in this file.)
//
// It is effectively the harmonic mean of expected time to
// reuse, and has no clear physical meaning. (It is actually
// the inverse of the harmonic mean.) It has the virtue of
// gracefully handling infinite time-to-reuse, but performs
// poorly in real examples because it lacks physical
// justification.
//
// It is computed in quadratic time.
inline Rank* HarmonicExpectedTimeToReuse(const uword n_elem, const vec& rd, fp_t rdmass) {
    vec cumrd = arma::cumsum(rd);
    vec revrd = rdmass - cumrd;

    auto eu = vec(n_elem, arma::fill::zeros);
    for (uword a = 0; a < n_elem; a++) {
        auto util = 0.;

        for (uword d = a+1; d < n_elem; d++) {
            util += rd(d) / (d-a);
        }

        eu(a) = (util > 0)? util / revrd(a) : 0.;
        eu(a) = std::min<fp_t>(eu(a), 1.);
        // std::cout << a << " " << util << " / " << revrd(a) << " = " << eu(a) << std::endl;
    }

    // eu = 1./eu;
    eu = -eu;
    assert(-1.001 < arma::min(eu));
    eu(0) = -1.001;

    return new Rank(eu);
}

inline Rank* HarmonicExpectedTimeToReuse(const uword n_elem, const Class& cl) {
    return HarmonicExpectedTimeToReuse(n_elem, cl.rd, cl.prob);
}

// ExpectedTimeToReuse
//
// Expected time to reuse, without considering cache effects.
// Fixes dead lines with a reuse distance of n_elem. This is a
// bullshit way of handling the problem, but there is no clear
// choice of this parameter.
inline Rank* ExpectedTimeToReuse(const uword n_elem, const vec& rd, fp_t rdmass) {
    vec etth(n_elem, arma::fill::zeros);

    fp_t etthUnconditioned = 0.;
    fp_t rdAbove = 0.;
    fp_t longLineFraction = (rdmass - arma::sum(rd));
    longLineFraction = std::max<fp_t>(0., longLineFraction);

    for (uword a = n_elem-1; a < n_elem; a--) { /* bounds are correct; unsigned wraps around */
        etthUnconditioned += rdAbove;
        rdAbove += rd(a);
        if (rdAbove > 0.) {
            fp_t probShort = rdAbove / (rdAbove + longLineFraction);
            etth(a) = etthUnconditioned * probShort + n_elem * (1. - probShort);
        } else {
            etth(a) = n_elem;
        }
    }

    etth(0) = -1e-5;
    return new Rank(etth);
}

inline Rank* ExpectedTimeToReuse(const uword n_elem, const Class& cl) {
    return ExpectedTimeToReuse(n_elem, cl.rd, cl.prob);
}

} // namespace repl

} // namespace woof
