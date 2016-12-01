#include <queue>
#include "woof.h"

namespace woof {

// Updates
void Class::refresh(const vec& _rd, fp_t _prob) {
    rd = _rd;
    prob = _prob;
    delete rankfn;
    rankfn = factory(rd.n_elem, *this);
    // std::cout << "Rank: " << rankfn->rows(0,512).t() << std::endl;

    vec sparseRd = sparse::compress(rd, cache->sparsePoints, true);
    vec sparseRanks = sparse::rank(*rankfn, cache->sparsePoints, cache->rankWeights);

    // std::cout << "Sparse points (" << cache->sparsePoints.n_elem << " elems)" << std::endl;
    // std::cout << "Original RD (" << rd.n_elem << " elems)" << std::endl;
    // std::cout << "Compressed RD (" << sparseRd.n_elem << " elems) :" << sparseRd.t() << std::endl;

    rd = sparseRd;
    rankfn->resize(rd.n_elem);
    *rankfn = sparseRanks;
}

// Iteration
void Class::iterate(fp_t cacheHitRate, const Rank& cacheRankDist, const Rank& cacheMaxRankDist, const uvec& sparsePoints) {
    vec cdfrd = arma::cumsum(rd);

    // std::cout << "Hit: " << hit.rows(0,512).t() << std::endl;
    // std::cout << "Evi: " << evict.rows(0,512).t() << std::endl;

    fp_t cumhit = 0.;
    fp_t cumevbelow = 0.;

    vec nage(age.n_elem, arma::fill::zeros);
    vec nhit(age.n_elem, arma::fill::zeros);
    vec nevict(age.n_elem, arma::fill::zeros);
    nage(0) = prob / cache->size;

    uword i = 0;
    if (sparsePoints[i] == 0) { // special case to deal with non-sparse solutions, where it is easier just to assume a == i
        ++i;
        assert(sparsePoints.n_elem == rd.n_elem);
    }

    uword lastPoint = 0;

    // main loop
    for (; i < sparsePoints.n_elem; i++) {
        nhit(i) = rd(i) * (1. - cumevbelow);
        nhit(i) = std::max<fp_t>(0., nhit(i));
        cumhit += nhit(i);

        if (i > 0) {
            nage(i) = nage(i-1) - (nhit(i-1) + nevict(i-1)) / cache->size;
            nage(i) = std::max<fp_t>(0., nage(i));
        }

        auto r = (*rankfn)(i);
        if (cacheRankDist.at(r) > 0.) {
            fp_t probVictim = (1. - cacheHitRate) * cacheMaxRankDist.at(r) * nage(i) * (sparsePoints(i) - lastPoint) / cacheRankDist.at(r);
            // std::cout << "Evict -- idx: " << i << ", age: " << b << ", hitRate: " << cacheHitRate << ", rank: " << r << ", P_max_rank(r) " << cacheMaxRankDist.at(r) << ", age(i) " << nage(i) << ", P_rank(r): " << cacheRankDist.at(r) << " ==> " << probVictim << std::endl;
            if (probVictim > 0) {
                nevict(i) += probVictim;
            }
        }
        assert(nevict(i) >= 0.);
        if (i+1 < sparsePoints.n_elem && prob - cdfrd(i+1) > 0.) {
            cumevbelow += nevict(i) / (prob - cdfrd(i+1));
        }

        // std::cout << "Index: " << i
        //           << ", region end: " << a
        //           << ", age(i): " << nage(i)
        //           << ", hit(i): " << nhit(i)
        //           << ", evict(i): " << nevict(i)
        //           << std::endl;

        lastPoint = sparsePoints(i);
    }

    // drag
    age += (nage - age) / DRAG;
    hit += (nhit - hit) / DRAG;
    evict += (nevict - evict) / DRAG;
    hitRate += (cumhit - hitRate) / DRAG;
}

namespace sparse {

uvec divide(const vec& vec, uword nsplit, uword nadaptive) {
    struct Region {
            fp_t mass;
            uword start, end;
            bool operator< (const Region& that) const { return this->mass < that.mass; }
    };

    std::priority_queue<Region> heap;

    uword lastPoint = 0;
    fp_t totalMass = 0.;
    for (uword i = 0; i < nsplit; i++) {
        uword point = (i + 1) * (vec.n_elem - 1) / nsplit;
        fp_t mass = arma::accu(vec.rows(lastPoint, point));
        totalMass += mass;
        heap.push(Region{ mass, lastPoint, point });
        lastPoint = point+1;
    }
    assert(std::abs(arma::accu(vec) - totalMass) < 1e-4);

    uword i = 0;

    while (i < nadaptive) {
        Region region = heap.top();
        heap.pop();

        uword p;
        fp_t w = 0.;
        bool split = false;

        // info("Region: %u - %u (%g)", region.start, region.end, region.mass);

        if (region.mass <= 1e-8 || region.start+1 == region.end) {
            split = false;
        } else {
            // find split
            // std::cout << "Find split: ";
            for (p = region.start; p < region.end + 1; p++) {
                // std::cout << p << " " << vec(p) << ", ";
                w += vec(p);
                if (w > region.mass / 2.) { break; }
            }
            // std::cout << std::endl;

            assert(region.start <= p);
            assert(p <= region.end);
            split = (p+1) < region.end;
            assert(std::abs(arma::accu(vec.rows(region.start, p)) - w) < 1e-6);
        }

        if (split) {
            // info("Split region %u - %u (%g) into regions %u - %u (%g) and %u - %u (%g)",
            //      region.start, region.end, region.mass,
            //      region.start, p+1, w,
            //      p+1, region.end, region.mass - w);
            heap.push(Region{ w, region.start, p });
            heap.push(Region{ region.mass - w, p+1, region.end });

            i += 1;
        } else {
            // can't split further
            // info("Skipping region %u - %u (%g)", region.start, region.end, region.mass);
            region.mass = -region.mass; // invert priority
            heap.push(region);
            if (std::abs(region.mass) < 1e-6) { break; } // early exit
        }
    }

    uvec sparsePoints = uvec(nsplit + i, arma::fill::zeros);

    i = 0;
    while (!heap.empty()) {
        // info("Popping region: %u - %u (%g)", heap.top().start, heap.top().end, heap.top().mass);
        sparsePoints(i) = heap.top().end;
        heap.pop();
        i++;
    }

    sparsePoints = arma::sort(sparsePoints);

    for (i = 1; i < sparsePoints.n_elem; i++) {
        assert(sparsePoints(i) > sparsePoints(i-1));
    }

    // std::cout << "Sparse (" << (nsplit + nadaptive) << ") " << sparsePoints.n_elem << ": " << sparsePoints.t() << std::endl;
    // std::cout << "Sparse mass ";
    // lastPoint = 0;
    // for (i = 0; i < sparsePoints.n_elem; i++) {
    //     std::cout << arma::accu(vec.rows(lastPoint, sparsePoints(i))) << " ";
    //     lastPoint = sparsePoints(i)+1;
    // }
    // std::cout << std::endl;

    return sparsePoints;
}

vec compress(const vec& vector, const uvec& sparsePoints, bool sum) {
    vec compressed = vec(sparsePoints.n_elem, arma::fill::zeros);

    uword i = 0;

    for (uword j = 0; j < sparsePoints.n_elem; j++) {
        uword i0 = i;

        while (i <= sparsePoints(j)) {
            compressed(j) += vector(i);
            i += 1;
        }

        uword i1 = i;

        if (!sum) {
            compressed(j) /= i1 - i0;
        }
    }

    // std::cout << "Compressed: " << vec.rows(0,128).t() << " to " << compressed.t() << std::endl;

    if (sum) {
        assert(std::abs(arma::accu(compressed) - arma::accu(vector.rows(0, sparsePoints(sparsePoints.n_elem-1)))) < 1e-6);
    }

    return compressed;
}

vec expand(const vec& vector, const uvec& sparsePoints, bool spread) {
    vec expanded = vec(sparsePoints(sparsePoints.n_elem-1)+1, arma::fill::zeros);

    uword i = 1;

    for (uword j = 0; j < sparsePoints.n_elem; j++) {
        uword p0 = (j > 0)? sparsePoints(j-1) : 0;
        uword p1 = sparsePoints(j);

        // fp_t y0 = (j > 0)? vec(j-1) : 0.;
        fp_t y1 = vector(j);

        while (i <= p1) {
            if (spread) {
                expanded(i) = y1 / (p1 - p0);
            } else {
                expanded(i) = y1;
            }
            i += 1;
        }
    }

    if (spread) {
        assert(std::abs(arma::accu(expanded) - arma::accu(vector)) < 1e-4);
    }

    return expanded;
}

uvec none(uword npoints) {
    uvec points = uvec(npoints, arma::fill::zeros);
    for (uword i = 0; i < npoints; i++) {
        points(i) = i;
    }
    return points;
}

vec rank(const vec& fullRanks,
         const uvec& sparsePoints,
         const vec& weights) {
    assert(weights.n_elem == fullRanks.n_elem);

    if (sparsePoints.n_elem == weights.n_elem) {
        return fullRanks;
    }

    vec sparseRanks(sparsePoints.n_elem, arma::fill::zeros);
    uword i = 0;
    fp_t weight;

    for (uword j = 0; j < sparsePoints.n_elem; j++) {
        weight = 0.;

        // std::cerr << "sparseRanks up to " << sparsePoints(j) << ": ";

        while (i <= sparsePoints(j)) {
            weight += weights(i);
            sparseRanks(j) += fullRanks(i) * weights(i);
            // std::cerr << fullRanks(i) << " "; // (" << weights(i) << "), ";
            i += 1;
        }

        if (weight > 1e-8) {
            sparseRanks(j) /= weight;
        }
        // std::cerr << " ==> " << sparseRanks(j) << std::endl;
    }

    return sparseRanks;
}

} // namespace sparse

} // namespace woof
