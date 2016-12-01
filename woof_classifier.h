#pragma once

namespace woof {

// currently the public release only includes a degenerate
// classifier that ranks all lines in the same class. the
// reused/non-reused classification scheme described in our
// technical report is available upon request, but ommitted for
// simplicity. -nzb
class Unclassifier : public Classifier {
    public:
        virtual vector<Class> init(const vec& rd, Rank::Factory factory, Cache* cache) {
            auto classes = vector<Class>{Class(rd.n_elem, factory)};
            classes.front().reseed(rd, 1., cache);
            return classes;
        }
        virtual void refresh(vector<Class>& classes, const vec& rd) {
            assert(classes.size() == 1);
            // the rd never changes for an Unclassifier -- no need to refresh! -nzb
            // classes[0].refresh(rd, 1.);
        }
};

} // namespace woof
