#pragma once

#include <cstddef>

namespace pyclustering {
namespace clst {

/*!
    @class  optics_descriptor optics_descriptor.hpp pyclustering/cluster/optics_descriptor.hpp
    @brief  Object description that used by OPTICS algorithm for cluster analysis
    @param  p_other: another clustering data.
    @param  p_index: index of optics object that corresponds to index of real object in dataset.
    @param  p_core_distance: core distance of optics-object.
    @param  p_reachability_distance: reachability distance of optics-object.
*/
struct optics_descriptor {
    public:
        static const double NONE_DISTANCE;              // Denotes if a distance value is not defined.

    public:
        std::size_t     m_index = -1;                   // Index of the object in the data set.
        double          m_core_distance = 0;            // Core distance that is minimum distance to specified number of neighbors.
        double          m_reachability_distance = 0;    // Reachability distance to this object.
        bool            m_processed = false;            // Defines the object is processed -`true` if is current object has been already processed.

    public:
        optics_descriptor() = default;
        optics_descriptor(const optics_descriptor & p_other) = default;
        optics_descriptor(optics_descriptor && p_other) = default;
        optics_descriptor(const std::size_t p_index, const double p_core_distance, const double p_reachability_distance);
        ~optics_descriptor() = default;

    public:
        void clear();
};



/*!
    @brief Less comparator for object description that used by OPTICS algorithm for cluster analysis
    @param  p_object1: the left operand to compare.
    @param  p_object2: the right operand to compare.
*/
struct optics_pointer_descriptor_less {
    bool operator()(const optics_descriptor* p_object1, const optics_descriptor* p_object2) const;  //  Returns true if left operand is less than right operand.
};


}
}