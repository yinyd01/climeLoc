#pragma once

#include "dbscan_data.hpp"
#include "optics_descriptor.hpp"

namespace pyclustering {
namespace clst {


using ordering                = std::vector<double>;                
using optics_object_sequence  = std::vector<optics_descriptor>;     

/*!
    @class  optics_data optics_data.hpp pyclustering/cluster/optics_data.hpp
    @brief  Clustering results of OPTICS algorithm that consists of information about allocated 
            clusters and noise (points that are not related to any cluster), ordering (that represents
            density-based clustering structure) and proper radius.
    @param  p_radius: new value of the connectivity radius.
*/
class optics_data : public dbscan_data {
    
    private:
        double                  m_radius   = 0;                 //  epsilon
        ordering                m_ordering = { };               //  Sequence container where ordering diagram is stored.
        optics_object_sequence  m_optics_objects = { };         //  Sequence container where OPTICS descriptors are stored.

    public:
        optics_data() = default;
        optics_data(const optics_data & p_other) = default;
        optics_data(optics_data && p_other) = default;
        virtual ~optics_data() = default;

    public:
        ordering & cluster_ordering() { return m_ordering; }                                // Returns reference to cluster-ordering that represents density-based clustering structure.
        const ordering & cluster_ordering() const { return m_ordering; }                    //  Returns const reference to cluster-ordering that represents density-based clustering structure.
        optics_object_sequence & optics_objects() { return m_optics_objects; }              //  Returns reference to optics objects that corresponds to points from input dataspace.
        const optics_object_sequence & optics_objects() const { return m_optics_objects; }  //  Returns const reference to optics objects that corresponds to points from input dataspace.

        double get_radius() const { return m_radius; }  //  Returns connectivity radius that can be differ from input parameter.
        void set_radius(const double p_radius) { m_radius = p_radius; }
};


}

}