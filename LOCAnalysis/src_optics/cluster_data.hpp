#pragma once

#include <vector>
#include <memory>

namespace pyclustering {
namespace clst {

using noise                 = std::vector<size_t>;
using noise_ptr             = std::shared_ptr<noise>;

using index_sequence        = std::vector<std::size_t>;

using cluster               = std::vector<std::size_t>;
using cluster_sequence      = std::vector<cluster>;
using cluster_sequence_ptr  = std::shared_ptr<cluster_sequence>;


/*!
    @class  cluster_data
    @brief  Represents result of cluster analysis.
    @param  p_other: another clustering data.
    @param  p_index: index of specified cluster.
*/
class cluster_data {
    protected:
        cluster_sequence      m_clusters = { };                 // Allocated clusters during clustering process.

    public:
        cluster_data() = default;
        cluster_data(const cluster_data & p_other) = default;   
        cluster_data(cluster_data && p_other) = default;   
        virtual ~cluster_data() = default;

    public:
        cluster_sequence & clusters();                          //  Returns reference to clusters.
        const cluster_sequence & clusters() const;              //  Returns constant reference to clusters.
        std::size_t size() const;                               //  Returns amount of clusters.
        
    public:
        cluster & operator[](const size_t p_index);             //  Provides access to specified cluster. 
        const cluster & operator[](const size_t p_index) const; //  Provides access to specified cluster. 
        bool operator == (const cluster_data & p_other) const;  //  Returns true if both objects have the same amount of clusters with the same elements.
        bool operator != (const cluster_data & p_other) const;  //  Returns true if both objects have are not the same. 
};


}

}