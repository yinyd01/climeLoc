#pragma once

#include <memory>
#include <vector>
#include "cluster_data.hpp"

namespace pyclustering {
namespace clst {

/*!
    @class  dbscan_data dbscan_data.hpp pyclustering/cluster/dbscan_data.hpp
    @brief  Clustering results of DBSCAM algorithm that consists of information about allocated clusters and noise (points that are not related to any cluster).
    @param  p_other: another DBSCAN clustering data.
*/
class dbscan_data : public cluster_data {
    
    private:
        clst::noise       m_noise;

    public:
        dbscan_data() = default;
        dbscan_data(const dbscan_data & p_other) = default;
        dbscan_data(dbscan_data && p_other) = default;
        virtual ~dbscan_data() = default;

    public:
        clst::noise & noise() { return m_noise; }               //  Returns reference to outliers represented by indexes.
        const clst::noise & noise() const { return m_noise; }   //  Returns constant reference to outliers represented by indexes.
    };
}
}