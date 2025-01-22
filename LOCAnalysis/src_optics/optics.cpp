#include "optics.h"
#include "ordering_analyser.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>

#include <pyclustering/container/kdtree_searcher.hpp>


namespace pyclustering {
namespace clst {


const double      optics::NONE_DISTANCE = optics_descriptor::NONE_DISTANCE;
const std::size_t optics::INVALID_INDEX = std::numeric_limits<std::size_t>::max();


optics::optics(const double p_radius, const std::size_t p_neighbors) : optics() { 
    m_radius = p_radius;
    m_neighbors = p_neighbors;
}


optics::optics(const double p_radius, const std::size_t p_neighbors, const std::size_t p_amount_clusters) : optics() { 
    m_radius = p_radius;
    m_neighbors = p_neighbors;
    m_amount_clusters = p_amount_clusters;
}


void optics::process(const dataset & p_data, optics_data & p_result)
{
    process(p_data, data_t::POINTS, p_result);
}


void optics::process(const dataset & p_data, const data_t p_type, optics_data & p_result) 
{
    m_data_ptr    = &p_data;
    m_result_ptr  = &p_result;
    m_type        = p_type;

    calculate_cluster_result();

    if ( (m_amount_clusters > 0) && (m_amount_clusters != m_result_ptr->clusters().size()) )
    {
        double radius = ordering_analyser::calculate_connvectivity_radius(m_result_ptr->cluster_ordering(), m_amount_clusters);

        if (radius > 0) {
            m_radius = radius;
            calculate_cluster_result();
        }
    }

    m_result_ptr->set_radius(m_radius);

    m_data_ptr    = nullptr;
    m_result_ptr  = nullptr;
}






void optics::calculate_cluster_result() 
{
    initialize();
    allocate_clusters();
    calculate_ordering();
}


void optics::initialize() 
{
    if (m_type == data_t::POINTS) 
        create_kdtree();
    
    m_optics_objects = &(m_result_ptr->optics_objects());
    if (m_optics_objects->empty()) 
    {
        m_optics_objects->reserve(m_data_ptr->size());

        for (std::size_t i = 0; i < m_data_ptr->size(); i++) 
            m_optics_objects->emplace_back(i, optics::NONE_DISTANCE, optics::NONE_DISTANCE);    
    }
    else 
        std::for_each(m_optics_objects->begin(), m_optics_objects->end(), [](auto & p_object) { p_object.clear(); });
    
    m_ordered_database.clear();
    m_result_ptr->clusters().clear();
    m_result_ptr->noise().clear();
}



void optics::allocate_clusters() 
{
    for (auto & optics_object : *m_optics_objects)
        if (!optics_object.m_processed) 
            expand_cluster_order(optics_object);
    
    extract_clusters();
}



void optics::expand_cluster_order(optics_descriptor &p_object) 
{
    // asign the point as processed
    p_object.m_processed = true;

    // get the neighbours of the point
    neighbors_collection neighbors;
    get_neighbors(p_object.m_index, neighbors);

    // output the point to the ordered_database
    m_ordered_database.push_back(&p_object);

    // if the point is a core_point
    if (neighbors.size() >= m_neighbors) 
    {
        // calculate the core_distance of the point
        p_object.m_core_distance = get_core_distance(neighbors);

        // creat an ordered seed
        std::multiset<optics_descriptor *, optics_pointer_descriptor_less> order_seed;
        
        // update the seed w.r.t the point
        update_order_seed(p_object, neighbors, order_seed);

        // update the seed w.r.t the point's neighbours
        // erase the first one and push_back new elements to the seed until the seed is empty 
        while(!order_seed.empty()) 
        {
            optics_descriptor *descriptor = *(order_seed.begin());
            order_seed.erase(order_seed.begin());

            get_neighbors(descriptor->m_index, neighbors);
            descriptor->m_processed = true;

            m_ordered_database.push_back(descriptor);

            if (neighbors.size() >= m_neighbors) 
            {
                descriptor->m_core_distance = get_core_distance(neighbors);
                update_order_seed(*descriptor, neighbors, order_seed);
            }
            else 
                descriptor->m_core_distance = optics::NONE_DISTANCE;
        }
    }
    else
        p_object.m_core_distance = optics::NONE_DISTANCE;
    
}



void optics::update_order_seed(const optics_descriptor &p_object, const neighbors_collection &p_neighbors, std::multiset<optics_descriptor *, optics_pointer_descriptor_less> &order_seed) 
{
    for (auto & descriptor : p_neighbors) 
    {
        std::size_t index_neighbor = descriptor.m_index;
        double current_reachability_distance = descriptor.m_reachability_distance;

        optics_descriptor & optics_object = m_optics_objects->at(index_neighbor);
        if (!optics_object.m_processed) 
        {
            double reachable_distance = std::max({ current_reachability_distance, p_object.m_core_distance });

            if (optics_object.m_reachability_distance == optics::NONE_DISTANCE) 
            {
                optics_object.m_reachability_distance = reachable_distance;
                order_seed.insert(&optics_object);
            }
            else 
            {
                if (reachable_distance < optics_object.m_reachability_distance) 
                {
                    optics_object.m_reachability_distance = reachable_distance;

                    auto object_iterator = std::find_if(order_seed.begin(), order_seed.end(), [&optics_object](optics_descriptor * obj) {
                        return obj->m_index == optics_object.m_index;
                    });

                    order_seed.erase(object_iterator);
                    order_seed.insert(&optics_object);
                }
            }
        }
    }
}



void optics::extract_clusters() {
    cluster_sequence & clusters = m_result_ptr->clusters();
    clst::noise & noise = m_result_ptr->noise();

    cluster * current_cluster = (cluster *) &noise;

    for (auto optics_object : m_ordered_database) {
        if ( (optics_object->m_reachability_distance == optics::NONE_DISTANCE) || (optics_object->m_reachability_distance > m_radius) ) {
            if ( (optics_object->m_core_distance != optics::NONE_DISTANCE) && (optics_object->m_core_distance <= m_radius) ) {
                clusters.push_back({ optics_object->m_index });
                current_cluster = &clusters.back();
            }
            else {
                noise.push_back(optics_object->m_index);
            }
        }
        else {
            current_cluster->push_back(optics_object->m_index);
        }
    }
}


void optics::get_neighbors(const size_t p_index, neighbors_collection & p_neighbors) {
    switch(m_type) {
    case data_t::POINTS:
        get_neighbors_from_points(p_index, p_neighbors);
        break;

    case data_t::DISTANCE_MATRIX:
        get_neighbors_from_distance_matrix(p_index, p_neighbors);
        break;

    default:
        throw std::invalid_argument("Incorrect input data type is specified '" + std::to_string((unsigned) m_type) + "'");
    }
}


void optics::get_neighbors_from_points(const std::size_t p_index, neighbors_collection & p_neighbors) {
    p_neighbors.clear();

    container::kdtree_searcher searcher((*m_data_ptr)[p_index], m_kdtree.get_root(), m_radius);

    container::kdtree_searcher::rule_store rule = [&p_index, &p_neighbors](const container::kdnode::ptr & p_node, const double p_distance) {
            if (p_index != (std::size_t) p_node->get_payload()) {
                p_neighbors.emplace((std::size_t) p_node->get_payload(), std::sqrt(p_distance));
            }
        };

    searcher.find_nearest(rule);
}


void optics::get_neighbors_from_distance_matrix(const std::size_t p_index, neighbors_collection & p_neighbors) {
    p_neighbors.clear();

    const auto & distances = m_data_ptr->at(p_index);
    for (std::size_t index_neighbor = 0; index_neighbor < distances.size(); index_neighbor++) {
        const double candidate_distance = distances[index_neighbor];
        if ( (candidate_distance <= m_radius) && (index_neighbor != p_index) ) {
            p_neighbors.emplace(index_neighbor, candidate_distance);
        }
    }
}


double optics::get_core_distance(const neighbors_collection & p_neighbors) const {
    auto iter = p_neighbors.cbegin();
    for (std::size_t index = 0; index < (m_neighbors - 1); ++index) {
        ++iter;
    }

    return iter->m_reachability_distance;
}


void optics::calculate_ordering() {
    if (!m_result_ptr->cluster_ordering().empty()) { return; }

    ordering & ordering = m_result_ptr->cluster_ordering();
    cluster_sequence & clusters = m_result_ptr->clusters();

    for (auto & cluster : clusters) {
        for (auto index_object : cluster) {
            const optics_descriptor & optics_object = m_optics_objects->at(index_object);
            if (optics_object.m_reachability_distance != optics::NONE_DISTANCE) {
                ordering.push_back(optics_object.m_reachability_distance);
            }
        }
    }
}


void optics::create_kdtree() {
    std::vector<void *> payload(m_data_ptr->size());
    for (std::size_t index = 0; index < m_data_ptr->size(); index++) {
        payload[index] = (void *)index;
    }

    m_kdtree = container::kdtree_balanced(*m_data_ptr, payload);
}


}

}