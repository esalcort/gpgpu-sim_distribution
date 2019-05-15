// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, George L. Yuan,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "dram_sched.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "../abstract_hardware_model.h"
#include "mem_latency_stat.h"

frfcfs_scheduler::frfcfs_scheduler( const memory_config *config, dram_t *dm, memory_stats_t *stats )
{
    m_config = config;
    m_stats = stats;
    m_num_pending = 0;
    m_num_write_pending = 0;
    m_dram = dm;
    m_queue = new std::list<dram_req_t*>[m_config->nbk];
    m_bins = new std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> >[ m_config->nbk ];
    m_last_row = new std::list<std::list<dram_req_t*>::iterator>*[ m_config->nbk ];
    curr_row_service_time = new unsigned[m_config->nbk];
    row_service_timestamp = new unsigned[m_config->nbk];
    for ( unsigned i=0; i < m_config->nbk; i++ ) {
        m_queue[i].clear();
        m_bins[i].clear();
        m_last_row[i] = NULL;
        curr_row_service_time[i] = 0;
        row_service_timestamp[i] = 0;
    }
    if(m_config->seperate_write_queue_enabled) {
        m_write_queue = new std::list<dram_req_t*>[m_config->nbk];
        m_write_bins = new std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> >[ m_config->nbk ];
        m_last_write_row = new std::list<std::list<dram_req_t*>::iterator>*[ m_config->nbk ];

        for ( unsigned i=0; i < m_config->nbk; i++ ) {
            m_write_queue[i].clear();
            m_write_bins[i].clear();
            m_last_write_row[i] = NULL;
        }
    }
    m_mode = READ_MODE;

}

void frfcfs_scheduler::add_req( dram_req_t *req )
{
    //TODO: Use as sample to access cluster, core, and other gpgpu_sim members. Then REMOVE
    //unsigned cluster = req->data->get_sid() / m_dram->m_gpu->m_shader_config->n_simt_cores_per_cluster;
    //unsigned core = req->data->get_sid() % m_dram->m_gpu->m_shader_config->n_simt_cores_per_cluster;
    //printf("DRAM access sid: %d, (cluster, core): (%d, %d)\n", req->data->get_sid(), cluster, core);
    if(m_config->seperate_write_queue_enabled && req->data->is_write()) {
        assert(m_num_write_pending < m_config->gpgpu_frfcfs_dram_write_queue_size);
        m_num_write_pending++;
        m_write_queue[req->bk].push_front(req);
        std::list<dram_req_t*>::iterator ptr = m_write_queue[req->bk].begin();
        m_write_bins[req->bk][req->row].push_front( ptr ); //newest reqs to the front
    } else {
        assert(m_num_pending < m_config->gpgpu_frfcfs_dram_sched_queue_size);
        m_num_pending++;
        m_queue[req->bk].push_front(req);
        std::list<dram_req_t*>::iterator ptr = m_queue[req->bk].begin();
        m_bins[req->bk][req->row].push_front( ptr ); //newest reqs to the front
    }

}

void frfcfs_scheduler::data_collection(unsigned int bank)
{
    if (gpu_sim_cycle > row_service_timestamp[bank]) {
        curr_row_service_time[bank] = gpu_sim_cycle - row_service_timestamp[bank];
        if (curr_row_service_time[bank] > m_stats->max_servicetime2samerow[m_dram->id][bank])
            m_stats->max_servicetime2samerow[m_dram->id][bank] = curr_row_service_time[bank];
    }
    curr_row_service_time[bank] = 0;
    row_service_timestamp[bank] = gpu_sim_cycle;
    if (m_stats->concurrent_row_access[m_dram->id][bank] > m_stats->max_conc_access2samerow[m_dram->id][bank]) {
        m_stats->max_conc_access2samerow[m_dram->id][bank] = m_stats->concurrent_row_access[m_dram->id][bank];
    }
    m_stats->concurrent_row_access[m_dram->id][bank] = 0;
    m_stats->num_activates[m_dram->id][bank]++;
}

dram_req_t *frfcfs_scheduler::schedule( unsigned bank, unsigned curr_row )
{
    //row
    bool rowhit = true;
    std::list<dram_req_t*> *m_current_queue = m_queue;
    std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> > *m_current_bins = m_bins ;
    std::list<std::list<dram_req_t*>::iterator> **m_current_last_row = m_last_row;

    if(m_config->seperate_write_queue_enabled) {
        if(m_mode == READ_MODE &&
                ((m_num_write_pending >= m_config->write_high_watermark )
                 // || (m_queue[bank].empty() && !m_write_queue[bank].empty())
                )) {
            m_mode = WRITE_MODE;
        }
        else if(m_mode == WRITE_MODE &&
                (( m_num_write_pending < m_config->write_low_watermark )
                 //  || (!m_queue[bank].empty() && m_write_queue[bank].empty())
                )){
            m_mode = READ_MODE;
        }
    }

    if(m_mode == WRITE_MODE) {
        m_current_queue = m_write_queue;
        m_current_bins = m_write_bins ;
        m_current_last_row = m_last_write_row;
    }

    if ( m_current_last_row[bank] == NULL ) {
        if ( m_current_queue[bank].empty() )
            return NULL;

        std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> >::iterator bin_ptr = m_current_bins[bank].find( curr_row );
        if ( bin_ptr == m_current_bins[bank].end()) {
            dram_req_t *req = m_current_queue[bank].back();
            bin_ptr = m_current_bins[bank].find( req->row );
            assert( bin_ptr != m_current_bins[bank].end() ); // where did the request go???
            m_current_last_row[bank] = &(bin_ptr->second);
            data_collection(bank);
            rowhit = false;
        } else {
            m_current_last_row[bank] = &(bin_ptr->second);
            rowhit = true;
        }
    }
    std::list<dram_req_t*>::iterator next = m_current_last_row[bank]->back();
    dram_req_t *req = (*next);

    //rowblp stats
    m_dram->access_num++;
    bool is_write = req->data->is_write();
    if(is_write)
        m_dram->write_num++;
    else
        m_dram->read_num++;

    if(rowhit) {
        m_dram->hits_num++;
        if(is_write)
            m_dram->hits_write_num++;
        else
            m_dram->hits_read_num++;
    }

    m_stats->concurrent_row_access[m_dram->id][bank]++;
    m_stats->row_access[m_dram->id][bank]++;
    m_current_last_row[bank]->pop_back();

    m_current_queue[bank].erase(next);
    if ( m_current_last_row[bank]->empty() ) {
        m_current_bins[bank].erase( req->row );
        m_current_last_row[bank] = NULL;
    }
#ifdef DEBUG_FAST_IDEAL_SCHED
    if ( req )
        printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n", 
                (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row );
#endif

    if(m_config->seperate_write_queue_enabled && req->data->is_write()) {
        assert( req != NULL && m_num_write_pending != 0 );
        m_num_write_pending--;
    }
    else {
        assert( req != NULL && m_num_pending != 0 );
        m_num_pending--;
    }

    return req;
}


void frfcfs_scheduler::print( FILE *fp )
{
    for ( unsigned b=0; b < m_config->nbk; b++ ) {
        printf(" %u: queue length = %u\n", b, (unsigned)m_queue[b].size() );
    }
}

void dram_t::scheduler_frfcfs()
{
    unsigned mrq_latency;
    frfcfs_scheduler *sched = m_frfcfs_scheduler;
    while ( !mrqq->empty() ) {
        dram_req_t *req = mrqq->pop();

        // Power stats
        //if(req->data->get_type() != READ_REPLY && req->data->get_type() != WRITE_ACK)
        m_stats->total_n_access++;

        if(req->data->get_type() == WRITE_REQUEST){
            m_stats->total_n_writes++;
        }else if(req->data->get_type() == READ_REQUEST){
            m_stats->total_n_reads++;
        }

        req->data->set_status(IN_PARTITION_MC_INPUT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        sched->add_req(req);
    }

    dram_req_t *req;
    unsigned i;
    for ( i=0; i < m_config->nbk; i++ ) {
        unsigned b = (i+prio)%m_config->nbk;
        if ( !bk[b]->mrq ) {

            req = sched->schedule(b, bk[b]->curr_row);

            if ( req ) {
                req->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                prio = (prio+1)%m_config->nbk;
                bk[b]->mrq = req;
                if (m_config->gpgpu_memlatency_stat) {
                    mrq_latency = gpu_sim_cycle + gpu_tot_sim_cycle - bk[b]->mrq->timestamp;
                    m_stats->tot_mrq_latency += mrq_latency;
                    m_stats->tot_mrq_num++;
                    bk[b]->mrq->timestamp = gpu_tot_sim_cycle + gpu_sim_cycle;
                    m_stats->mrq_lat_table[LOGB2(mrq_latency)]++;
                    if (mrq_latency > m_stats->max_mrq_latency) {
                        m_stats->max_mrq_latency = mrq_latency;
                    }
                }

                break;
            }
        }
    }
}

frmp_scheduler::frmp_scheduler( const memory_config *config, dram_t *dm, memory_stats_t *stats) : frfcfs_scheduler(config, dm, stats){
    //nothing special
}


dram_req_t *frmp_scheduler::schedule( unsigned bank, unsigned curr_row )
{
    //row
    bool rowhit = true;
    std::list<dram_req_t*> *m_current_queue = m_queue;
    std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> > *m_current_bins = m_bins ;
    std::list<std::list<dram_req_t*>::iterator> **m_current_last_row = m_last_row;

    if(m_config->seperate_write_queue_enabled) {
        if(m_mode == READ_MODE &&
                ((m_num_write_pending >= m_config->write_high_watermark )
                 // || (m_queue[bank].empty() && !m_write_queue[bank].empty())
                )) {
            m_mode = WRITE_MODE;
        }
        else if(m_mode == WRITE_MODE &&
                (( m_num_write_pending < m_config->write_low_watermark )
                 //  || (!m_queue[bank].empty() && m_write_queue[bank].empty())
                )){
            m_mode = READ_MODE;
        }
    }

    if(m_mode == WRITE_MODE) {
        m_current_queue = m_write_queue;
        m_current_bins = m_write_bins ;
        m_current_last_row = m_last_write_row;
    }

    if ( m_current_last_row[bank] == NULL ) {
        if ( m_current_queue[bank].empty() ){
            return NULL;
        }
 
        //find the max pressure in the queue
        int64_t mx_pressure = 0;
        dram_req_t* mx_it = m_current_queue[bank].back(); //default to fcfs
        dram_req_t* q_item = nullptr;
        for (auto it = m_current_queue[bank].rbegin(); it != m_current_queue[bank].rend(); ++it){
            q_item = *it;
            
            if ( ((gpu_sim_cycle + gpu_tot_sim_cycle) - q_item->timestamp) > (m_config->gpgpu_frfcfs_dram_write_queue_size) ){
                mx_it = q_item;
                break;
            } 
            if (q_item->data->get_sid() == -1){
                continue;
            }

            unsigned clid = q_item->data->get_sid() / m_dram->m_gpu->m_shader_config->n_simt_cores_per_cluster;
            unsigned sid = q_item->data->get_sid() % m_dram->m_gpu->m_shader_config->n_simt_cores_per_cluster;
            int64_t pressure = m_dram->m_gpu->m_cluster[clid]->get_cores()[sid]->mshr_pressure;
            //std::cout << "p,c,s" << pressure << "," << clid << "," << sid << std::endl;
            if (pressure > mx_pressure){
                mx_pressure = pressure;
                mx_it = q_item;
            }
        }
        dram_req_t *req = mx_it;
        auto bin_ptr = m_current_bins[bank].find( req->row );
        assert( bin_ptr != m_current_bins[bank].end() ); // where did the request go???
        m_current_last_row[bank] = &(bin_ptr->second);
        data_collection(bank);
        rowhit = false;
    } else {
        rowhit = true;
    }

    std::list<dram_req_t*>::iterator next = m_current_last_row[bank]->back();
    dram_req_t *req = (*next);

    //rowblp stats
    m_dram->access_num++;
    bool is_write = req->data->is_write();
        if(is_write)
            m_dram->write_num++;
        else
            m_dram->read_num++;

    if(rowhit) {
        m_dram->hits_num++;
        if(is_write)
            m_dram->hits_write_num++;
        else
            m_dram->hits_read_num++;
    }

    m_stats->concurrent_row_access[m_dram->id][bank]++;
    m_stats->row_access[m_dram->id][bank]++;
    m_current_last_row[bank]->pop_back();

    m_current_queue[bank].erase(next);
    if ( m_current_last_row[bank]->empty() ) {
        m_current_bins[bank].erase( req->row );
        m_current_last_row[bank] = NULL;
    }
#ifdef DEBUG_FAST_IDEAL_SCHED
    if ( req )
        printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n", 
                (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row );
#endif

    if(m_config->seperate_write_queue_enabled && req->data->is_write()) {
        assert( req != NULL && m_num_write_pending != 0 );
        m_num_write_pending--;
    }
    else {
        assert( req != NULL && m_num_pending != 0 );
        m_num_pending--;
        //std::cout << m_num_pending << std::endl;
    }

    return req;
}

frlp_scheduler::frlp_scheduler( const memory_config *config, dram_t *dm, memory_stats_t *stats) : frfcfs_scheduler(config, dm, stats){
    //nothing special
}



dram_req_t *frlp_scheduler::schedule( unsigned bank, unsigned curr_row )
{
    //row
    bool rowhit = true;
    std::list<dram_req_t*> *m_current_queue = m_queue;
    std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> > *m_current_bins = m_bins ;
    std::list<std::list<dram_req_t*>::iterator> **m_current_last_row = m_last_row;

    if(m_config->seperate_write_queue_enabled) {
        if(m_mode == READ_MODE &&
                ((m_num_write_pending >= m_config->write_high_watermark )
                 // || (m_queue[bank].empty() && !m_write_queue[bank].empty())
                )) {
            m_mode = WRITE_MODE;
        }
        else if(m_mode == WRITE_MODE &&
                (( m_num_write_pending < m_config->write_low_watermark )
                 //  || (!m_queue[bank].empty() && m_write_queue[bank].empty())
                )){
            m_mode = READ_MODE;
        }
    }

    if(m_mode == WRITE_MODE) {
        m_current_queue = m_write_queue;
        m_current_bins = m_write_bins ;
        m_current_last_row = m_last_write_row;
    }

    if ( m_current_last_row[bank] == NULL ) {
        if ( m_current_queue[bank].empty() ){
            return NULL;
        }

        //find the min pressure in the queue
        int64_t min_pressure = std::numeric_limits<int64_t>::max();
        dram_req_t* min_it = m_current_queue[bank].back(); //default to fcfs
        for (dram_req_t* q_item : m_current_queue[bank]){
            if (q_item->data->get_sid() == -1){
                min_it = q_item;
                break;
            }
            unsigned clid = q_item->data->get_sid() / m_dram->m_gpu->m_shader_config->n_simt_cores_per_cluster;
            unsigned sid = q_item->data->get_sid() % m_dram->m_gpu->m_shader_config->n_simt_cores_per_cluster;
            int64_t pressure = m_dram->m_gpu->m_cluster[clid]->get_cores()[sid]->mshr_pressure;
            if (pressure < min_pressure){
                min_pressure = pressure;
                min_it = q_item;
            }
        }
        dram_req_t *req = min_it;
        auto bin_ptr = m_current_bins[bank].find( req->row );
        assert( bin_ptr != m_current_bins[bank].end() ); // where did the request go???
        m_current_last_row[bank] = &(bin_ptr->second);
        data_collection(bank);
        rowhit = false;
    } else {
        rowhit = true;
    }

    std::list<dram_req_t*>::iterator next = m_current_last_row[bank]->back();
    dram_req_t *req = (*next);

    //rowblp stats
    m_dram->access_num++;
    bool is_write = req->data->is_write();
        if(is_write)
            m_dram->write_num++;
        else
            m_dram->read_num++;

    if(rowhit) {
        m_dram->hits_num++;
        if(is_write)
            m_dram->hits_write_num++;
        else
            m_dram->hits_read_num++;
    }

    m_stats->concurrent_row_access[m_dram->id][bank]++;
    m_stats->row_access[m_dram->id][bank]++;
    m_current_last_row[bank]->pop_back();

    m_current_queue[bank].erase(next);
    if ( m_current_last_row[bank]->empty() ) {
        m_current_bins[bank].erase( req->row );
        m_current_last_row[bank] = NULL;
    }
#ifdef DEBUG_FAST_IDEAL_SCHED
    if ( req )
        printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n", 
                (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row );
#endif

    if(m_config->seperate_write_queue_enabled && req->data->is_write()) {
        assert( req != NULL && m_num_write_pending != 0 );
        m_num_write_pending--;
    }
    else {
        assert( req != NULL && m_num_pending != 0 );
        m_num_pending--;
        //std::cout << m_num_pending << std::endl;
    }

    return req;
}


frmpB_scheduler::frmpB_scheduler( const memory_config *config, dram_t *dm, memory_stats_t *stats) : frfcfs_scheduler(config, dm, stats){
    //nothing special
}


dram_req_t *frmpB_scheduler::schedule( unsigned bank, unsigned curr_row )
{
    //row
    bool rowhit = true;
    std::list<dram_req_t*> *m_current_queue = m_queue;
    std::map<unsigned,std::list<std::list<dram_req_t*>::iterator> > *m_current_bins = m_bins ;
    std::list<std::list<dram_req_t*>::iterator> **m_current_last_row = m_last_row;

    if(m_config->seperate_write_queue_enabled) {
        if(m_mode == READ_MODE &&
                ((m_num_write_pending >= m_config->write_high_watermark )
                 // || (m_queue[bank].empty() && !m_write_queue[bank].empty())
                )) {
            m_mode = WRITE_MODE;
        }
        else if(m_mode == WRITE_MODE &&
                (( m_num_write_pending < m_config->write_low_watermark )
                 //  || (!m_queue[bank].empty() && m_write_queue[bank].empty())
                )){
            m_mode = READ_MODE;
        }
    }

    if(m_mode == WRITE_MODE) {
        m_current_queue = m_write_queue;
        m_current_bins = m_write_bins ;
        m_current_last_row = m_last_write_row;
    }

    if ( m_current_last_row[bank] == NULL ) {
        if ( m_current_queue[bank].empty() ){
            return NULL;
        }
 
        //find the max pressure in the queue
        int64_t mx_pressure = 0;
        dram_req_t* mx_it = m_current_queue[bank].back(); //default to fcfs
        dram_req_t* q_item = nullptr;
        for (auto it = m_current_queue[bank].rbegin(); it != m_current_queue[bank].rend(); ++it){
            q_item = *it;
            
            if ( ((gpu_sim_cycle + gpu_tot_sim_cycle) - q_item->timestamp) > (m_config->gpgpu_frfcfs_dram_write_queue_size) ){
                mx_it = q_item;
                break;
            } 
            if (q_item->data->get_sid() == -1){
                continue;
            }

            unsigned clid = q_item->data->get_sid() / m_dram->m_gpu->m_shader_config->n_simt_cores_per_cluster;
            unsigned sid = q_item->data->get_sid() % m_dram->m_gpu->m_shader_config->n_simt_cores_per_cluster;
            shader_core_ctx* sc = m_dram->m_gpu->m_cluster[clid]->get_cores()[sid];
            int64_t pressure = sc->mshr_pressure * 20 + sc->get_mshr_size();
            //std::cout << "p,c,s" << pressure << "," << clid << "," << sid << std::endl;
            if (pressure > mx_pressure){
                mx_pressure = pressure;
                mx_it = q_item;
            }
        }
        dram_req_t *req = mx_it;
        auto bin_ptr = m_current_bins[bank].find( req->row );
        assert( bin_ptr != m_current_bins[bank].end() ); // where did the request go???
        m_current_last_row[bank] = &(bin_ptr->second);
        data_collection(bank);
        rowhit = false;
    } else {
        rowhit = true;
    }

    std::list<dram_req_t*>::iterator next = m_current_last_row[bank]->back();
    dram_req_t *req = (*next);

    //rowblp stats
    m_dram->access_num++;
    bool is_write = req->data->is_write();
        if(is_write)
            m_dram->write_num++;
        else
            m_dram->read_num++;

    if(rowhit) {
        m_dram->hits_num++;
        if(is_write)
            m_dram->hits_write_num++;
        else
            m_dram->hits_read_num++;
    }

    m_stats->concurrent_row_access[m_dram->id][bank]++;
    m_stats->row_access[m_dram->id][bank]++;
    m_current_last_row[bank]->pop_back();

    m_current_queue[bank].erase(next);
    if ( m_current_last_row[bank]->empty() ) {
        m_current_bins[bank].erase( req->row );
        m_current_last_row[bank] = NULL;
    }
#ifdef DEBUG_FAST_IDEAL_SCHED
    if ( req )
        printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n", 
                (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row );
#endif

    if(m_config->seperate_write_queue_enabled && req->data->is_write()) {
        assert( req != NULL && m_num_write_pending != 0 );
        m_num_write_pending--;
    }
    else {
        assert( req != NULL && m_num_pending != 0 );
        m_num_pending--;
        //std::cout << m_num_pending << std::endl;
    }

    return req;
}


clams_scheduler::clams_scheduler( const memory_config *config, dram_t *dm, memory_stats_t *stats) : frfcfs_scheduler(config, dm, stats){
    //nothing special
}

dram_req_t *clams_scheduler::schedule( unsigned bank, unsigned curr_row )
{
    //row
    bool rowhit = true;
    std::list<dram_req_t*> *m_current_queue = m_queue;

    if(m_config->seperate_write_queue_enabled) {
        if(m_mode == READ_MODE &&
                ((m_num_write_pending >= m_config->write_high_watermark )
                 // || (m_queue[bank].empty() && !m_write_queue[bank].empty())
                )) {
            m_mode = WRITE_MODE;
        }
        else if(m_mode == WRITE_MODE &&
                (( m_num_write_pending < m_config->write_low_watermark )
                 //  || (!m_queue[bank].empty() && m_write_queue[bank].empty())
                )){
            m_mode = READ_MODE;
        }
    }

    if(m_mode == WRITE_MODE) {
        m_current_queue = m_write_queue;
    }

     //find the max pressure in the queue
       
    if ( m_current_queue[bank].empty() ){
        return NULL;
    }
    std::map<int, std::pair<int, dram_req_t*> > qid_count;

    std::cout << "q:[" ;
    for (auto q_item = m_current_queue[bank].rbegin(); q_item != m_current_queue[bank].rend(); ++q_item){
        std::cout << (*q_item)->data->get_wid() << "-" << (*q_item)->data->get_sid() << " , ";
        int qid = (*q_item)->data->get_wid() + (*q_item)->data->get_sid();
        auto f_it = qid_count.find(qid);
        if (f_it != qid_count.end()){
            f_it->second.first++;
        } else {
            qid_count[qid] = std::make_pair(0,*q_item);
        }
    }
    std::cout << "]" << std::endl;

    int min_v = std::numeric_limits<int>::max();
    dram_req_t *req = nullptr; //last item in queue by default
    for (auto it = qid_count.cbegin(); it != qid_count.cend(); ++it){
        if (it->second.first < min_v){
            min_v = it->second.first;
            req = it->second.second;
        }
    }

    if (req == nullptr){
        assert(0);
    }
    
    std::cout << "sch: wid, sid" << req->data->get_wid() << "," << req->data->get_sid() << std::endl; 
 
    data_collection(bank);
    rowhit = false;

    //rowblp stats
    m_dram->access_num++;
    bool is_write = req->data->is_write();
        if(is_write)
            m_dram->write_num++;
        else
            m_dram->read_num++;

    m_current_queue[bank].remove(req);
   
    if(m_config->seperate_write_queue_enabled && req->data->is_write()) {
        assert( req != NULL && m_num_write_pending != 0 );
        m_num_write_pending--;
    }
    else {
        assert( req != NULL && m_num_pending != 0 );
        m_num_pending--;
    }

    return req;
}



/*
 *=======
 if(m_config->seperate_write_queue_enabled && req->data->is_write()) {
 assert(m_num_write_pending < m_config->gpgpu_frfcfs_dram_write_queue_size);
 m_num_write_pending++;
 if (req->data->get_sid() < 40){
 m_write_queue[req->bk].push_back(req);
 std::list<dram_req_t*>::iterator ptr = std::prev(m_write_queue[req->bk].end());
 m_write_bins[req->bk][req->row].push_back( ptr ); //newest reqs to the front
 } else {
 m_write_queue[req->bk].push_front(req);
 std::list<dram_req_t*>::iterator ptr = m_write_queue[req->bk].begin();
 m_write_bins[req->bk][req->row].push_front( ptr ); //newest reqs to the front
 }

 } else {
 assert(m_num_pending < m_config->gpgpu_frfcfs_dram_sched_queue_size);
 m_num_pending++;

 if (req->data->get_sid() < 40){
 m_queue[req->bk].push_back(req);
 std::list<dram_req_t*>::iterator ptr = std::prev(m_queue[req->bk].end());
 m_bins[req->bk][req->row].push_back( ptr ); //newest reqs to the front
 } else {
 m_queue[req->bk].push_front(req);
 std::list<dram_req_t*>::iterator ptr = m_queue[req->bk].begin();
 m_bins[req->bk][req->row].push_front( ptr ); //newest reqs to the front
 }
 }
 >>>>>>> Stashed changes
 */
