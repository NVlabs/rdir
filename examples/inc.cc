/*
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

#include "inc.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;

///
/// Mapper
///

LegionRuntime::Logger::Category log_mapper("mapper");

class GpuTaskMapper : public DefaultMapper
{
public:
  GpuTaskMapper(Machine machine, HighLevelRuntime *rt, Processor local);
  virtual void select_task_options(Task *task);
  virtual void slice_domain(const Task *task, const Domain &domain,
                            std::vector<DomainSplit> &slices);
  virtual void select_task_variant(Task *task);
  virtual bool map_task(Task *task);
  virtual bool map_inline(Inline *inline_operation);
  virtual void notify_mapping_failed(const Mappable *mappable);
private:
  Color get_task_color_by_region(Task *task, const RegionRequirement &requirement);
  LogicalRegion get_root_region(LogicalRegion handle);
  LogicalRegion get_root_region(LogicalPartition handle);
private:
  std::vector<Processor> all_cpus;
  std::vector<Processor> all_gpus;
  std::map<Processor, Memory> all_sysmems;
  bool map_to_gpu;
  std::set<std::string> gpu_tasks;
  std::set<std::string> inner_tasks;
};

GpuTaskMapper::GpuTaskMapper(Machine machine, HighLevelRuntime *rt, Processor local)
  : DefaultMapper(machine, rt, local)
{
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  for (std::set<Processor>::const_iterator it = all_procs.begin();
        it != all_procs.end(); it++)
  {
    Processor::Kind k = it->kind();
    switch (k)
    {
      case Processor::LOC_PROC:
        all_cpus.push_back(*it);
        break;
      case Processor::TOC_PROC:
        all_gpus.push_back(*it);
        break;
      default:
        break;
    }
  }
  map_to_gpu = !all_gpus.empty();
  printf("map_to_gpu %d\n", map_to_gpu);
  {
    for (std::vector<Processor>::iterator itr = all_cpus.begin();
         itr != all_cpus.end(); ++itr) {
      Memory sysmem = machine_interface.find_memory_kind(*itr, Memory::SYSTEM_MEM);
      all_sysmems[*itr] = sysmem;
    }
  }

  gpu_tasks.insert("inc");
  gpu_tasks.insert("__fused__inc__inc");
  gpu_tasks.insert("dummy");

  inner_tasks.insert("inc1");
  inner_tasks.insert("inc2");
  inner_tasks.insert("dummy1");
}

void GpuTaskMapper::select_task_options(Task *task)
{
  if (map_to_gpu && gpu_tasks.count(task->variants->name))
  {
    DefaultMapper::select_task_options(task);
    task->target_proc = all_gpus[0];
  }
  else
  {
    task->inline_task = false;
    task->spawn_task = false;
    task->map_locally = true;
    task->profile_task = false;
  }
}

void GpuTaskMapper::slice_domain(const Task *task, const Domain &domain,
                                 std::vector<DomainSplit> &slices)
{
  if (map_to_gpu && gpu_tasks.count(task->variants->name))
  {
    decompose_index_space(domain, all_gpus, 1/*splitting factor*/, slices);
  }
  else
  {
    decompose_index_space(domain, all_cpus, 1/*splitting factor*/, slices);
  }
}

void GpuTaskMapper::select_task_variant(Task *task)
{
  // printf("Task name: %s, Task id: %u, Target proc: " IDFMT ", #variants: %lu\n",
  //     task->variants->name, task->task_id, task->target_proc.id, task->variants->get_all_variants().size());
  if (map_to_gpu && gpu_tasks.count(task->variants->name))
  {
    task->selected_variant =
      task->variants->get_variant(task->target_proc.kind(), true, true);
  }
  else
  {
    // Use the SOA variant for all tasks.
    DefaultMapper::select_task_variant(task);
  }

  std::vector<RegionRequirement> &regions = task->regions;
  for (std::vector<RegionRequirement>::iterator it = regions.begin();
        it != regions.end(); it++) {
    RegionRequirement &req = *it;

    // Select SOA layout for all regions.
    req.blocking_factor = req.max_blocking_factor;
  }
}

bool GpuTaskMapper::map_task(Task *task)
{
  bool map_virtual = false;
  // task->variants->get_variant(task->selected_variant).inner;
  if (inner_tasks.count(task->variants->name)) {
    map_virtual = true;
  }

  if (map_to_gpu && gpu_tasks.count(task->variants->name))
  {
    // Otherwise do custom mappings for GPU memories
    Memory zc_mem = machine_interface.find_memory_kind(task->target_proc,
                                                       Memory::Z_COPY_MEM);
    assert(zc_mem.exists());
    Memory fb_mem = machine_interface.find_memory_kind(task->target_proc,
                                                       Memory::GPU_FB_MEM);
    assert(fb_mem.exists());
    for (unsigned idx = 0; idx < task->regions.size(); idx++)
    {
      task->regions[idx].target_ranking.push_back(fb_mem);
      task->regions[idx].virtual_map = map_virtual;
      task->regions[idx].enable_WAR_optimization = war_enabled;
      task->regions[idx].reduction_list = false;
      // Make everything SOA
      task->regions[idx].blocking_factor =
        task->regions[idx].max_blocking_factor;
    }
  }
  else
  {
    Memory sys_mem = all_sysmems[task->target_proc];
    assert(sys_mem.exists());
    for (unsigned idx = 0; idx < task->regions.size(); idx++)
    {
      task->regions[idx].target_ranking.push_back(sys_mem);
      task->regions[idx].virtual_map = map_virtual;
      task->regions[idx].enable_WAR_optimization = war_enabled;
      task->regions[idx].reduction_list = false;
      // Make everything SOA
      task->regions[idx].blocking_factor =
        task->regions[idx].max_blocking_factor;
    }
  }

  return false;
}

bool GpuTaskMapper::map_inline(Inline *inline_operation)
{
  Memory sysmem = all_sysmems[local_proc];
  RegionRequirement &req = inline_operation->requirement;

  // Region options:
  req.virtual_map = false;
  req.enable_WAR_optimization = false;
  req.reduction_list = false;
  req.blocking_factor = req.max_blocking_factor;

  // Place all regions in global memory.
  req.target_ranking.push_back(sysmem);

  return false;
}

void GpuTaskMapper::notify_mapping_failed(const Mappable *mappable)
{
  switch (mappable->get_mappable_kind()) {
  case Mappable::TASK_MAPPABLE:
    {
      log_mapper.warning("mapping failed on task");
      break;
    }
  case Mappable::COPY_MAPPABLE:
    {
      log_mapper.warning("mapping failed on copy");
      break;
    }
  case Mappable::INLINE_MAPPABLE:
    {
      //Inline *_inline = mappable->as_mappable_inline();
      //RegionRequirement &req = _inline->requirement;
      //LogicalRegion region = req.region;
      break;
    }
  case Mappable::ACQUIRE_MAPPABLE:
    {
      log_mapper.warning("mapping failed on acquire");
      break;
    }
  case Mappable::RELEASE_MAPPABLE:
    {
      log_mapper.warning("mapping failed on release");
      break;
    }
  }
  assert(0 && "mapping failed");
}

Color GpuTaskMapper::get_task_color_by_region(Task *task, const RegionRequirement &requirement)
{
  if (requirement.handle_type == SINGULAR) {
    return get_logical_region_color(requirement.region);
  }
  return 0;
}

LogicalRegion GpuTaskMapper::get_root_region(LogicalRegion handle)
{
  if (has_parent_logical_partition(handle)) {
    return get_root_region(get_parent_logical_partition(handle));
  }
  return handle;
}

LogicalRegion GpuTaskMapper::get_root_region(LogicalPartition handle)
{
  return get_root_region(get_parent_logical_region(handle));
}

static void create_mappers(Machine machine, HighLevelRuntime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    runtime->replace_default_mapper(new GpuTaskMapper(machine, runtime, *it), *it);
  }
}

void register_mappers()
{
  HighLevelRuntime::set_registration_callback(create_mappers);
}
