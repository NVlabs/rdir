-- Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
--
-- Redistribution and use in source and binary forms, with or without
-- modification, are permitted provided that the following conditions
-- are met:
--  * Redistributions of source code must retain the above copyright
--    notice, this list of conditions and the following disclaimer.
--  * Redistributions in binary form must reproduce the above copyright
--    notice, this list of conditions and the following disclaimer in the
--    documentation and/or other materials provided with the distribution.
--  * Neither the name of NVIDIA CORPORATION nor the names of its
--    contributors may be used to endorse or promote products derived
--    from this software without specific prior written permission.
--
-- THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
-- EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
-- IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
-- PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
-- CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
-- EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
-- PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
-- PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
-- OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
-- OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

-- Loop Fusion Optimization for Dataflow IR

local ast = require("regent/ast")
local flow = require("regent/flow")
local flow_region_tree = require("regent/flow_region_tree")
local std = require("regent/std")

local context = {}
context.__index = context

function context:new_graph_scope(graph)
  local cx = {
    tree = graph.region_tree,
    graph = graph,
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

local flow_loop_fusion = {}

local function has_same_node_type(cx, nid1, nid2)
  return cx.graph:node_label(nid1).node_type ==
    cx.graph:node_label(nid2).node_type
end

local function has_compute_nodes_between(cx, nid1, nid2)
  local between = cx.graph:between_nodes(nid1, nid2)
  for _, nid in ipairs(between) do
    local label = cx.graph:node_label(nid)
    if not (label:is(flow.node.Region) or label:is(flow.node.Partition) or
              label:is(flow.node.Scalar))
    then
      return true
    end
  end
  return false
end

local function has_opaque_nodes_inside(cx, nid)
  return cx.graph:node_label(nid).block:any_nodes(
    function(_, label) return flow.is_opaque_node(label) end)
end

local function only_reads(cx, nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  for _, edges in pairs(inputs) do
    for _, edge in ipairs(edges) do
      if not (edge.label:is(flow.edge.Name) or
                edge.label:is(flow.edge.None) or
                edge.label:is(flow.edge.Read))
      then
        return false
      end
    end
  end
    return true
end

local function has_loop_carry_dependence(cx, nid)
  local label = cx.graph:node_label(nid)
  local cx = cx:new_graph_scope(label.block)
  local data = cx.graph:filter_nodes(
    function(nid, label)
      return label:is(flow.node.Region) or label:is(flow.node.Partition) or
        label:is(flow.node.Scalar)
  end)
  for i1, nid1 in ipairs(data) do
    for i2, nid2 in ipairs(data) do
      if nid1 ~= nid2 then
        local label1 = cx.graph:node_label(nid1)
        local label2 = cx.graph:node_label(nid2)
        if label1.node_type == label2.node_type then
          if label1:is(flow.node.Region) or label1:is(flow.node.Partition) then
            local region1 = std.as_read(label1.value.expr_type)
            local region2 = std.as_read(label2.value.expr_type)
            if std.type_eq(region1, region2) then
              if cx.tree:has_region_index(region1) then
                local index = cx.tree:region_index(region1)
                if index:is(ast.typed.ExprID) and index.value == label.symbol then
                  return cx.tree:aliased(cx.tree:parent(region1))
                end
              end
            else
              if not cx.tree:can_alias(region1, region2) then
                return true
              end
            end
            return only_reads(cx, nid1) and only_reads(cx, nid2)
          elseif label1:is(flow.node.Scalar) then
            if label1.value.value == label2.value.value then
              return true
            end
          end
        end
      end
    end
  end
  return false
end

local function loop_bounds_match(cx, nid1, nid2)
  local inputs1 = cx.graph:incoming_edges_by_port(nid1)
  local inputs2 = cx.graph:incoming_edges_by_port(nid2)
  local num_values
  if cx.graph:node_label(nid1):is(flow.node.ForNum) then
    num_values = 3
  elseif cx.graph:node_label(nid1):is(flow.node.ForList) then
    num_values = 1
  else
    assert(false)
  end
  for i = 0, num_values do
    if rawget(inputs1, i) then
      assert(rawget(inputs1, i) and #inputs1[i] == 1)
      assert(rawget(inputs2, i) and #inputs2[i] == 1)

      local input1_label = cx.graph:node_label(inputs1[i][1].from_node)
      local input2_label = cx.graph:node_label(inputs2[i][1].from_node)
      if input1_label.value.value ~= input2_label.value.value then
        return false
      end
    end
  end
  return true
end

local function can_fuse(cx, nid1, nid2)
  return has_same_node_type(cx, nid1, nid2) and
    not has_compute_nodes_between(cx, nid1, nid2) and
    not has_opaque_nodes_inside(cx, nid1) and
    not has_opaque_nodes_inside(cx, nid2) and
    not has_loop_carry_dependence(cx, nid1) and
    not has_loop_carry_dependence(cx, nid2) and
    loop_bounds_match(cx, nid1, nid2)
end

local function map_region(cx, index1, index2, region_type)
  local index = cx.tree:has_region_index(region_type)
  if index and index:is(ast.typed.ExprID) and index.value == index1 then
    local parent = cx.tree:parent(region_type)
    return parent:subregion_constant(index2)
  else
    return region_type
  end
end

local function fuse(cx, loop1, loop2)
  local loop1_label = cx.graph:node_label(loop1)
  local loop2_label = cx.graph:node_label(loop2)

  -- Construct the body of the loop:

  -- Copy first loop verbatim.
  local new_cx = cx:new_graph_scope(loop1_label.block:copy())

  -- Copy second loop.
  local loop2_cx = cx:new_graph_scope(loop2_label.block)

  local mapping = {}
  for _, loop2_nid in ipairs(loop2_cx.graph:toposort()) do
    local label = loop2_cx.graph:node_label(loop2_nid)
    local inputs = loop2_cx.graph:incoming_edges_by_port(loop2_nid)
    local new_nid
    local needs_edges = true
    if label:is(flow.node.Region) or label:is(flow.node.Partition) then
      local has_write = false
      for _, input in pairs(inputs) do
        for _, edge in pairs(input) do
          if edge.label:is(flow.edge.Write) then
            has_write = true
          end
        end
      end
      local region_type = map_region(
        new_cx, loop2_label.symbol, loop1_label.symbol, std.as_read(label.value.expr_type))
      if not has_write then
        local new_nids = new_cx.graph:filter_nodes(
          function(nid1, label1)
            return (label1:is(flow.node.Region) or label1:is(flow.node.Partition)) and
              std.as_read(label1.value.expr_type) == std.as_read(region_type) and
              not new_cx.graph:any_nodes(
                function(nid2, label2)
                  return nid1 ~= nid2 and
                    (label2:is(flow.node.Region) or label2:is(flow.node.Partition)) and
                    std.as_read(label2.value.expr_type) == std.as_read(region_type) and
                    new_cx.graph:reachable(nid1, nid2)
                end)
          end)
        assert(#new_nids <= 1)
        if #new_nids == 1 then
          new_nid = new_nids[1]
          needs_edges = false
        else
          new_nid = new_cx.graph:add_node(
            label {
              value = label.value {
                value = new_cx.tree:region_symbol(region_type),
                expr_type = region_type,
              },
            })
        end
      else
        new_nid = new_cx.graph:add_node(
          label {
            value = label.value {
              value = new_cx.tree:region_symbol(region_type),
            expr_type = region_type,
            },
          })
      end
    elseif label:is(flow.node.Scalar) then
      if label.value.value == loop2_label.symbol then
        local new_nids = new_cx.graph:filter_nodes(
          function(nid1, label1)
            return label1:is(flow.node.Scalar) and
              label1.value.value == loop1_label.symbol
          end)

        assert(#new_nids <= 1)
        if #new_nids == 1 then
          new_nid = new_nids[1]
          needs_edges = false
        else
          new_nid = new_cx.graph:add_node(
            label {
              value = label.value {
              value = loop1_label.symbol,
              },
            })
        end
      else
        new_nid = new_cx.graph:add_node(label)
      end
    else
      new_nid = new_cx.graph:add_node(label)
    end
    mapping[loop2_nid] = new_nid

    if needs_edges then
      for _, input in pairs(inputs) do
        for _, edge in pairs(input) do
          new_cx.graph:add_edge(
            edge.label,
            mapping[edge.from_node], edge.from_port,
            new_nid, edge.to_port)
        end
      end
    end
  end

  -- Create the new loop node.
  local label
  if cx.graph:node_label(loop1):is(flow.node.ForNum) then
    label = flow.node.ForNum {
      symbol = loop1_label.symbol,
      block = new_cx.graph,
      parallel = loop1_label.parallel,
      span = loop1_label.span,
    }
  elseif cx.graph:node_label(loop1):is(flow.node.ForList) then
    label = flow.node.ForList {
      symbol = loop1_label.symbol,
      block = new_cx.graph,
      vectorize = loop1_label.vectorize,
      span = loop1_label.span,
    }
  else
    assert(false)
  end
  local compute_nid = cx.graph:add_node(label)

  local num_values
  if cx.graph:node_label(loop1):is(flow.node.ForNum) then
    num_values = 3
  elseif cx.graph:node_label(loop1):is(flow.node.ForList) then
    num_values = 1
  else
    assert(false)
  end

  -- Connect inputs.
  local inputs1 = cx.graph:incoming_edges_by_port(loop1)
  for _, edges in pairs(inputs1) do
    for _, edge in ipairs(edges) do
      cx.graph:add_edge(
        edge.label,
        edge.from_node, edge.from_port,
        compute_nid, edge.to_port)
    end
  end
  local inputs2 = cx.graph:incoming_edges_by_port(loop2)
  for port, edges in pairs(inputs2) do
    if port > num_values then
      for _, edge in ipairs(edges) do
        if not (cx.graph:reachable(edge.from_node, loop1) or
                  cx.graph:reachable(loop1, edge.from_node))
        then
          cx.graph:add_edge(
            edge.label,
            edge.from_node, edge.from_port,
            compute_nid, edge.to_port)
        end
      end
    end
  end

  -- Connect outputs.
  local outputs1 = cx.graph:outgoing_edges_by_port(loop1)
  for _, edges in pairs(outputs1) do
    for _, edge in ipairs(edges) do
      if not (cx.graph:reachable(edge.from_node, loop2) or
                cx.graph:reachable(loop2, edge.from_node))
      then
        cx.graph:add_edge(
          edge.label,
          compute_nid, edge.from_port,
          edge.to_node, edge.to_port)
      end
    end
  end
  local outputs2 = cx.graph:outgoing_edges_by_port(loop2)
  for _, edges in pairs(outputs2) do
    for _, edge in ipairs(edges) do
      cx.graph:add_edge(
        edge.label,
        compute_nid, edge.from_port,
        edge.to_node, edge.to_port)
    end
  end
  return compute_nid
end

local function fuse_eligible_loop(cx, loops)
  for loop1, _ in pairs(loops) do
    for loop2, _ in pairs(loops) do
      if loop1 ~= loop2 then
        if cx.graph:reachable(loop1, loop2) then
          local x = can_fuse(cx, loop1, loop2)
          if x then
            local new_loop = fuse(cx, loop1, loop2)
            cx.graph:remove_node(loop1)
            cx.graph:remove_node(loop2)
            loops[loop1] = nil
            loops[loop2] = nil
            loops[new_loop] = true
            return true
          end
        end
      end
    end
  end
  return false
end

local function fuse_eligible_loop_pairs(cx, original_loops)
  local loops = {}
  for _, nid in pairs(original_loops) do
    loops[nid] = true
  end
  repeat until not fuse_eligible_loop(cx, loops)
end

function flow_loop_fusion.graph(cx, graph)
  assert(flow.is_graph(graph))
  local cx = cx:new_graph_scope(graph:copy())
  local for_num_loops = cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.ForNum) end)
  fuse_eligible_loop_pairs(cx, for_num_loops)
  local for_list_loops = cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.ForList) end)
  fuse_eligible_loop_pairs(cx, for_list_loops)
  return cx.graph
end

function flow_loop_fusion.stat_task(cx, node)
  return node { body = flow_loop_fusion.graph(cx, node.body) }
end

function flow_loop_fusion.stat_top(cx, node)
  if node:is(ast.typed.StatTask) then
    return flow_loop_fusion.stat_task(cx, node)

  else
    return node
  end
end

function flow_loop_fusion.entry(node)
  local cx = context.new_global_scope()
  return flow_loop_fusion.stat_top(cx, node)
end

return flow_loop_fusion
