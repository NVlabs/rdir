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

-- Dead Code Elimination for Dataflow IR

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

local flow_dead_code_elimination = {}

function is_dead_node(cx, nid)
  if #cx.graph:immediate_successors(nid) == 0 then
    local label = cx.graph:node_label(nid)
    return label:is(flow.node.Deref) or
      label:is(flow.node.IndexAccess) or
      label:is(flow.node.Constant) or
      label:is(flow.node.Function)
  end
end

function remove_dead_nodes(cx)
  local nids = cx.graph:inverse_toposort()
  for _, nid in ipairs(nids) do
    local label = cx.graph:node_label(nid)
    if label:is(flow.node.ForNum) or label:is(flow.node.ForList) then
      local block_cx = cx:new_graph_scope(label.block:copy())
      remove_dead_nodes(block_cx)
      cx.graph:set_node_label(nid, label { block = block_cx.graph })
    end
    if is_dead_node(cx, nid) then
      cx.graph:remove_node(nid)
    end
  end
end

function flow_dead_code_elimination.graph(cx, graph)
  assert(flow.is_graph(graph))
  local cx = cx:new_graph_scope(graph:copy())
  remove_dead_nodes(cx)
  return cx.graph
end

function flow_dead_code_elimination.stat_task(cx, node)
  return node { body = flow_dead_code_elimination.graph(cx, node.body) }
end

function flow_dead_code_elimination.stat_top(cx, node)
  if node:is(ast.typed.StatTask) then
    return flow_dead_code_elimination.stat_task(cx, node)

  else
    return node
  end
end

function flow_dead_code_elimination.entry(node)
  local cx = context.new_global_scope()
  return flow_dead_code_elimination.stat_top(cx, node)
end

return flow_dead_code_elimination
