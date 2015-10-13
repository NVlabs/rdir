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

-- Dataflow IR

local ast = require("regent/ast")

local flow = {}

-- Dataflow Graph
local graph = setmetatable({}, {
    __index = function(t, k) error("graph has no field " .. tostring(k), 2) end
})
graph.__index = graph

function flow.empty_graph(region_tree)
  return setmetatable(
    {
      region_tree = region_tree,

      next_node = 1,
      -- node ID -> node label
      nodes = {},
      -- from node ID -> to node ID -> [(from port, to port, edge label)]
      edges = {},
      -- to node ID -> from node ID -> [(to port, from port, edge label)]
      backedges = {},
    }, graph)
end

function flow.is_graph(x)
  return getmetatable(x) == graph
end

function flow.null()
  return 0
end

function flow.is_null(node)
  return node == 0
end

function flow.is_valid_node(node)
  return node and node > 0
end

function flow.is_opaque_node(label)
  return label:is(flow.node.Opaque) or (
    label:is(flow.node.Task) and label.opaque) or
    -- FIXME: Depends on contents of subgraph.
    label:is(flow.node.ForNum) or label:is(flow.node.ForList)
end

function graph:node_label(node)
  assert(flow.is_valid_node(node))
  return self.nodes[node]
end

function graph:set_node_label(node, label)
  assert(rawget(self.nodes, node))
  self.nodes[node] = label
end

function graph:node_result_port(node)
  return -1
end

function graph:node_sync_port(node)
  return -2
end

function graph:copy()
  local result = flow.empty_graph(self.region_tree)
  result.next_node = self.next_node
  for node, label in pairs(self.nodes) do
    result.nodes[node] = label
    result.edges[node] = {}
    result.backedges[node] = {}
  end
  for from, to_list in pairs(self.edges) do
    for to, edge_list in pairs(to_list) do
      for _, edge in pairs(edge_list) do
        result:add_edge(edge.label, from, edge.from_port, to, edge.to_port)
      end
    end
  end
  return result
end

function graph:add_node(label)
  assert(label:is(flow.node))

  local node = self.next_node
  self.next_node = self.next_node + 1

  self.nodes[node] = label
  self.edges[node] = {}
  self.backedges[node] = {}
  return node
end

function graph:remove_node(node)
  assert(rawget(self.nodes, node))

  self.nodes[node] = nil
  self.edges[node] = nil
  self.backedges[node] = nil

  for _, to_list in pairs(self.edges) do
    if rawget(to_list, node) then
      to_list[node] = nil
    end
  end
  for _, from_list in pairs(self.backedges) do
    if rawget(from_list, node) then
      from_list[node] = nil
    end
  end
end

function graph:add_edge(label, from_node, from_port, to_node, to_port)
  assert(label:is(flow.edge))
  assert(flow.is_valid_node(from_node) and flow.is_valid_node(to_node))
  if not rawget(self.edges[from_node], to_node) then
    self.edges[from_node][to_node] = terralib.newlist()
  end
  self.edges[from_node][to_node]:insert(
    {
      from_port = from_port,
      to_port = to_port,
      label = label,
  })
  if not rawget(self.backedges[to_node], from_node) then
    self.backedges[to_node][from_node] = terralib.newlist()
  end
  self.backedges[to_node][from_node]:insert(
    {
      to_port = to_port,
      from_port = from_port,
      label = label,
  })
end

function graph:map_nodes(fn)
  for node, label in pairs(self.nodes) do
    fn(node, label)
  end
end

function graph:filter_nodes(fn)
  local result = terralib.newlist()
  for node, label in pairs(self.nodes) do
    if fn(node, label) then
      result:insert(node)
    end
  end
  return result
end

function graph:any_nodes(fn)
  for node, label in pairs(self.nodes) do
    if fn(node, label) then
      return true
    end
  end
  return false
end

function graph:map_edges(fn)
  for from, to_list in pairs(self.edges) do
    for to, edge_list in pairs(to_list) do
      for _, edge in pairs(edge_list) do
        fn(from, edge.from_port, self.nodes[from],
           to, edge.to_port, self.nodes[to],
           edge.label)
      end
    end
  end
end

function graph:incoming_edges_by_port(node)
  local result = {}
  for from_node, edges in pairs(self.backedges[node]) do
    for _, edge in pairs(edges) do
      if not rawget(result, edge.to_port) then
        result[edge.to_port] = terralib.newlist()
      end
      result[edge.to_port]:insert(
        {
          from_node = from_node,
          from_port = edge.from_port,
          to_node = node,
          to_port = edge.to_port,
          label = edge.label,
      })
    end
  end
  return result
end

function graph:outgoing_edges_by_port(node)
  local result = {}
  for to_node, edges in pairs(self.edges[node]) do
    for _, edge in pairs(edges) do
      if not rawget(result, edge.to_port) then
        result[edge.from_port] = terralib.newlist()
      end
      result[edge.from_port]:insert(
        {
          from_node = node,
          from_port = edge.from_port,
          to_node = to_node,
          to_port = edge.to_port,
          label = edge.label,
      })
    end
  end
  return result
end

function graph:immediate_predecessor(node)
  local pred
  for from_node, _ in pairs(self.backedges[node]) do
    assert(not pred)
    pred = from_node
  end
  assert(pred)
  return pred
end

function graph:immediate_successor(node)
  local succ
  for to_node, _ in pairs(self.edges[node]) do
    assert(not succ)
    succ = to_node
  end
  assert(succ)
  return succ
end

function graph:immediate_predecessors(node)
  local result = terralib.newlist()
  for from_node, _ in pairs(self.backedges[node]) do
    result:insert(from_node)
  end
  return result
end

function graph:immediate_successors(node)
  local result = terralib.newlist()
  for to_node, _ in pairs(self.edges[node]) do
    result:insert(to_node)
  end
  return result
end

function graph:incoming_read_set(node)
  local result = terralib.newlist()
  for from_node, edges in pairs(self.backedges[node]) do
    for _, edge in pairs(edges) do
      if edge.label:is(flow.edge.Read) then
        result:insert(from_node)
        break
      end
    end
  end
  return result
end

function graph:incoming_write_set(node)
  local result = terralib.newlist()
  for from_node, edges in pairs(self.backedges[node]) do
    for _, edge in pairs(edges) do
      if edge.label:is(flow.edge.Write) then
        result:insert(from_node)
        break
      end
    end
  end
  return result
end

function graph:outgoing_read_set(node)
  local result = terralib.newlist()
  for to_node, edges in pairs(self.edges[node]) do
    for _, edge in pairs(edges) do
      if edge.label:is(flow.edge.Read) then
        result:insert(to_node)
        break
      end
    end
  end
  return result
end

function graph:outgoing_use_set(node)
  local result = terralib.newlist()
  for to_node, edges in pairs(self.edges[node]) do
    for _, edge in pairs(edges) do
      if edge.label:is(flow.edge.None) or edge.label:is(flow.edge.Read) then
        result:insert(to_node)
        break
      end
    end
  end
  return result
end

function graph:outgoing_write_set(node)
  local result = terralib.newlist()
  for to_node, edges in pairs(self.edges[node]) do
    for _, edge in pairs(edges) do
      if edge.label:is(flow.edge.Write) or edge.label:is(flow.edge.Name) then
        result:insert(to_node)
        break
      end
    end
  end
  return result
end

local function dfs(graph, node, predicate, visited)
  if predicate(node) then
    return true
  end

  if rawget(visited, node) then
    return false
  end
  visited[node] = true

  for to_node, edges in pairs(graph.edges[node]) do
    if dfs(graph, to_node, predicate, visited) then
      return true
    end
  end
  return false
end

function graph:reachable(src_node, dst_node)
  return dfs(self, src_node, function(node) return node == dst_node end, {})
end

function graph:between_nodes(src_node, dst_node)
  local result = terralib.newlist()
  dfs(
    self, src_node, function(node)
      if node ~= src_node and node ~= dst_node and
        dfs(self, node, function(other) return other == dst_node end, {})
      then
        result:insert(node)
      end
    end, {})
  return result
end

local function toposort_node(graph, node, visited, path, sort)
  if rawget(path, node) then
    error("cycle in toposort")
  end
  if not rawget(visited, node) then
    path[node] = true
    for _, child in pairs(graph:immediate_successors(node)) do
      toposort_node(graph, child, visited, path, sort)
    end
    path[node] = false
    visited[node] = true
    sort:insert(node)
  end
end

local function inverse_toposort_node(graph, node, visited, path, sort)
  if rawget(path, node) then
    error("cycle in inverse_toposort")
  end
  if not rawget(visited, node) then
    path[node] = true
    for _, child in pairs(graph:immediate_predecessors(node)) do
      inverse_toposort_node(graph, child, visited, path, sort)
    end
    path[node] = false
    visited[node] = true
    sort:insert(node)
  end
end

local function reverse(list)
  local result = terralib.newlist()
  for i = #list, 1, -1 do
    result:insert(list[i])
  end
  return result
end

function graph:toposort()
  local visited = {}
  local sort = terralib.newlist()
  self:map_nodes(
    function(node)
      toposort_node(self, node, visited, {}, sort)
  end)
  return reverse(sort)
end

function graph:inverse_toposort()
  local visited = {}
  local sort = terralib.newlist()
  self:map_nodes(
    function(node)
      inverse_toposort_node(self, node, visited, {}, sort)
  end)
  return reverse(sort)
end

function graph:printpretty()
  print("digraph {")
  print("rankdir = LR;")
  print("node [ margin = \"0.055,0.0275\" ];")
  self:map_nodes(function(i, node)
    local label = tostring(node:type()):gsub("[^.]+[.]", ""):lower()
    if node:is(flow.node.Region) or node:is(flow.node.Partition) or
      node:is(flow.node.Scalar) or
      node:is(flow.node.Constant) or node:is(flow.node.Function)
    then
      label = label .. " " .. tostring(node.value.value) .. " " .. tostring(node.value.expr_type)
    end
    local shape
    if node:is(flow.node.Opaque) or node:is(flow.node.Deref) or
      node:is(flow.node.Reduce) or node:is(flow.node.Task) or
      node:is(flow.node.IndexAccess) or
      node:is(flow.node.ForNum) or node:is(flow.node.ForList)
    then
      shape = "rectangle"
    elseif node:is(flow.node.Open) or node:is(flow.node.Close) then
      shape = "diamond"
    elseif node:is(flow.node.Region) or node:is(flow.node.Scalar) or
      node:is(flow.node.Constant) or node:is(flow.node.Function)
    then
      shape = "ellipse"
    elseif node:is(flow.node.Partition) then
      shape = "octagon"
    else
      print(node)
      assert(false)
    end
    print(tostring(i) .. " [ label = \"" .. label .. "\", shape = " .. shape .. " ];")
  end)
  self:map_edges(
    function(from, from_port, from_label, to, to_port, to_label, edge)
      local label = tostring(edge:type()):gsub("[^.]+[.]", ""):lower()
      local style = "solid"
      if edge:is(flow.edge.HappensBefore) then
        style = "dotted"
      end
      print(tostring(from) .. " -> " .. tostring(to) ..
              " [ label = " .. label .. ", style = " .. style .. " ];")
  end)
  print("}")
end

-- Dataflow Graph: Nodes
flow.node = ast.factory("flow.node")

-- Compute
flow.node("Opaque", {"action"})

flow.node("IndexAccess", {"expr_type", "span"})
flow.node("Deref", {"expr_type", "span"})
flow.node("Reduce", {"op", "span"})
flow.node("Task", {"opaque", "expr_type", "span"})

flow.node("Open", {})
flow.node("Close", {})

-- Control
flow.node("ForNum", {"symbol", "block", "parallel", "span"})
flow.node("ForList", {"symbol", "block", "vectorize", "span"})

-- Data
flow.node("Region", {"value"})
flow.node("Partition", {"value"})
flow.node("Scalar", {"value", "fresh"})
flow.node("Constant", {"value"})
flow.node("Function", {"value"})

-- Dataflow Graph: Edges
flow.edge = ast.factory("flow.edge")

flow.edge("HappensBefore", {})
flow.edge("Name", {})

flow.edge("None", {})
flow.edge("Read", {})
flow.edge("Discard", {})
flow.edge("Write", {})
flow.edge("Reduce", {"op"})

return flow
