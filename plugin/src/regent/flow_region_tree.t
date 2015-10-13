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

local std = require("regent/std")

local flow_region_tree = {}

-- For the purposes of analysis, consider partitions to be regions as
-- well.
function flow_region_tree.is_region(region_type)
  return std.is_region(region_type) or std.is_partition(region_type)
end

-- Region Tree

local region_tree = setmetatable({}, { __index = function(t, k) error("region tree has no field " .. tostring(k), 2) end})
region_tree.__index = region_tree

function flow_region_tree.new_region_tree(constraints, region_universe)
  -- Copy region_universe to allow safe modifications.
  local initial_universe = {}
  for k, v in pairs(region_universe) do
    initial_universe[k] = v
  end
  return setmetatable({
      -- Region tree structure.
      constraints = constraints,
      region_universe = initial_universe,
      interned_scalars = {},

      -- Region identity and indexing.
      region_symbols = {},
      region_var_types = {},
      region_spans = {},
      region_indices = {},
      region_is_point = {},
      region_point_partitions = {},

      -- Query cache.
      cache_ancestors = {},
      cache_children = {},
      cache_siblings = {},
  }, region_tree)
end

function region_tree:has_region_symbol(region_type)
  assert(flow_region_tree.is_region(region_type))
  return rawget(self.region_symbols, region_type)
end

function region_tree:region_symbol(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_symbols, region_type))
  return self.region_symbols[region_type]
end

function region_tree:region_var_type(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_var_types, region_type))
  return self.region_var_types[region_type]
end

function region_tree:region_span(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_spans, region_type))
  return self.region_spans[region_type]
end

function region_tree:has_region_index(region_type)
  assert(flow_region_tree.is_region(region_type))
  return rawget(self.region_indices, region_type)
end

function region_tree:region_index(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_indices, region_type))
  return self.region_indices[region_type]
end

function region_tree:point_partition(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_point_partitions, region_type))
  return self.region_point_partitions[region_type]
end

function region_tree:is_point(region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(rawget(self.region_is_point, region_type) ~= nil)
  return self.region_is_point[region_type]
end

function region_tree:intern_variable(expr_type, symbol, span)
  -- Assign a fresh region to non-region symbols.
  local value_type = std.as_read(expr_type)
  local region_type = value_type
  if not flow_region_tree.is_region(value_type) then
    if rawget(self.interned_scalars, symbol) then
      return self.interned_scalars[symbol]
    end

    region_type = std.region(region_type)
    for other, _ in pairs(self.region_universe) do
      std.add_constraint(self, region_type, other, "*", true)
    end
    self.interned_scalars[symbol] = region_type
  end
  assert(flow_region_tree.is_region(region_type))

  self.region_universe[region_type] = true
  if not rawget(self.region_symbols, region_type) then
    assert(not rawget(self.region_spans, region_type))
    self.region_symbols[region_type] = symbol
    self.region_var_types[region_type] = expr_type
    self.region_spans[region_type] = span
    self.region_is_point[region_type] = false
    if std.is_region(value_type) then
      local partition = std.partition(std.disjoint, symbol)
      self.region_point_partitions[region_type] = partition
      std.add_constraint(self, partition, region_type, "<=", false)
      self:intern_region_expr(partition, span)
    end
  end
  return region_type
end

function region_tree:intern_region_expr(expr_type, span)
  local region_type = std.as_read(expr_type)
  assert(flow_region_tree.is_region(region_type))
  if self:has_region_symbol(region_type) then
    return self:region_symbol(region_type)
  end

  local symbol = terralib.newsymbol(region_type)
  self.region_symbols[region_type] = symbol
  self.region_var_types[region_type] = expr_type
  self.region_spans[region_type] = span
  self.region_is_point[region_type] = false
  if std.is_region(region_type) then
    local partition = std.partition(std.disjoint, symbol)
    self.region_point_partitions[region_type] = partition
    std.add_constraint(self, partition, region_type, "<=", false)
    self:intern_region_expr(partition, span)
  end
end

function region_tree:intern_region_point_expr(parent, index, span)
  assert(std.is_region(parent) and not self:is_point(parent))
  local partition = self:point_partition(parent)
  local subregion
  if index then
    assert(terralib.issymbol(index.value))
    subregion = partition:subregion_constant(index.value)
  else
    subregion = partition:subregion_dynamic()
  end

  if self:has_region_symbol(subregion) then
    return subregion
  end

  local symbol = terralib.newsymbol(subregion)
  self.region_symbols[subregion] = symbol
  self.region_var_types[subregion] = subregion
  self.region_spans[subregion] = span
  self.region_is_point[subregion] = true
  std.add_constraint(self, subregion, partition, "<=", false)
  self:attach_region_index(subregion, index)
  return subregion
end

function region_tree:attach_region_index(region_type, index)
  assert(flow_region_tree.is_region(region_type))
  assert(self:region_symbol(region_type))
  self.region_indices[region_type] = index
end

function region_tree:ensure_variable(expr_type, symbol)
  local region_type = std.as_read(expr_type)
  if not flow_region_tree.is_region(region_type) then
    assert(symbol and rawget(self.interned_scalars, symbol))
    region_type = self.interned_scalars[symbol]
  end
  assert(self:has_region_symbol(region_type))
  return region_type
end

local function search_constraint_paths(constraints, region_type, path, visited,
                                       predicate)
  assert(not rawget(visited, region_type))
  visited[region_type] = true

  path:insert(region_type)
  if rawget(constraints, "<=") and rawget(constraints["<="], region_type) then
    for parent, _ in pairs(constraints["<="][region_type]) do
      local result = search_constraint_paths(
        constraints, parent, path, visited, predicate)
      if result then
        return result
      end
    end
  else
    if predicate(path) then
      return path
    end
  end
  path:remove()
end

function region_tree:aliased(region_type)
  assert(flow_region_tree.is_region(region_type))
  if std.is_region(region_type) then
    return true
  elseif std.is_partition(region_type) then
    return not region_type:is_disjoint()
  else
    assert(false)
  end
end

function region_tree:can_alias(region_type, other_region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(flow_region_tree.is_region(other_region_type))
  return not std.check_constraint(
    self, { lhs = region_type, rhs = other_region_type, op = "*" })
end

function region_tree:ancestors(region_type)
  assert(flow_region_tree.is_region(region_type))
  if rawget(self.cache_ancestors, region_type) then
    return self.cache_ancestors[region_type]:map(function(x) return x end)
  end

  local path = search_constraint_paths(
    self.constraints, region_type, terralib.newlist(), {},
    function() return true end)
  self.cache_ancestors[region_type] = path
  return path
end

function region_tree:lowest_common_ancestor(region_type, other_region_type)
  assert(flow_region_tree.is_region(region_type))
  assert(flow_region_tree.is_region(other_region_type))
  return std.search_constraint_predicate(
    self, region_type, {},
    function(cx, ancestor, x)
      if std.check_constraint(
        self, { lhs = other_region_type, rhs = ancestor, op = "<="})
      then
        return ancestor
      end
    end)
end

function region_tree:parent(region_type)
  assert(flow_region_tree.is_region(region_type))
  if rawget(self.constraints, "<=") and
    rawget(self.constraints["<="], region_type)
  then
    for parent, _ in pairs(self.constraints["<="][region_type]) do
      return parent
    end
  end
end

function region_tree:children(region_type)
  assert(flow_region_tree.is_region(region_type))
  if rawget(self.cache_children, region_type) then
    return self.cache_children[region_type]:map(function(x) return x end)
  end

  local result = terralib.newlist()
  if rawget(self.constraints, "<=") then
    for other, parents in pairs(self.constraints["<="]) do
      for parent, _ in pairs(parents) do
        if parent == region_type then
          result:insert(other)
          break
        end
      end
    end
  end
  self.cache_children[region_type] = result
  return result
end

function region_tree:siblings(region_type)
  assert(flow_region_tree.is_region(region_type))
  if rawget(self.cache_siblings, region_type) then
    return self.cache_siblings[region_type]:map(function(x) return x end)
  end

  local siblings = terralib.newlist()
  for other, _ in pairs(self.region_universe) do
    local is_subregion = std.check_constraint(
      self, { lhs = region_type, rhs = other, op = "<=" })
    local is_superregion = std.check_constraint(
      self, { lhs = other, rhs = region_type, op = "<=" })
    local is_disjoint = std.check_constraint(
      self, { lhs = region_type, rhs = other, op = "*" })
    if other ~= region_type and not (is_subregion or is_superregion or is_disjoint) then
      siblings:insert(other)
    end
  end
  self.cache_siblings[region_type] = siblings
  return siblings
end

return flow_region_tree
