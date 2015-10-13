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

-- Conversion from AST to Dataflow IR

local ast = require("regent/ast")
local flow = require("regent/flow")
local flow_region_tree = require("regent/flow_region_tree")
local std = require("regent/std")

local context = setmetatable({}, { __index = function(t, k) error("context has no field " .. tostring(k), 2) end})
context.__index = context

local region_tree_state

-- Context

function context:new_local_scope()
  local cx = {
    constraints = self.constraints,
    graph = flow.empty_graph(self.tree),
    epoch = terralib.newlist(),
    next_epoch = terralib.newlist(),
    next_epoch_opaque = false,
    tree = self.tree,
    state = region_tree_state.new(self.tree),
  }
  return setmetatable(cx, context)
end

function context:new_task_scope(constraints, region_universe)
  local tree = flow_region_tree.new_region_tree(constraints, region_universe)
  local cx = {
    constraints = constraints,
    graph = flow.empty_graph(tree),
    epoch = terralib.newlist(),
    next_epoch = terralib.newlist(),
    next_epoch_opaque = false,
    tree = tree,
    state = region_tree_state.new(tree),
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

-- Graph Construction

local function as_ast(cx, nid)
  return cx.graph:node_label(nid).value
end

local function sequence_depend(cx, nid)
  local label = cx.graph:node_label(nid)
  local opaque = flow.is_opaque_node(label)
  if opaque and not cx.next_epoch_opaque then
    if #cx.next_epoch > 0 then
      cx.epoch = cx.next_epoch
      cx.next_epoch = terralib.newlist()
    end
    cx.next_epoch_opaque = true
  end
  for _, epoch_nid in ipairs(cx.epoch) do
    if not cx.graph:reachable(epoch_nid, nid) then
      cx.graph:add_edge(
        flow.edge.HappensBefore {},
        epoch_nid, cx.graph:node_sync_port(epoch_nid),
        nid, cx.graph:node_sync_port(nid))
    end
  end
  return nid
end

local function sequence_advance(cx, nid)
  cx.next_epoch:insert(nid)
  if cx.next_epoch_opaque then
    cx.epoch = cx.next_epoch
    cx.next_epoch = terralib.newlist()
    cx.next_epoch_opaque = false
  end
  return nid
end

local function add_node(cx, label)
  return cx.graph:add_node(label)
end

local function add_input_edge(cx, from_nid, to_nid, to_port, privilege)
  assert(to_port > 0)
  local label
  if privilege == "none" then
    label = flow.edge.None {}
  elseif privilege == "reads" or privilege == "reads_writes" then
    label = flow.edge.Read {}
  else
    assert(false)
  end
  cx.graph:add_edge(
    label,
    from_nid, cx.graph:node_result_port(from_nid),
    to_nid, to_port)
end

local function add_output_edge(cx, from_nid, from_port, to_nid, privilege)
  assert(from_port > 0)
  local label
  if privilege == "reads_writes" then
    label = flow.edge.Write {}
  else
    assert(false)
  end
  cx.graph:add_edge(label, from_nid, from_port, to_nid, 0)
end

local function add_name_edge(cx, from_nid, to_nid)
  cx.graph:add_edge(
    flow.edge.Name {},
    from_nid, cx.graph:node_result_port(from_nid),
    to_nid, 0)
end

local function add_result(cx, from_nid, expr_type, span)
  if expr_type == terralib.types.unit then
    return from_nid
  end

  local label = ast.typed.ExprID {
    value = terralib.newsymbol(expr_type),
    expr_type = expr_type,
    span = span,
  }
  local result_nid = cx.graph:add_node(
    flow.node.Scalar {
      value = label,
      fresh = true,
  })
  local edge_label
  if flow_region_tree.is_region(expr_type) then
    edge_label = flow.edge.Name {}
  else
    edge_label = flow.edge.Write {}
  end
  cx.graph:add_edge(
    edge_label,
    from_nid, cx.graph:node_result_port(from_nid),
    result_nid, 0)
  return result_nid
end

-- Region Tree State

local region_state = setmetatable({}, { __index = function(t, k) error("region state has no field " .. tostring(k), 2) end})
region_state.__index = region_state

local modes = setmetatable(
  {
    closed = "closed",
    read = "read",
    write = "write",
  }, { __index = function(t, k) error("no such mode " .. tostring(k), 2) end})

local function is_mode(x)
  return rawget(modes, x)
end

region_tree_state = setmetatable({}, { __index = function(t, k) error("region tree state has no field " .. tostring(k), 2) end})
region_tree_state.__index = region_tree_state

function region_tree_state.new(tree)
  return setmetatable(
    {
      tree = tree,
      region_tree_state = {},
    }, region_tree_state)
end

function region_tree_state:ensure(region_type)
  assert(flow_region_tree.is_region(region_type))
  if not rawget(self.region_tree_state, region_type) then
    self.region_tree_state[region_type] = region_state.new()
  end
end

function region_tree_state:mode(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].mode
end

function region_tree_state:set_mode(region_type, mode)
  assert(rawget(self.region_tree_state, region_type))
  assert(is_mode(mode))
  self.region_tree_state[region_type].mode = mode
end

function region_tree_state:current(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].current
end

function region_tree_state:set_current(region_type, nid)
  assert(rawget(self.region_tree_state, region_type))
  self.region_tree_state[region_type].current = nid
end

function region_tree_state:open(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].open
end

function region_tree_state:set_open(region_type, nid)
  assert(rawget(self.region_tree_state, region_type))
  self.region_tree_state[region_type].open = nid
end

function region_tree_state:dirty(region_type)
  assert(rawget(self.region_tree_state, region_type))
  return self.region_tree_state[region_type].dirty
end

function region_tree_state:set_dirty(region_type, dirty)
  assert(rawget(self.region_tree_state, region_type))
  self.region_tree_state[region_type].dirty = dirty
end

function region_tree_state:clear(region_type)
  assert(rawget(self.region_tree_state, region_type))
  self.region_tree_state[region_type] = region_state.new()
end

function region_tree_state:dirty_children(region_type)
  local result = terralib.newlist()
  for _, child in ipairs(self.tree:children(region_type)) do
    if self:dirty(child) then
      result:insert(child)
    end
  end
  return result
end

function region_tree_state:open_siblings(region_type)
  local result = terralib.newlist()
  if not self.tree:aliased(region_type) then return result end
  for _, sibling in ipairs(self.tree:siblings(region_type)) do
    self:ensure(sibling)
    if self:mode(sibling) ~= modes.closed then
      result:insert(sibling)
    end
  end
  return result
end

function region_tree_state:dirty_siblings(region_type)
  local result = terralib.newlist()
  if not self.tree:aliased(region_type) then return result end
  local result = terralib.newlist()
  for _, sibling in ipairs(self.tree:siblings(region_type)) do
    self:ensure(sibling)
    if self:dirty(sibling) then
      result:insert(sibling)
    end
  end
  return result
end

function region_state.new()
  return setmetatable({
      mode = modes.closed,
      current = flow.null(),
      open = flow.null(),
      dirty = false,
  }, region_state)
end

-- Region Identity Analysis

local analyze_regions = {}

function analyze_regions.vars(cx)
  return function(node)
    if node:is(ast.typed.StatVar) then
      for i, var_symbol in ipairs(node.symbols) do
        local var_type = std.rawref(&node.types[i])
        cx.tree:intern_variable(var_type, var_symbol, node.span)
      end
    elseif node:is(ast.typed.StatVarUnpack) then
      for i, var_symbol in ipairs(node.symbols) do
        local var_type = std.rawref(&node.field_types[i])
        cx.tree:intern_variable(var_type, var_symbol, node.span)
      end
    elseif node:is(ast.typed.StatForNum) or node:is(ast.typed.StatForList) then
      local var_symbol = node.symbol
      local var_type = node.symbol.type
      cx.tree:intern_variable(var_type, var_symbol, node.span)
    elseif node:is(ast.typed.StatTask) then
      for i, param in ipairs(node.params) do
        local param_type = std.rawref(&param.param_type)
        cx.tree:intern_variable(param_type, param.symbol, param.span)
      end
    end
  end
end

function analyze_regions.expr(cx)
  return function(node)
    local expr_type = std.as_read(node.expr_type)
    if flow_region_tree.is_region(expr_type) then
      cx.tree:intern_region_expr(node.expr_type, node.span)
      if node:is(ast.typed.ExprIndexAccess) then
        cx.tree:attach_region_index(expr_type, node.index)
      end
    elseif std.is_cross_product(expr_type) then
      -- FIXME: This is kind of a hack. Cross products aren't really
      -- first class, but this ought not be necessary.
      cx.tree:intern_region_expr(expr_type:partition(), node.span)
      if node:is(ast.typed.ExprIndexAccess) then
        cx.tree:attach_region_index(expr_type:partition(), node.index)
      end
    end

    if node:is(ast.typed.ExprDeref) then
      local value_type = std.as_read(node.value.expr_type)
      if std.is_bounded_type(value_type) then
        local bounds = value_type:bounds()
        for _, parent in ipairs(bounds) do
          local index
          if node.value:is(ast.typed.ExprID) and
            not std.is_rawref(node.value.expr_type)
          then
            index = node.value
          end
          cx.tree:intern_region_point_expr(parent, index, node.span)
        end
      end
    end
  end
end

function analyze_regions.stat_task(cx, node)
  ast.traverse_node_postorder(analyze_regions.vars(cx), node)
  ast.traverse_expr_postorder(analyze_regions.expr(cx), node)
end

-- Region Tree Analysis

local function privilege_mode(privilege)
  if privilege == "none" then
    return false
  elseif privilege == "reads" then
    return modes.read
  else
    return modes.write
  end
end

local function get_region_label(cx, region_type)
  local symbol = cx.tree:region_symbol(region_type)
  local expr_type = cx.tree:region_var_type(region_type)
  local name = ast.typed.ExprID {
    value = cx.tree:region_symbol(region_type),
    expr_type = expr_type,
    span = cx.tree:region_span(region_type),
  }
  if std.is_region(std.as_read(expr_type)) then
    return flow.node.Region { value = name }
  elseif std.is_partition(std.as_read(expr_type)) then
    return flow.node.Partition { value = name }
  else
    return flow.node.Scalar { value = name, fresh = false }
  end
end

local transitions = setmetatable(
  {}, { __index = function(t, k) error("no such transition " .. tostring(k), 2) end})

function transitions.nothing(cx, path, index)
  return cx.state:current(path[index]), false
end

function transitions.create(cx, path, index)
  local current_nid = cx.state:current(path[index])
  if not flow.is_null(current_nid) then
    return current_nid, false
  end

  local next_nid = cx.graph:add_node(get_region_label(cx, path[index]))
  local parent_index = index + 1
  if parent_index <= #path then
    local open_nid = cx.state:open(path[parent_index])
    if flow.is_valid_node(open_nid) then
      add_output_edge(cx, open_nid, 1, next_nid, "reads_writes")
    end
  end
  cx.state:set_current(path[index], next_nid)
  return next_nid, true
end

function transitions.open(cx, path, index)
  local current_nid, fresh = transitions.create(cx, path, index)
  assert(flow.is_null(cx.state:open(path[index])))
  local open_nid = cx.graph:add_node(flow.node.Open {})
  add_input_edge(cx, current_nid, open_nid, 1, "reads")
  cx.state:set_open(path[index], open_nid)

  -- Add sequence dependencies here to avoid redundant edges already
  -- encoded by true data dependencies.
  if fresh then sequence_depend(cx, current_nid) end
end

function transitions.close(cx, path, index)
  -- Close all children.
  for _, child in ipairs(cx.tree:children(path[index])) do
    cx.state:ensure(child)
    if cx.state:mode(child) ~= modes.closed then
      local child_path = std.newtuple(child) .. path:slice(index, #path)
      transitions.close(cx, child_path, 1, cx.graph:node_label(cx.state:current(child)).value)
    end
  end

  -- Create and link the close node.
  local close_nid = cx.graph:add_node(flow.node.Close {})
  add_input_edge(cx, cx.state:current(path[index]), close_nid, 1, "reads")
  local port = 2
  for _, child in ipairs(cx.tree:children(path[index])) do
    local child_nid = cx.state:current(child)
    if flow.is_valid_node(child_nid) then
      add_input_edge(cx, child_nid, close_nid, port, "reads")
      port = port + 1
    end
  end

  -- Create and link the next node.
  local next_nid = cx.graph:add_node(get_region_label(cx, path[index]))
  add_output_edge(cx, close_nid, 1, next_nid, "reads_writes")

  -- Clear child state.
  for _, child in ipairs(cx.tree:children(path[index])) do
    cx.state:clear(child)
  end

  -- Set node state.
  cx.state:set_mode(path[index], modes.closed)
  cx.state:set_current(path[index], next_nid)
  cx.state:set_open(path[index], flow.null())
  cx.state:set_dirty(path[index], true)

  return next_nid, true
end

function transitions.close_conflicting_children(cx, path, index, _)
  for _, child in ipairs(cx.tree:children(path[index])) do
    cx.state:ensure(child)
    if cx.state:mode(child) ~= modes.closed then
      local child_path = std.newtuple(child) .. path:slice(index, #path)
      transitions.close(cx, child_path, 1, cx.graph:node_label(cx.state:current(child)).value)
    end
  end
end

function transitions.close_and_reopen(cx, path, index, _)
  transitions.close(cx, path, index, nil)
  transitions.open(cx, path, index, nil)
end

local function select_transition(cx, path, index, desired_mode)
  local current_mode = cx.state:mode(path[index])
  if index == 1 then -- Leaf
    if current_mode == modes.closed then
      return modes.closed, transitions.create
    elseif current_mode == modes.read and desired_mode == modes.read then
      return modes.read, transitions.nothing
    elseif current_mode == modes.read and desired_mode == modes.write then
      return modes.write, transitions.close
    elseif current_mode == modes.write then
      return modes.closed, transitions.close
    else
      assert(false)
    end
  else -- Inner
    local child_index = index - 1
    if current_mode == modes.closed then
      return desired_mode, transitions.open
    elseif current_mode == modes.read and desired_mode == modes.read then
      return modes.read, transitions.nothing
    elseif current_mode == modes.read and desired_mode == modes.write then
      if #cx.state:open_siblings(path[child_index]) > 0 then
        -- FIXME: This doesn't actually work. When writing to an
        -- aliased partition that was previously read, we will miss a
        -- WAR dependence if we don't fully close the tree. For now,
        -- be conservative and use close_and_reopen instead.
        return modes.write, transitions.close_and_reopen
        -- return modes.write, transitions.close_conflicting_children
      else
        return modes.write, transitions.nothing
      end
    elseif current_mode == modes.write then
      if #cx.state:dirty_siblings(path[child_index]) > 0 then
        return modes.write, transitions.close_and_reopen
      else
        return modes.write, transitions.nothing
      end
    else
      assert(false)
    end
  end
end

local function open_region_tree_node(cx, path, index, mode)
  assert(index >= 1)
  cx.state:ensure(path[index])
  local next_mode, transition = select_transition(cx, path, index, mode)
  local next_nid, fresh = transition(cx, path, index)
  cx.state:set_mode(path[index], next_mode)
  if index >= 2 then
    return open_region_tree_node(cx, path, index-1, mode)
  end
  return next_nid, fresh
end

local function open_region_tree(cx, expr_type, symbol, privilege)
  local region_type = cx.tree:ensure_variable(expr_type, symbol)
  assert(flow_region_tree.is_region(region_type))

  local path = std.newtuple(unpack(cx.tree:ancestors(region_type)))
  local desired_mode = privilege_mode(privilege)
  if not desired_mode then
    -- Special case for "none" privilege: just create the node and
    -- exit without linking it up to anything.
    local next_nid = cx.graph:add_node(get_region_label(cx, region_type))
    sequence_depend(cx, next_nid)
    return next_nid
  end
  local current_nid, fresh = open_region_tree_node(
    cx, path, #path, desired_mode)
  if fresh then sequence_depend(cx, current_nid) end
  local next_nid
  if desired_mode == modes.write then
    next_nid = add_node(cx, cx.graph:node_label(current_nid))
    cx.state:ensure(region_type)
    cx.state:set_current(region_type, next_nid)
    cx.state:set_open(region_type, flow.null())
    cx.state:set_dirty(region_type, true)
  end
  assert(flow.is_valid_node(current_nid))
  assert(next_nid == nil or flow.is_valid_node(next_nid))
  return current_nid, next_nid
end

local function preopen_region_tree(cx, region_type, privilege)
  assert(flow_region_tree.is_region(region_type))

  local path = std.newtuple(unpack(cx.tree:ancestors(region_type)))
  local desired_mode = privilege_mode(privilege)
  assert(desired_mode)
  for index = #path, 2, -1 do
    cx.state:ensure(path[index])
    cx.state:set_mode(path[index], desired_mode)
  end
end

-- Summarization of Privileges

local region_privileges = {}
region_privileges.__index = region_privileges

function region_privileges:__tostring()
  local result = "region_privileges(\n"
  for region_type, privilege in pairs(self) do
    result = result .. "  " .. tostring(region_type) .. " = " .. tostring(privilege) .. ",\n"
  end
  result = result .. ")"
  return result
end

local function uses(cx, region_type, privilege)
  return setmetatable({ [region_type] = privilege }, region_privileges)
end

local function privilege_meet(...)
  local usage = {}
  for _, a in pairs({...}) do
    if a then
      for region_type, privilege in pairs(a) do
        usage[region_type] = std.meet_privilege(usage[region_type], privilege)
      end
    end
  end
  return setmetatable(usage, region_privileges)
end

local function strip_indexing(cx, region_type)
  local path = std.newtuple(unpack(cx.tree:ancestors(region_type)))
  local last_index = 0
  for index = 1, #path do
    if cx.tree:has_region_index(path[index]) and
      not cx.tree:region_index(path[index]):is(ast.typed.ExprConstant)
    then
      last_index = index
    end
  end
  assert(last_index < #path)
  return path[last_index + 1]
end

local function privilege_summary(cx, usage, strip)
  local summary = {}
  if not usage then return summary end
  for region_type, privilege in pairs(usage) do
    if privilege ~= "none" then
      local region = region_type
      if strip then
        region = strip_indexing(cx, region_type)
      end

      local recorded = false
      local next_summary = {}
      for other, other_privilege in pairs(summary) do
        if other_privilege ~= "none" then
          local ancestor = cx.tree:lowest_common_ancestor(region, other)
          if ancestor then
            assert(not rawget(next_summary, ancestor))
            next_summary[ancestor] = std.meet_privilege(privilege, other_privilege)
            recorded = true
          else
            assert(not rawget(next_summary, other))
            next_summary[other] = other_privilege
          end
        end
      end
      if not recorded then
        next_summary[region] = privilege
      end
      summary = next_summary
    end
  end
  return setmetatable(summary, region_privileges)
end

-- FIXME: This is wrong. Need to pass privileges in, because
-- e.g. deref can't tell how it's used without the caller's
-- privileges.
local analyze_privileges = {}

function analyze_privileges.expr_id(cx, node, privilege)
  local expr_type = std.as_read(node.expr_type)
  if flow_region_tree.is_region(expr_type) then
    return uses(cx, expr_type, privilege)
  end
end

function analyze_privileges.expr_field_access(cx, node, privilege)
  return analyze_privileges.expr(cx, node.value, privilege)
end

function analyze_privileges.expr_index_access(cx, node, privilege)
  local expr_type = std.as_read(node.expr_type)
  local value_privilege = "reads"
  local usage
  if flow_region_tree.is_region(expr_type) then
    value_privilege = "none"
    usage = uses(cx, expr_type, privilege)
  end
  return privilege_meet(
    analyze_privileges.expr(cx, node.value, value_privilege),
    analyze_privileges.expr(cx, node.index, "reads"),
    usage)
end

function analyze_privileges.expr_method_call(cx, node, privilege)
  local usage = analyze_privileges.expr(cx, node.value, "reads")
  for _, arg in ipairs(node.args) do
    usage = privilege_meet(usage, analyze_privileges.expr(cx, arg, "reads"))
  end
  return usage
end

function analyze_privileges.expr_call(cx, node, privilege)
  local usage = analyze_privileges.expr(cx, node.fn, "reads")
  local is_task = std.is_task(node.fn.value)
  for i, arg in ipairs(node.args) do
    local arg_type = std.as_read(arg.expr_type)
    local param_type = node.fn.expr_type.parameters[i]
    if is_task and std.is_region(param_type) then
      local privileges, privilege_field_paths, _ =
        std.find_task_privileges(param_type, node.fn.value:getprivileges())
      -- Field insensitive for now.
      local privilege = std.reduce(std.meet_privilege, privileges, "none")
      if std.is_reduction_op(privilege) then
        privilege = "reads_writes"
      end

      usage = privilege_meet(
        usage, analyze_privileges.expr(cx, arg, privilege))
    end
  end
  return usage
end

function analyze_privileges.expr_cast(cx, node, privilege)
  return privilege_meet(analyze_privileges.expr(cx, node.fn, "reads"),
                        analyze_privileges.expr(cx, node.arg, "reads"))
end

function analyze_privileges.expr_ctor(cx, node, privilege)
  local usage = nil
  for _, field in ipairs(node.fields) do
    usage = privilege_meet(
      usage, analyze_privileges.expr(cx, field.value, "reads"))
  end
  return usage
end

function analyze_privileges.expr_raw_physical(cx, node, privilege)
  assert(false) -- This case needs special handling.
  return privilege_meet(
    analyze_privileges.expr(cx, node.region, "reads_writes"))
end

function analyze_privileges.expr_raw_fields(cx, node, privilege)
  return analyze_privileges.expr(cx, node.region, "none")
end

function analyze_privileges.expr_raw_value(cx, node, privilege)
  return analyze_privileges.expr(cx, node.value, "none")
end

function analyze_privileges.expr_isnull(cx, node, privilege)
  return analyze_privileges.expr(cx, node.pointer, "reads")
end

function analyze_privileges.expr_dynamic_cast(cx, node, privilege)
  return analyze_privileges.expr(cx, node.value, "reads")
end

function analyze_privileges.expr_static_cast(cx, node, privilege)
  return analyze_privileges.expr(cx, node.value, "reads")
end

function analyze_privileges.expr_ispace(cx, node, privilege)
  return privilege_meet(
    analyze_privileges.expr(cx, node.extent, "reads"),
    node.start and analyze_privileges.expr(cx, node.start, "reads"))
end

function analyze_privileges.expr_region(cx, node, privilege)
  return analyze_privileges.expr(cx, node.ispace, "reads")
end

function analyze_privileges.expr_partition(cx, node, privilege)
  return analyze_privileges.expr(cx, node.coloring, "reads")
end

function analyze_privileges.expr_cross_product(cx, node, privilege)
  return std.reduce(
    privilege_meet,
    node.args:map(
      function(arg) return analyze_privileges.expr(cx, arg, "reads") end))
end

function analyze_privileges.expr_unary(cx, node, privilege)
  return analyze_privileges.expr(cx, node.rhs, "reads")
end

function analyze_privileges.expr_binary(cx, node, privilege)
  return privilege_meet(analyze_privileges.expr(cx, node.lhs, "reads"),
                        analyze_privileges.expr(cx, node.rhs, "reads"))
end

function analyze_privileges.expr_deref(cx, node, privilege)
  local value_type = std.as_read(node.value.expr_type)
  local usage
  if std.is_bounded_type(value_type) then
    local bounds = value_type:bounds()
    for _, parent in ipairs(bounds) do
      local index
      if node.value:is(ast.typed.ExprID) and
        not std.is_rawref(node.value.expr_type)
      then
        index = node.value
      end
      local subregion = cx.tree:intern_region_point_expr(parent, index, node.span)
      usage = privilege_meet(usage, uses(cx, subregion, privilege))
    end
  end
  return privilege_meet(
    analyze_privileges.expr(cx, node.value, "reads"),
    usage)
end

function analyze_privileges.expr(cx, node, privilege)
  if node:is(ast.typed.ExprID) then
    return analyze_privileges.expr_id(cx, node, privilege)

  elseif node:is(ast.typed.ExprConstant) then
    return nil

  elseif node:is(ast.typed.ExprFunction) then
    return nil

  elseif node:is(ast.typed.ExprFieldAccess) then
    return analyze_privileges.expr_field_access(cx, node, privilege)

  elseif node:is(ast.typed.ExprIndexAccess) then
    return analyze_privileges.expr_index_access(cx, node, privilege)

  elseif node:is(ast.typed.ExprMethodCall) then
    return analyze_privileges.expr_method_call(cx, node, privilege)

  elseif node:is(ast.typed.ExprCall) then
    return analyze_privileges.expr_call(cx, node, privilege)

  elseif node:is(ast.typed.ExprCast) then
    return analyze_privileges.expr_cast(cx, node, privilege)

  elseif node:is(ast.typed.ExprCtor) then
    return analyze_privileges.expr_ctor(cx, node, privilege)

  elseif node:is(ast.typed.ExprRawContext) then
    return nil

  elseif node:is(ast.typed.ExprRawFields) then
    return analyze_privileges.expr_raw_fields(cx, node, privilege)

  elseif node:is(ast.typed.ExprRawPhysical) then
    return analyze_privileges.expr_raw_physical(cx, node, privilege)

  elseif node:is(ast.typed.ExprRawRuntime) then
    return nil

  elseif node:is(ast.typed.ExprRawValue) then
    return analyze_privileges.expr_raw_value(cx, node, privilege)

  elseif node:is(ast.typed.ExprIsnull) then
    return analyze_privileges.expr_isnull(cx, node, privilege)

  elseif node:is(ast.typed.ExprNew) then
    return nil

  elseif node:is(ast.typed.ExprNull) then
    return nil

  elseif node:is(ast.typed.ExprDynamicCast) then
    return analyze_privileges.expr_dynamic_cast(cx, node, privilege)

  elseif node:is(ast.typed.ExprStaticCast) then
    return analyze_privileges.expr_static_cast(cx, node, privilege)

  elseif node:is(ast.typed.ExprIspace) then
    return analyze_privileges.expr_ispace(cx, node, privilege)

  elseif node:is(ast.typed.ExprRegion) then
    return analyze_privileges.expr_region(cx, node, privilege)

  elseif node:is(ast.typed.ExprPartition) then
    return analyze_privileges.expr_partition(cx, node, privilege)

  elseif node:is(ast.typed.ExprCrossProduct) then
    return analyze_privileges.expr_cross_product(cx, node, privilege)

  elseif node:is(ast.typed.ExprUnary) then
    return analyze_privileges.expr_unary(cx, node, privilege)

  elseif node:is(ast.typed.ExprBinary) then
    return analyze_privileges.expr_binary(cx, node, privilege)

  elseif node:is(ast.typed.ExprDeref) then
    return analyze_privileges.expr_deref(cx, node, privilege)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function analyze_privileges.block(cx, node)
  return std.reduce(
    privilege_meet,
    node.stats:map(function(stat) return analyze_privileges.stat(cx, stat) end))
end

function analyze_privileges.stat_if(cx, node)
  return
    privilege_meet(
      analyze_privileges.expr(cx, node.cond, "reads"),
      analyze_privileges.block(cx, node.then_block),
      std.reduce(
        privilege_meet,
        node.elseif_blocks:map(
          function(block) return analyze_privileges.stat_elseif(cx, block) end)),
      analyze_privileges.block(cx, node.else_block))
end

function analyze_privileges.stat_elseif(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.cond, "reads"),
    analyze_privileges.block(cx, node.block))
end

function analyze_privileges.stat_while(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.cond, "reads"),
    analyze_privileges.block(cx, node.block))
end

function analyze_privileges.stat_for_num(cx, node)
  return
    std.reduce(
      privilege_meet,
      node.values:map(
        function(value) return analyze_privileges.expr(cx, value, "reads") end),
      analyze_privileges.block(cx, node.block))
end

function analyze_privileges.stat_for_list(cx, node)
  return privilege_meet(
    analyze_privileges.expr(cx, node.value, "reads"),
    analyze_privileges.block(cx, node.block))
end

function analyze_privileges.stat_repeat(cx, node)
  return privilege_meet(
    analyze_privileges.block(cx, node.block),
    analyze_privileges.expr(cx, node.until_cond, "reads"))
end

function analyze_privileges.stat_block(cx, node)
  return analyze_privileges.block(cx, node.block)
end

function analyze_privileges.stat_var(cx, node)
  return std.reduce(
    privilege_meet,
    node.values:map(
      function(value) return analyze_privileges.expr(cx, value, "reads") end))
end

function analyze_privileges.stat_var_unpack(cx, node)
  return analyze_privileges.expr(cx, node.value, "reads")
end

function analyze_privileges.stat_return(cx, node) 
  if node.value then
    return analyze_privileges.expr(cx, node.value, "reads")
  else
    return nil
  end
end

function analyze_privileges.stat_assignment(cx, node)
  return
    std.reduce(
      privilege_meet,
      node.lhs:map(
        function(lh)
          return analyze_privileges.expr(cx, lh, "reads_writes")
      end),
      std.reduce(
        privilege_meet,
        node.rhs:map(
          function(rh) return analyze_privileges.expr(cx, rh, "reads") end)))
end

function analyze_privileges.stat_reduce(cx, node)
  return
    std.reduce(
      privilege_meet,
      node.lhs:map(
        function(lh) return analyze_privileges.expr(cx, lh, "reads_writes") end),
      std.reduce(
        privilege_meet,
        node.rhs:map(
          function(rh) return analyze_privileges.expr(cx, rh, "reads") end)))
end

function analyze_privileges.stat_expr(cx, node)
  return analyze_privileges.expr(cx, node.expr, "reads")
end

function analyze_privileges.stat(cx, node)
  if node:is(ast.typed.StatIf) then
    return analyze_privileges.stat_if(cx, node)

  elseif node:is(ast.typed.StatWhile) then
    return analyze_privileges.stat_while(cx, node)

  elseif node:is(ast.typed.StatForNum) then
    return analyze_privileges.stat_for_num(cx, node)

  elseif node:is(ast.typed.StatForList) then
    return analyze_privileges.stat_for_list(cx, node)

  elseif node:is(ast.typed.StatRepeat) then
    return analyze_privileges.stat_repeat(cx, node)

  elseif node:is(ast.typed.StatBlock) then
    return analyze_privileges.stat_block(cx, node)

  elseif node:is(ast.typed.StatVar) then
    return analyze_privileges.stat_var(cx, node)

  elseif node:is(ast.typed.StatVarUnpack) then
    return analyze_privileges.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.StatReturn) then
    return analyze_privileges.stat_return(cx, node)

  elseif node:is(ast.typed.StatBreak) then
    return nil

  elseif node:is(ast.typed.StatAssignment) then
    return analyze_privileges.stat_assignment(cx, node)

  elseif node:is(ast.typed.StatReduce) then
    return analyze_privileges.stat_reduce(cx, node)

  elseif node:is(ast.typed.StatExpr) then
    return analyze_privileges.stat_expr(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

-- AST -> Dataflow IR

local flow_from_ast = {}

local function as_stat(cx, label)
  local compute_nid = add_node(cx, label)
  sequence_depend(cx, compute_nid)
  return sequence_advance(cx, compute_nid)
end

local function as_opaque_stat(cx, node)
  local label = flow.node.Opaque { action = node }
  return as_stat(cx, label)
end

local function as_fornum_stat(cx, symbol, block, parallel, args, span)
  local label = flow.node.ForNum {
    symbol = symbol,
    block = block,
    parallel = parallel,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  for i, arg in pairs(args) do
    local privilege, input_nid, output_nid = unpack(arg)
    add_input_edge(cx, input_nid, compute_nid, i, privilege)
    if output_nid then
      add_output_edge(cx, compute_nid, i, output_nid, privilege)
    end
  end
  sequence_depend(cx, compute_nid)
  return sequence_advance(cx, compute_nid)
end

local function as_forlist_stat(cx, symbol, block, vectorize, args, span)
  local label = flow.node.ForList {
    symbol = symbol,
    block = block,
    vectorize = vectorize,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  for i, arg in pairs(args) do
    local privilege, input_nid, output_nid = unpack(arg)
    add_input_edge(cx, input_nid, compute_nid, i, privilege)
    if output_nid then
      add_output_edge(cx, compute_nid, i, output_nid, privilege)
    end
  end
  sequence_depend(cx, compute_nid)
  return sequence_advance(cx, compute_nid)
end

local function as_reduce_stat(cx, op, args, span)
  local label = flow.node.Reduce {
    op = op,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  for i, arg in pairs(args) do
    local input_nid, output_nid = unpack(arg)
    add_input_edge(cx, input_nid, compute_nid, i, "reads")
    if output_nid then
      add_output_edge(cx, compute_nid, i, output_nid, "reads_writes")
    end
  end
  sequence_depend(cx, compute_nid)
  return compute_nid
end

local function as_opaque_expr(cx, node, input_nids)
  local label = flow.node.Opaque { action = node }
  local compute_nid = add_node(cx, label)
  local result_nid = add_result(
    cx, compute_nid, std.as_read(node.expr_type), node.span)
  for i, input_nid in ipairs(input_nids) do
    add_input_edge(cx, input_nid, compute_nid, i, "reads")
  end
  sequence_depend(cx, compute_nid)
  return result_nid
end

local function as_call_expr(cx, args, opaque, expr_type, span)
  local label = flow.node.Task {
    opaque = opaque,
    expr_type = expr_type,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  local result_nid = add_result(cx, compute_nid, expr_type, span)
  for i, arg in ipairs(args) do
    local privilege, input_nid, output_nid = unpack(arg)
    add_input_edge(cx, input_nid, compute_nid, i, privilege)
    if output_nid then
      add_output_edge(cx, compute_nid, i, output_nid, privilege)
    end
  end
  sequence_depend(cx, compute_nid)
  return result_nid
end

local function as_index_expr(cx, args, result_nid, expr_type, span)
  local label = flow.node.IndexAccess {
    expr_type = expr_type,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_name_edge(cx, compute_nid, result_nid)
  for i, input_pair in ipairs(args) do
    local privilege, input_nid = unpack(input_pair)
    add_input_edge(cx, input_nid, compute_nid, i, privilege)
  end
  sequence_depend(cx, compute_nid)
  return result_nid
end

local function as_deref_expr(cx, args, result_nid, expr_type, span)
  local label = flow.node.Deref {
    expr_type = expr_type,
    span = span,
  }
  local compute_nid = add_node(cx, label)
  add_name_edge(cx, compute_nid, result_nid)
  for i, input_pair in ipairs(args) do
    local privilege, input_nid = unpack(input_pair)
    add_input_edge(cx, input_nid, compute_nid, i, privilege)
  end
  sequence_depend(cx, compute_nid)
  return result_nid
end

function flow_from_ast.expr_id(cx, node, privilege)
  return open_region_tree(cx, node.expr_type, node.value, privilege)
end

function flow_from_ast.expr_constant(cx, node, privilege)
  return cx.graph:add_node(flow.node.Constant { value = node })
end

function flow_from_ast.expr_function(cx, node, privilege)
  return cx.graph:add_node(flow.node.Function { value = node })
end

function flow_from_ast.expr_field_access(cx, node, privilege)
  local value = flow_from_ast.expr(cx, node.value, "reads")
  return as_opaque_expr(
    cx,
    node { value = as_ast(cx, value) },
    terralib.newlist({value}))
end

function flow_from_ast.expr_index_access(cx, node, privilege)
  local expr_type = std.as_read(node.expr_type)
  local value_privilege = "reads"
  if flow_region_tree.is_region(expr_type) then
    value_privilege = "none"
  end
  local value = flow_from_ast.expr(cx, node.value, value_privilege)

  local index_privilege = "reads"
  local index = flow_from_ast.expr(cx, node.index, index_privilege)

  if flow_region_tree.is_region(expr_type) then
    local inputs = terralib.newlist({
        {value_privilege, value},
        {index_privilege, index},
    })
    local input_nid, output_nid = open_region_tree(
      cx, node.expr_type, nil, privilege)
    as_index_expr(
      cx, inputs, input_nid, expr_type, node.span)
    return input_nid, output_nid
  else
    return as_opaque_expr(
      cx,
      node {
        value = as_ast(cx, value),
        index = as_ast(cx, index),
      },
      terralib.newlist({value, index}))
  end
end

function flow_from_ast.expr_method_call(cx, node, privilege)
  local value = flow_from_ast.expr(cx, node.value, "reads")
  local args = node.args:map(function(arg) return flow_from_ast.expr(cx, arg, "reads") end)
  local inputs = terralib.newlist({value})
  inputs:insertall(args)
  return as_opaque_expr(
    cx,
    node {
      value = as_ast(cx, value),
      index = args:map(function(arg) return as_ast(cx, arg) end),
    },
    inputs)
end

function flow_from_ast.expr_call(cx, node, privilege)
  local fn = flow_from_ast.expr(cx, node.fn, "reads")
  local arg_privileges = terralib.newlist()
  local args = terralib.newlist()
  for i, arg in ipairs(node.args) do
    local param_type = node.fn.expr_type.parameters[i]
    local privilege
    if std.is_task(node.fn.value) and std.is_region(param_type) then
      local privileges, privilege_field_paths, _ =
        std.find_task_privileges(param_type, node.fn.value:getprivileges())
      -- Field insensitive for now.
      privilege = std.reduce(std.meet_privilege, privileges, "none")
      if std.is_reduction_op(privilege) then
        privilege = "reads_writes"
      end
    else
      privilege = "reads"
    end

    arg_privileges:insert(privilege)
    args:insert({privilege, flow_from_ast.expr(cx, arg, privilege)})
  end

  local inputs = terralib.newlist({{"reads", fn}})
  inputs:insertall(args)
  return as_call_expr(
    cx, inputs,
    not std.is_task(node.fn.value), std.as_read(node.expr_type), node.span)
end

function flow_from_ast.expr_cast(cx, node, privilege)
  local fn = flow_from_ast.expr(cx, node.fn, "reads")
  local arg = flow_from_ast.expr(cx, node.arg, "reads")
  return as_opaque_expr(
    cx,
    node {
      fn = as_ast(cx, fn),
      arg = as_ast(cx, arg),
    },
    terralib.newlist({fn, arg}))
end

function flow_from_ast.expr_ctor(cx, node, privilege)
  local values = node.fields:map(
    function(field) return flow_from_ast.expr(cx, field.value, "reads") end)
  local fields = std.zip(node.fields, values):map(
    function(pair)
      local field, value = unpack(pair)
      return field { value = as_ast(cx, value) }
    end)
  return as_opaque_expr(
    cx,
    node { fields = fields },
    values)
end

function flow_from_ast.expr_raw_context(cx, node, privilege)
  return as_opaque_expr(cx, node, terralib.newlist())
end

function flow_from_ast.expr_raw_fields(cx, node, privilege)
  local region = flow_from_ast.expr(cx, node.region, "reads")
  return as_opaque_expr(
    cx,
    node { region = as_ast(cx, region) },
    terralib.newlist({region}))
end

function flow_from_ast.expr_raw_physical(cx, node, privilege)
  local region = flow_from_ast.expr(cx, node.region, "reads")
  return as_opaque_expr(
    cx,
    node { region = as_ast(cx, region) },
    terralib.newlist({region}))
end

function flow_from_ast.expr_raw_runtime(cx, node, privilege)
  return as_opaque_expr(cx, node, terralib.newlist())
end

function flow_from_ast.expr_raw_value(cx, node, privilege)
  local value = flow_from_ast.expr(cx, node.value, "reads")
  return as_opaque_expr(
    cx,
    node { value = as_ast(cx, value) },
    terralib.newlist({value}))
end

function flow_from_ast.expr_isnull(cx, node, privilege)
  local pointer = flow_from_ast.expr(cx, node.pointer, "reads")
  return as_opaque_expr(
    cx,
    node { pointer = as_ast(cx, pointer) },
    terralib.newlist({pointer}))
end

function flow_from_ast.expr_new(cx, node, privilege)
  local region = flow_from_ast.expr(cx, node.region, "none")
  return as_opaque_expr(
    cx,
    node { region = as_ast(cx, region) },
    terralib.newlist({region}))
end

function flow_from_ast.expr_dynamic_cast(cx, node, privilege)
  local value = flow_from_ast.expr(cx, node.value, "reads")
  return as_opaque_expr(
    cx,
    node { value = as_ast(cx, value) },
    terralib.newlist({value}))
end

function flow_from_ast.expr_static_cast(cx, node, privilege)
  local value = flow_from_ast.expr(cx, node.value, "reads")
  return as_opaque_expr(
    cx,
    node { value = as_ast(cx, value) },
    terralib.newlist({value}))
end

function flow_from_ast.expr_unary(cx, node, privilege)
  local rhs = flow_from_ast.expr(cx, node.rhs, "reads")
  return as_opaque_expr(
    cx,
    node { rhs = as_ast(cx, rhs) },
    terralib.newlist({rhs}))
end

function flow_from_ast.expr_binary(cx, node, privilege)
  local lhs = flow_from_ast.expr(cx, node.lhs, "reads")
  local rhs = flow_from_ast.expr(cx, node.rhs, "reads")
  return as_opaque_expr(
    cx,
    node {
      lhs = as_ast(cx, lhs),
      rhs = as_ast(cx, rhs),
    },
    terralib.newlist({lhs, rhs}))
end

function flow_from_ast.expr_deref(cx, node, privilege)
  local value = flow_from_ast.expr(cx, node.value, "reads")
  local value_type = std.as_read(node.value.expr_type)
  if std.is_bounded_type(value_type) then
    local bounds = value_type:bounds()
    if #bounds == 1 and std.is_region(bounds[1]) then
      local parent = bounds[1]
      local index
      if node.value:is(ast.typed.ExprID) and
        not std.is_rawref(node.value.expr_type)
      then
        index = node.value
      end
      local subregion = cx.tree:intern_region_point_expr(parent, index, node.span)

      local inputs = terralib.newlist({
          {"reads", value},
      })
      local input_nid, output_nid = open_region_tree(cx, subregion, nil, privilege)
      as_deref_expr(
        cx, inputs, input_nid, node.expr_type, node.span)
      return input_nid, output_nid
    end
  end

  return as_opaque_expr(
    cx,
    node { value = as_ast(cx, value) },
    terralib.newlist({value}))
end

function flow_from_ast.expr(cx, node, privilege)
  if node:is(ast.typed.ExprID) then
    return flow_from_ast.expr_id(cx, node, privilege)

  elseif node:is(ast.typed.ExprConstant) then
    return flow_from_ast.expr_constant(cx, node, privilege)

  elseif node:is(ast.typed.ExprFunction) then
    return flow_from_ast.expr_function(cx, node, privilege)

  elseif node:is(ast.typed.ExprFieldAccess) then
    return flow_from_ast.expr_field_access(cx, node, privilege)

  elseif node:is(ast.typed.ExprIndexAccess) then
    return flow_from_ast.expr_index_access(cx, node, privilege)

  elseif node:is(ast.typed.ExprMethodCall) then
    return flow_from_ast.expr_method_call(cx, node, privilege)

  elseif node:is(ast.typed.ExprCall) then
    return flow_from_ast.expr_call(cx, node, privilege)

  elseif node:is(ast.typed.ExprCast) then
    return flow_from_ast.expr_cast(cx, node, privilege)

  elseif node:is(ast.typed.ExprCtor) then
    return flow_from_ast.expr_ctor(cx, node, privilege)

  elseif node:is(ast.typed.ExprRawContext) then
    return flow_from_ast.expr_raw_context(cx, node, privilege)

  elseif node:is(ast.typed.ExprRawFields) then
    return flow_from_ast.expr_raw_fields(cx, node, privilege)

  elseif node:is(ast.typed.ExprRawPhysical) then
    return flow_from_ast.expr_raw_physical(cx, node, privilege)

  elseif node:is(ast.typed.ExprRawRuntime) then
    return flow_from_ast.expr_raw_runtime(cx, node, privilege)

  elseif node:is(ast.typed.ExprRawValue) then
    return flow_from_ast.expr_raw_value(cx, node, privilege)

  elseif node:is(ast.typed.ExprIsnull) then
    return flow_from_ast.expr_isnull(cx, node, privilege)

  elseif node:is(ast.typed.ExprNew) then
    return flow_from_ast.expr_new(cx, node, privilege)

  elseif node:is(ast.typed.ExprNull) then
    return flow_from_ast.expr_null(cx, node, privilege)

  elseif node:is(ast.typed.ExprDynamicCast) then
    return flow_from_ast.expr_dynamic_cast(cx, node, privilege)

  elseif node:is(ast.typed.ExprStaticCast) then
    return flow_from_ast.expr_static_cast(cx, node, privilege)

  elseif node:is(ast.typed.ExprIspace) then
    return flow_from_ast.expr_ispace(cx, node, privilege)

  elseif node:is(ast.typed.ExprRegion) then
    return flow_from_ast.expr_region(cx, node, privilege)

  elseif node:is(ast.typed.ExprPartition) then
    return flow_from_ast.expr_partition(cx, node, privilege)

  elseif node:is(ast.typed.ExprCrossProduct) then
    return flow_from_ast.expr_cross_product(cx, node, privilege)

  elseif node:is(ast.typed.ExprUnary) then
    return flow_from_ast.expr_unary(cx, node, privilege)

  elseif node:is(ast.typed.ExprBinary) then
    return flow_from_ast.expr_binary(cx, node, privilege)

  elseif node:is(ast.typed.ExprDeref) then
    return flow_from_ast.expr_deref(cx, node, privilege)

  else
    assert(false, "unexpected node type " .. tostring(node.node_type))
  end
end

function flow_from_ast.block(cx, node)
  return node.stats:map(
    function(stat) return flow_from_ast.stat(cx, stat) end)
end

function flow_from_ast.stat_if(cx, node)
  return as_opaque_stat(cx, node)
end

function flow_from_ast.stat_while(cx, node)
  return as_opaque_stat(cx, node)
end

function flow_from_ast.stat_for_num(cx, node)
  local values = node.values:map(
    function(value) return flow_from_ast.expr(cx, value, "reads") end)

  local block_cx = cx:new_local_scope()
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local inner_privileges = privilege_summary(block_cx, block_privileges, false)
  local outer_privileges = privilege_summary(block_cx, block_privileges, true)
  for region_type, privilege in pairs(inner_privileges) do
    if privilege ~= "none" then
      preopen_region_tree(block_cx, region_type, privilege)
    end
  end
  local block = flow_from_ast.block(block_cx, node.block)

  local inputs = terralib.newlist()
  for i, value in ipairs(values) do
    inputs[i] = {"reads", value}
  end
  do
    assert(#values <= 3)
    local i = 4
    for region_type, privilege in pairs(outer_privileges) do
      inputs[i] = {privilege, open_region_tree(cx, region_type, nil, privilege)}
    end
  end

  as_fornum_stat(cx, node.symbol, block_cx.graph, node.parallel, inputs, node.span)
end

function flow_from_ast.stat_for_list(cx, node)
  local value = flow_from_ast.expr(cx, node.value, "none")

  local block_cx = cx:new_local_scope()
  local block_privileges = analyze_privileges.block(block_cx, node.block)
  local inner_privileges = privilege_summary(block_cx, block_privileges, false)
  local outer_privileges = privilege_summary(block_cx, block_privileges, true)
  for region_type, privilege in pairs(inner_privileges) do
    preopen_region_tree(block_cx, region_type, privilege)
  end
  local block = flow_from_ast.block(block_cx, node.block)

  local inputs = terralib.newlist({{"none", value}})
  for region_type, privilege in pairs(outer_privileges) do
    inputs:insert({privilege, open_region_tree(cx, region_type, nil, privilege)})
  end

  as_forlist_stat(cx, node.symbol, block_cx.graph, node.vectorize, inputs, node.span)
end

function flow_from_ast.stat_repeat(cx, node)
  return as_opaque_stat(cx, node)
end

function flow_from_ast.stat_block(cx, node)
  return flow_from_ast.block(cx, node.block)
end

function flow_from_ast.stat_var(cx, node)
  -- FIXME: Workaround for bug in inline optimization.
  if std.all(
    node.types:map(
      function(type) return type == terralib.types.unit end))
  then
    return
  end

  return as_opaque_stat(cx, node)
end

function flow_from_ast.stat_var_unpack(cx, node)
  return as_opaque_stat(cx, node)
end

function flow_from_ast.stat_return(cx, node)
  return as_opaque_stat(cx, node)
end

function flow_from_ast.stat_break(cx, node)
  return as_opaque_stat(cx, node)
end

function flow_from_ast.stat_assignment(cx, node)
  return as_opaque_stat(cx, node)
end

function flow_from_ast.stat_reduce(cx, node)
  local rhs = node.rhs:map(
    function(rh) return {flow_from_ast.expr(cx, rh, "reads")} end)
  local lhs = node.lhs:map(
    function(lh) return {flow_from_ast.expr(cx, lh, "reads_writes")} end)


  local inputs = terralib.newlist()
  inputs:insertall(lhs)
  inputs:insertall(rhs)

  return as_reduce_stat(cx, node.op, inputs, node.span)
end

function flow_from_ast.stat_expr(cx, node)
  local result_nid = flow_from_ast.expr(cx, node.expr, "reads")
  if cx.graph:node_label(result_nid):is(flow.node.Scalar) then
    return sequence_advance(cx, cx.graph:immediate_predecessor(result_nid))
  else
    return sequence_advance(cx, result_nid)
  end
end

function flow_from_ast.stat(cx, node)
  if node:is(ast.typed.StatIf) then
    return flow_from_ast.stat_if(cx, node)

  elseif node:is(ast.typed.StatWhile) then
    return flow_from_ast.stat_while(cx, node)

  elseif node:is(ast.typed.StatForNum) then
    return flow_from_ast.stat_for_num(cx, node)

  elseif node:is(ast.typed.StatForList) then
    return flow_from_ast.stat_for_list(cx, node)

  elseif node:is(ast.typed.StatRepeat) then
    return flow_from_ast.stat_repeat(cx, node)

  elseif node:is(ast.typed.StatBlock) then
    return flow_from_ast.stat_block(cx, node)

  elseif node:is(ast.typed.StatVar) then
    return flow_from_ast.stat_var(cx, node)

  elseif node:is(ast.typed.StatVarUnpack) then
    return flow_from_ast.stat_var_unpack(cx, node)

  elseif node:is(ast.typed.StatReturn) then
    return flow_from_ast.stat_return(cx, node)

  elseif node:is(ast.typed.StatBreak) then
    return flow_from_ast.stat_break(cx, node)

  elseif node:is(ast.typed.StatAssignment) then
    return flow_from_ast.stat_assignment(cx, node)

  elseif node:is(ast.typed.StatReduce) then
    return flow_from_ast.stat_reduce(cx, node)

  elseif node:is(ast.typed.StatExpr) then
    return flow_from_ast.stat_expr(cx, node)

  else
    assert(false, "unexpected node type " .. tostring(node:type()))
  end
end

function flow_from_ast.stat_task(cx, node)
  local task = node.prototype
  local cx = cx:new_task_scope(task:get_constraints(),
                               task:get_region_universe())
  analyze_regions.stat_task(cx, node)
  flow_from_ast.block(cx, node.body)
  return node { body = cx.graph }
end

function flow_from_ast.stat_top(cx, node)
  if node:is(ast.typed.StatTask) then
    return flow_from_ast.stat_task(cx, node)

  else
    return node
  end
end

function flow_from_ast.entry(node)
  local cx = context.new_global_scope()
  return flow_from_ast.stat_top(cx, node)
end

return flow_from_ast
