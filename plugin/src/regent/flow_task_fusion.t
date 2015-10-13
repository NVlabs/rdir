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

-- Task Fusion Optimization for Dataflow IR

local ast = require("regent/ast")
local codegen = require("regent/codegen")
local flow = require("regent/flow")
local flow_dead_code_elimination = require("regent/flow_dead_code_elimination")
local flow_from_ast = require("regent/flow_from_ast")
local flow_loop_fusion = require("regent/flow_loop_fusion")
local flow_region_tree = require("regent/flow_region_tree")
local flow_to_ast = require("regent/flow_to_ast")
local inline_tasks = require("regent/inline_tasks")
local optimize_config_options = require("regent/optimize_config_options")
local optimize_divergence = require("regent/optimize_divergence")
local optimize_futures = require("regent/optimize_futures")
local optimize_inlines = require("regent/optimize_inlines")
local optimize_loops = require("regent/optimize_loops")
local std = require("regent/std")
local vectorize_loops = require("regent/vectorize_loops")

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

local flow_task_fusion = {}

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

local function is_opaque(cx, nid)
  return cx.graph:node_label(nid).opaque
end

local function can_fuse(cx, task1, task2)
  return not has_compute_nodes_between(cx, task1, task2) and
    not is_opaque(cx, task1) and
    not is_opaque(cx, task2)
end

local function get_task_fn(cx, inputs)
  assert(rawget(inputs, 1) and #inputs[1] == 1)
  local nid = inputs[1][1].from_node
  local label = cx.graph:node_label(nid)
  assert(label:is(flow.node.Function) and std.is_task(label.value.value))
  return label.value.value
end

local function get_arg_type(cx, inputs, i)
  assert(rawget(inputs, i+1) and #inputs[i+1] == 1)
  local nid = inputs[i+1][1].from_node
  local label = cx.graph:node_label(nid)
  return label.value.expr_type
end

local function fuse_params(cx, fn1, fn2, inputs1, inputs2)
  assert(std.is_task(fn1) and std.is_task(fn2))
  local param1_mapping = {}
  local param2_mapping = {}
  local arg_to_param1_mapping = {}
  local param_type_mapping = {}

  local next_i = 1
  local param_symbols1 = fn1:get_param_symbols()
  for i, param_symbol in ipairs(param_symbols1) do
    local param_type = param_symbol.type
    local arg_type = get_arg_type(cx, inputs1, i)
    param1_mapping[i] = next_i
    if std.is_ispace(param_type) or std.is_region(param_type) or
      std.is_partition(param_type) or std.is_cross_product(param_type)
    then
      arg_to_param1_mapping[arg_type] = i

      assert(not param_type_mapping[param_symbol])
      param_type_mapping[param_symbol] = param_symbol
      assert(not param_type_mapping[param_type])
      param_type_mapping[param_type] = param_type
    else
      param_type_mapping[param_symbol] = param_symbol
    end
    next_i = next_i + 1
  end
  local param_symbols2 = fn2:get_param_symbols()
  for i, param_symbol in ipairs(param_symbols2) do
    local param_type = param_symbol.type
    local arg_type = get_arg_type(cx, inputs1, i)
    if std.is_ispace(param_type) or std.is_region(param_type) or
      std.is_partition(param_type) or std.is_cross_product(param_type)
    then
      if arg_to_param1_mapping[arg_type] then
        param2_mapping[i] = param1_mapping[arg_to_param1_mapping[arg_type]]

        assert(not param_type_mapping[param_symbol] or
                 param_type_mapping[param_symbol] == param_symbols1[
                   arg_to_param1_mapping[arg_type]])
        param_type_mapping[param_symbol] = param_symbols1[
          arg_to_param1_mapping[arg_type]]
        assert(not param_type_mapping[param_type] or
                 param_type_mapping[param_type] == param_symbols1[
                   arg_to_param1_mapping[arg_type]].type)
        param_type_mapping[param_type] = param_symbols1[
          arg_to_param1_mapping[arg_type]].type
      else
        assert(not param_type_mapping[param_symbol])
        param_type_mapping[param_symbol] = param_symbol
        assert(not param_type_mapping[param_type])
        param_type_mapping[param_type] = param_type

        param2_mapping[i] = next_i
        next_i = next_i + 1
      end
    else
      param_type_mapping[param_symbol] = param_symbol
      param2_mapping[i] = next_i
      next_i = next_i + 1
    end
  end
  return param1_mapping, param2_mapping, param_type_mapping
end

local function fuse_tasks(params1_mapping, params2_mapping, mapping, fn1, fn2)
  assert(std.is_task(fn1) and std.is_task(fn2))
  assert(fn1:getast() and fn2:getast())
  local node1 = fn1:getast()
  local node2 = fn2:getast()

  local name = "__fused__" .. node1.name .. "__" .. node2.name

  local symbol1_mapping = {}
  local symbol2_mapping = {}
  local params = terralib.newlist()
  -- FIXME: This hack wouldn't be necessarily if codegen propertly
  -- understood symbols as params. (I.e. different symbols with the
  -- same displayname shouldn't cause any problems.)
  for _, param in ipairs(node1.params) do
    local symbol = terralib.newsymbol(
      std.type_sub(param.param_type, mapping),
      param.symbol.displayname .. "_1"
      -- tostring(terralib.newsymbol())
    )
    symbol1_mapping[param.symbol] = symbol
    params:insert(param { symbol = symbol })
  end
  for _, param in ipairs(node2.params) do
    if not mapping[param.param_type] then
      local symbol = terralib.newsymbol(
        std.type_sub(param.param_type, mapping),
        param.symbol.displayname .. "_2"
        -- tostring(terralib.newsymbol())
      )
      symbol2_mapping[param.symbol] = symbol
      params:insert(param { symbol = symbol })
    else
      assert(symbol1_mapping[param.symbol])
      symbol2_mapping[param.symbol] = symbol1_mapping[param.symbol]
    end
  end

  assert(node1.return_type == terralib.types.unit)
  local return_type = std.type_sub(node2.return_type, mapping)

  local privileges = terralib.newlist()
  for _, privilege_list in ipairs(node1.privileges) do
    for _, privilege in ipairs(privilege_list) do
      privileges:insert(
        std.privilege(
          privilege.privilege,
          terralib.newlist({
              {
                region = symbol1_mapping[privilege.region],
                fields = terralib.newlist({privilege.field_path}),
              },
      })))
    end
  end
  for _, privilege_list in ipairs(node2.privileges) do
    for _, privilege in ipairs(privilege_list) do
      privileges:insert(
        std.privilege(
          privilege.privilege,
          terralib.newlist({
              {
                region = symbol2_mapping[privilege.region],
                fields = terralib.newlist({privilege.field_path}),
              },
      })))
    end
  end

  local constraints = terralib.newlist()
  for _, constraint in ipairs(node1.constraints) do
    constraints:insert(
      std.constraint(
        symbol1_mapping[constraint.lhs],
        symbol1_mapping[constraint.rhs],
        constraint.op))
  end
  for _, constraint in ipairs(node2.constraints) do
    constraints:insert(
      std.constraint(
        symbol2_mapping[constraint.lhs],
        symbol2_mapping[constraint.rhs],
        constraint.op))
  end

  local graph_constraints = {}
  for op, v1 in pairs(node1.prototype:get_constraints()) do
    graph_constraints[op] = graph_constraints[op] or {}
    for lhs, v2 in pairs(v1) do
      local new_lhs = mapping[lhs] or lhs
      graph_constraints[op][new_lhs] = graph_constraints[op][new_lhs] or {}
      for rhs, _ in pairs(v2) do
        local new_rhs = mapping[rhs] or rhs
        graph_constraints[op][new_lhs][new_rhs] = true
      end
    end
  end
  for op, v1 in pairs(node2.prototype:get_constraints()) do
    graph_constraints[op] = graph_constraints[op] or {}
    for lhs, v2 in pairs(v1) do
      local new_lhs = mapping[lhs] or lhs
      graph_constraints[op][new_lhs] = graph_constraints[op][new_lhs] or {}
      for rhs, _ in pairs(v2) do
        local new_rhs = mapping[rhs] or rhs
        graph_constraints[op][new_lhs][new_rhs] = true
      end
    end
  end

  local region_universe = {}
  for region, _ in pairs(node1.prototype:get_region_universe()) do
    local new_region = mapping[region] or region
    region_universe[new_region] = true
  end
  for region, _ in pairs(node2.prototype:get_region_universe()) do
    local new_region = mapping[region] or region
    region_universe[new_region] = true
  end

  local stats = terralib.newlist()
  local body = ast.typed.Block {
    stats = terralib.newlist({
        ast.typed.StatExpr {
          expr = ast.typed.ExprCall {
            fn = ast.typed.ExprFunction {
              value = fn1,
              expr_type = fn1:gettype(),
              span = node1.span,
            },
            args = node1.params:map(
              function(param)
                return ast.typed.ExprID {
                  value = symbol1_mapping[param.symbol],
                  expr_type = symbol1_mapping[param.symbol].type,
                  span = param.span,
                }
            end),
            inline = "demand",
            expr_type = std.type_sub(fn1:gettype().returntype, mapping),
            span = node1.span,
          },
          span = node1.span,
        },
        ast.typed.StatExpr {
          expr = ast.typed.ExprCall {
            fn = ast.typed.ExprFunction {
              value = fn2,
              expr_type = fn2:gettype(),
              span = node2.span,
            },
            args = node2.params:map(
              function(param)
                return ast.typed.ExprID {
                  value = symbol2_mapping[param.symbol],
                  expr_type = symbol2_mapping[param.symbol].type,
                  span = param.span,
                }
            end),
            inline = "demand",
            expr_type = std.type_sub(fn2:gettype().returntype, mapping),
            span = node2.span,
          },
          span = node2.span,
        },
    }),
    span = node1.span,
  }

  local prototype = std.newtask(name)
  prototype:set_param_symbols(params:map(function(param) return param.symbol end))
  local task_type = terralib.types.functype(
    params:map(function(param) return param.param_type end), return_type, false)
  prototype:settype(task_type)
  prototype:setprivileges(privileges)
  prototype:set_param_constraints(constraints)
  prototype:set_constraints(graph_constraints)
  prototype:set_region_universe(region_universe)

  local new_task = ast.typed.StatTask {
    name = name,
    params = params,
    return_type = return_type,
    privileges = privileges,
    constraints = constraints,
    body = body,
    config_options = ast.typed.StatTaskConfigOptions {
      leaf = false,
      inner = false,
      idempotent = false,
    },
    region_divergence = false,
    prototype = prototype,
    inline = node1.inline,
    cuda = node1.cuda,
    span = node1.span,
  }

  -- Follow through with the rest of the compilation pipeline.
  do
    local ast = new_task
    if std.config["task-inlines"] then ast = inline_tasks.entry(ast) end
    ast = flow_from_ast.entry(ast)
    ast = flow_loop_fusion.entry(ast)
    ast = flow_task_fusion.entry(ast)
    ast = flow_dead_code_elimination.entry(ast)
    ast = flow_to_ast.entry(ast)
    if std.config["index-launches"] then ast = optimize_loops.entry(ast) end
    if std.config["futures"] then ast = optimize_futures.entry(ast) end
    if std.config["inlines"] then ast = optimize_inlines.entry(ast) end
    if std.config["leaf"] then ast = optimize_config_options.entry(ast) end
    if std.config["no-dynamic-branches"] then ast = optimize_divergence.entry(ast) end
    if std.config["vectorize"] then ast = vectorize_loops.entry(ast) end
    ast = codegen.entry(ast)
    return ast
  end
end

local function fuse(cx, task1, task2)
  local inputs1 = cx.graph:incoming_edges_by_port(task1)
  local inputs2 = cx.graph:incoming_edges_by_port(task2)
  local outputs1 = cx.graph:outgoing_edges_by_port(task1)
  local outputs2 = cx.graph:outgoing_edges_by_port(task2)
  local fn1 = get_task_fn(cx, inputs1)
  local fn2 = get_task_fn(cx, inputs2)
  local param1_mapping, param2_mapping, param_type_mapping = fuse_params(
    cx, fn1, fn2, inputs1, inputs2)
  local new_fn = fuse_tasks(
    param1_mapping, param2_mapping, param_type_mapping, fn1, fn2)
  assert(std.is_task(new_fn))

  local new_nid = cx.graph:add_node(
    flow.node.Task {
      opaque = false,
      expr_type = std.type_sub(
        cx.graph:node_label(task2).expr_type,
        param_type_mapping),
      span = cx.graph:node_label(task1).span,
  })

  local new_fn_nid = cx.graph:add_node(
    flow.node.Function {
      value = ast.typed.ExprFunction {
        value = new_fn,
        expr_type = new_fn:gettype(),
        span = new_fn:getast().span,
      },
  })

  cx.graph:add_edge(
    flow.edge.Read {},
    new_fn_nid, cx.graph:node_result_port(new_fn_nid),
    new_nid, 1)

  local inputs_mapped = {}
  for i, param in pairs(param1_mapping) do
    assert(rawget(inputs1, i+1) and #inputs1[i+1] == 1)
    local edge = inputs1[i+1][1]
    inputs_mapped[param] = true
    cx.graph:add_edge(
      edge.label, edge.from_node, edge.from_port,
      new_nid, param+1)
  end
  for i, param in pairs(param2_mapping) do
    assert(rawget(inputs2, i+1) and #inputs2[i+1] == 1)
    local edge = inputs2[i+1][1]
    if not inputs_mapped[param] then
      inputs_mapped[param] = true
      cx.graph:add_edge(
        edge.label, edge.from_node, edge.from_port,
        new_nid, param+1)
    end
  end

  local outputs_mapped = {}
  for i, param in pairs(param1_mapping) do
    if rawget(outputs1, i+1) then
      assert(#outputs1[i+1] == 1)
      local edge = outputs1[i+1][1]
      outputs_mapped[param] = true
      cx.graph:add_edge(
        edge.label, new_nid, param+1,
        edge.to_node, edge.to_port)
    end
  end
  for i, param in pairs(param2_mapping) do
    if rawget(outputs2, i+1) then
      assert(#outputs2[i+1] == 1)
      local edge = outputs2[i+1][1]
      if not outputs_mapped[param] then
        outputs_mapped[param] = true
        cx.graph:add_edge(
          edge.label, new_nid, param+1,
          edge.to_node, edge.to_port)
      end
    end
  end

  if rawget(outputs2, cx.graph:node_result_port(task2)) then
    assert(#outputs2[cx.graph:node_result_port(task2)] == 1)
    local edge = outputs2[cx.graph:node_result_port(task2)][1]
    cx.graph:add_edge(
      edge.label,
      new_nid, cx.graph:node_result_port(new_nid),
      edge.to_node, edge.to_port)
  end

  return new_nid
end

local function fuse_eligible_task(cx, tasks)
  for task1, _ in pairs(tasks) do
    for task2, _ in pairs(tasks) do
      if task1 ~= task2 then
        if cx.graph:reachable(task1, task2) then
          if can_fuse(cx, task1, task2) then
            local new_task = fuse(cx, task1, task2)
            cx.graph:remove_node(task1)
            cx.graph:remove_node(task2)
            tasks[task1] = nil
            tasks[task2] = nil
            tasks[new_task] = true
            return true
          end
        end
      end
    end
  end
  return false
end

local function fuse_eligible_task_pairs(cx)
  local original_tasks = cx.graph:filter_nodes(
    function(nid, label) return label:is(flow.node.Task) end)
  local tasks = {}
  for _, nid in pairs(original_tasks) do
    tasks[nid] = true
  end
  repeat until not fuse_eligible_task(cx, tasks)
end

function flow_task_fusion.node_for_num(cx, nid)
  local label = cx.graph:node_label(nid)
  local block = flow_task_fusion.graph(cx, label.block)
  cx.graph:set_node_label(nid, label { block = block })
  return nid
end

function flow_task_fusion.node_for_list(cx, nid)
  local label = cx.graph:node_label(nid)
  local block = flow_task_fusion.graph(cx, label.block)
  cx.graph:set_node_label(nid, label { block = block })
  return nid
end

function flow_task_fusion.node(cx, nid)
  local label = cx.graph:node_label(nid)
  if label:is(flow.node.Opaque) then
    return

  elseif label:is(flow.node.Reduce) then
    return

  elseif label:is(flow.node.Task) then
    return

  elseif label:is(flow.node.IndexAccess) then
    return

  elseif label:is(flow.node.Deref) then
    return

  elseif label:is(flow.node.Open) then
    return

  elseif label:is(flow.node.Close) then
    return

  elseif label:is(flow.node.ForNum) then
    return flow_task_fusion.node_for_num(cx, nid)

  elseif label:is(flow.node.ForList) then
    return flow_task_fusion.node_for_list(cx, nid)

  elseif label:is(flow.node.Region) then
    return

  elseif label:is(flow.node.Partition) then
    return

  elseif label:is(flow.node.Scalar) then
    return

  elseif label:is(flow.node.Constant) then
    return

  elseif label:is(flow.node.Function) then
    return

  else
    assert(false, "unexpected node type " .. tostring(label:type()))
  end
end

function flow_task_fusion.graph(cx, graph)
  assert(flow.is_graph(graph))
  local cx = cx:new_graph_scope(graph:copy())
  cx.graph:map_nodes(function(nid) flow_task_fusion.node(cx, nid) end)
  fuse_eligible_task_pairs(cx)
  return cx.graph
end

function flow_task_fusion.stat_task(cx, node)
  return node { body = flow_task_fusion.graph(cx, node.body) }
end

function flow_task_fusion.stat_top(cx, node)
  if node:is(ast.typed.StatTask) then
    return flow_task_fusion.stat_task(cx, node)

  else
    return node
  end
end

function flow_task_fusion.entry(node)
  local cx = context.new_global_scope()
  return flow_task_fusion.stat_top(cx, node)
end

return flow_task_fusion
