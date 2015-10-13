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

-- Conversion from Dataflow IR to AST

local ast = require("regent/ast")
local flow = require("regent/flow")
local std = require("regent/std")

local context = {}
context.__index = context

function context:new_graph_scope(graph)
  local cx = {
    graph = graph,
    ast = {},
    region_ast = {},
  }
  return setmetatable(cx, context)
end

function context.new_global_scope()
  local cx = {}
  return setmetatable(cx, context)
end

local flow_to_ast = {}

function flow_to_ast.node_opaque(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  local outputs = cx.graph:outgoing_edges_by_port(nid)

  local actions = terralib.newlist()
  for input_port, input in pairs(inputs) do
    if input_port >= 0 then
      assert(#input == 1)
      local input_nid = input[1].from_node
      local input_label = cx.graph:node_label(input_nid)
      if input_label:is(flow.node.Scalar) and input_label.fresh then
        actions:insert(
          ast.typed.StatVar {
            symbols = terralib.newlist({ input_label.value.value }),
            types = terralib.newlist({ input_label.value.expr_type }),
            values = terralib.newlist({
                cx.ast[input_nid],
            }),
            span = input_label.value.span,
        })
      elseif input_label:is(flow.node.Region) and
        not cx.ast[input_nid]:is(ast.typed.ExprID)
      then
        local region_ast = cx.region_ast[input_label.value.expr_type]
        assert(region_ast)
        local action = ast.typed.StatVar {
          symbols = terralib.newlist({ input_label.value.value }),
          types = terralib.newlist({ std.as_read(region_ast.expr_type) }),
          values = terralib.newlist({ region_ast }),
          span = region_ast.span,
        }
        actions:insert(action)
        -- Hack: Stuff the new variable back into the context so
        -- that if another opaque node attempts to read it, it'll
        -- find this one.
        cx.ast[input_nid] = input_label.value
      end
    end
  end

  if not rawget(outputs, cx.graph:node_result_port(nid)) then
    actions:insert(label.action)
    return actions
  else
    assert(#outputs[cx.graph:node_result_port(nid)] == 1)
    local result_nid = outputs[cx.graph:node_result_port(nid)][1].to_node
    local result = cx.graph:node_label(result_nid)
    local read_nids = cx.graph:outgoing_use_set(result_nid)
    if #read_nids > 0 then
      cx.ast[nid] = label.action
      return terralib.newlist(actions)
    else
      return terralib.newlist({
          ast.typed.StatExpr {
            expr = label.action,
            span = label.action.span,
          },
      })
    end
  end
end

function flow_to_ast.node_index_access(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  local outputs = cx.graph:outgoing_edges_by_port(nid)

  assert(rawget(inputs, 1) and #inputs[1] == 1)
  local value = cx.ast[inputs[1][1].from_node]
  assert(rawget(inputs, 2) and #inputs[2] == 1)
  local index = cx.ast[inputs[2][1].from_node]

  local action = ast.typed.ExprIndexAccess {
    value = value,
    index = index,
    expr_type = label.expr_type,
    span = label.span,
  }

  if rawget(outputs, cx.graph:node_result_port(nid)) then
    assert(#outputs[cx.graph:node_result_port(nid)] == 1)
    local result_nid = outputs[cx.graph:node_result_port(nid)][1].to_node
    local result = cx.graph:node_label(result_nid)
    local read_nids = cx.graph:outgoing_use_set(result_nid)
    if #read_nids > 0 then
      cx.ast[nid] = action
      return terralib.newlist()
    end
  end

  return terralib.newlist({
      ast.typed.StatExpr {
        expr = action,
        span = action.span,
      },
  })
end

function flow_to_ast.node_deref(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  local outputs = cx.graph:outgoing_edges_by_port(nid)

  assert(rawget(inputs, 1) and #inputs[1] == 1)
  local value = cx.ast[inputs[1][1].from_node]

  local action = ast.typed.ExprDeref {
    value = value,
    expr_type = label.expr_type,
    span = label.span,
  }

  if rawget(outputs, cx.graph:node_result_port(nid)) then
    assert(#outputs[cx.graph:node_result_port(nid)] == 1)
    local result_nid = outputs[cx.graph:node_result_port(nid)][1].to_node
    local result = cx.graph:node_label(result_nid)
    local read_nids = cx.graph:outgoing_use_set(result_nid)
    if #read_nids > 0 then
      cx.ast[nid] = action
      return terralib.newlist()
    end
  end

  return terralib.newlist({
      ast.typed.StatExpr {
        expr = action,
        span = action.span,
      },
  })
end

function flow_to_ast.node_reduce(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  local outputs = cx.graph:outgoing_edges_by_port(nid)

  local maxport = 0
  for i, _ in pairs(inputs) do
    maxport = std.max(maxport, i)
  end
  assert(maxport % 2 == 0)

  local lhs = terralib.newlist()
  for i = 1, maxport/2 do
    assert(rawget(inputs, i) and #inputs[i] == 1)
    lhs:insert(cx.ast[inputs[i][1].from_node])
  end

  local rhs = terralib.newlist()
  for i = maxport/2 + 1, maxport do
    assert(rawget(inputs, i) and #inputs[i] == 1)
    rhs:insert(cx.ast[inputs[i][1].from_node])
  end

  local action = ast.typed.StatReduce {
    lhs = lhs,
    rhs = rhs,
    op = label.op,
    span = label.span,
  }
  return terralib.newlist({action})
end

function flow_to_ast.node_task(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)
  local outputs = cx.graph:outgoing_edges_by_port(nid)

  local maxport = 0
  for i, _ in pairs(inputs) do
    maxport = std.max(maxport, i)
  end

  assert(rawget(inputs, 1) and #inputs[1] == 1)
  local fn = cx.ast[inputs[1][1].from_node]

  if std.is_task(fn.value) then
    assert(maxport-1 == #fn.value:gettype().parameters)
  end

  local args = terralib.newlist()
  for i = 2, maxport do
    assert(rawget(inputs, i) and #inputs[i] == 1)
    args:insert(cx.ast[inputs[i][1].from_node])
  end

  local action = ast.typed.ExprCall {
    fn = fn,
    args = args,
    inline = "allow",
    expr_type = label.expr_type,
    span = label.span,
  }

  if rawget(outputs, cx.graph:node_result_port(nid)) then
    assert(#outputs[cx.graph:node_result_port(nid)] == 1)
    local result_nid = outputs[cx.graph:node_result_port(nid)][1].to_node
    local result = cx.graph:node_label(result_nid)
    local read_nids = cx.graph:outgoing_use_set(result_nid)
    if #read_nids > 0 then
      assert(result.fresh)
      cx.ast[nid] = action
      return terralib.newlist()
    end
  end

  return terralib.newlist({
      ast.typed.StatExpr {
        expr = action,
        span = action.span,
      },
  })
end

function flow_to_ast.node_for_num(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)

  local maxport = 0
  for i, _ in pairs(inputs) do
    if i <= 3 then
      maxport = std.max(maxport, i)
    end
  end

  local values = terralib.newlist()
  for i = 1, maxport do
    assert(rawget(inputs, i) and #inputs[i] == 1)
    values:insert(cx.ast[inputs[i][1].from_node])
  end

  local block = flow_to_ast.graph(cx, label.block)

  return terralib.newlist({
      ast.typed.StatForNum {
        symbol = label.symbol,
        values = values,
        block = block,
        parallel = label.parallel,
        span = label.span,
      },
  })
end

function flow_to_ast.node_for_list(cx, nid)
  local label = cx.graph:node_label(nid)
  local inputs = cx.graph:incoming_edges_by_port(nid)

  assert(rawget(inputs, 1) and #inputs[1] == 1)
  local value = cx.ast[inputs[1][1].from_node]

  local block = flow_to_ast.graph(cx, label.block)

  return terralib.newlist({
      ast.typed.StatForList {
        symbol = label.symbol,
        value = value,
        block = block,
        vectorize = label.vectorize,
        span = label.span,
      },
  })
end

function flow_to_ast.node_region(cx, nid)
  local label = cx.graph:node_label(nid)

  local inputs = cx.graph:incoming_edges_by_port(nid)
  for _, edges in pairs(inputs) do
    for _, edge in ipairs(edges) do
      if edge.label:is(flow.edge.Name) then
        assert(not cx.region_ast[label.value.expr_type])
        cx.ast[nid] = cx.ast[edge.from_node]
        cx.region_ast[label.value.expr_type] = cx.ast[edge.from_node]
        return terralib.newlist({})
      end
    end
  end

  if cx.region_ast[label.value.expr_type] then
    cx.ast[nid] = cx.region_ast[label.value.expr_type]
  else
    cx.ast[nid] = label.value
  end
  return terralib.newlist({})
end

function flow_to_ast.node_partition(cx, nid)
  local label = cx.graph:node_label(nid)

  local inputs = cx.graph:incoming_edges_by_port(nid)
  for _, edges in pairs(inputs) do
    for _, edge in ipairs(edges) do
      if edge.label:is(flow.edge.Name) then
        assert(not cx.region_ast[label.value.expr_type])
        cx.ast[nid] = cx.ast[edge.from_node]
        cx.region_ast[label.value.expr_type] = cx.ast[edge.from_node]
        return terralib.newlist({})
      end
    end
  end

  if cx.region_ast[label.value.expr_type] then
    cx.ast[nid] = cx.region_ast[label.value.expr_type]
  else
    cx.ast[nid] = label.value
  end
  return terralib.newlist({})
end

function flow_to_ast.node_scalar(cx, nid)
  local label = cx.graph:node_label(nid)
  if label.fresh then
    local inputs = cx.graph:incoming_edges_by_port(nid)
    assert(rawget(inputs, 0) and #inputs[0] == 1)
    cx.ast[nid] = cx.ast[inputs[0][1].from_node]
  else
    cx.ast[nid] = cx.graph:node_label(nid).value
  end
  return terralib.newlist({})
end

function flow_to_ast.node_constant(cx, nid)
  cx.ast[nid] = cx.graph:node_label(nid).value
  return terralib.newlist({})
end

function flow_to_ast.node_function(cx, nid)
  cx.ast[nid] = cx.graph:node_label(nid).value
  return terralib.newlist({})
end

function flow_to_ast.node(cx, nid)
  local label = cx.graph:node_label(nid)
  if label:is(flow.node.Opaque) then
    return flow_to_ast.node_opaque(cx, nid)

  elseif label:is(flow.node.IndexAccess) then
    return flow_to_ast.node_index_access(cx, nid)

  elseif label:is(flow.node.Deref) then
    return flow_to_ast.node_deref(cx, nid)

  elseif label:is(flow.node.Reduce) then
    return flow_to_ast.node_reduce(cx, nid)

  elseif label:is(flow.node.Task) then
    return flow_to_ast.node_task(cx, nid)

  elseif label:is(flow.node.Open) then
    return

  elseif label:is(flow.node.Close) then
    return

  elseif label:is(flow.node.ForNum) then
    return flow_to_ast.node_for_num(cx, nid)

  elseif label:is(flow.node.ForList) then
    return flow_to_ast.node_for_list(cx, nid)

  elseif label:is(flow.node.Region) then
    return flow_to_ast.node_region(cx, nid)

  elseif label:is(flow.node.Partition) then
    return flow_to_ast.node_partition(cx, nid)

  elseif label:is(flow.node.Scalar) then
    return flow_to_ast.node_scalar(cx, nid)

  elseif label:is(flow.node.Constant) then
    return flow_to_ast.node_constant(cx, nid)

  elseif label:is(flow.node.Function) then
    return flow_to_ast.node_function(cx, nid)

  else
    assert(false, "unexpected node type " .. tostring(label:type()))
  end
end

function flow_to_ast.graph(cx, graph)
  assert(flow.is_graph(graph))
  local cx = cx:new_graph_scope(graph)

  local nodes = graph:toposort()
  local stats = terralib.newlist()
  for _, node in ipairs(nodes) do
    local actions = flow_to_ast.node(cx, node)
    if actions then stats:insertall(actions) end
  end
  return ast.typed.Block {
    stats = stats,
    span = ast.trivial_span(),
  }
end

function flow_to_ast.stat_task(cx, node)
  return node { body = flow_to_ast.graph(cx, node.body) }
end

function flow_to_ast.stat_top(cx, node)
  if node:is(ast.typed.StatTask) then
    return flow_to_ast.stat_task(cx, node)

  else
    return node
  end
end

function flow_to_ast.entry(node)
  local cx = context.new_global_scope()
  return flow_to_ast.stat_top(cx, node)
end

return flow_to_ast
