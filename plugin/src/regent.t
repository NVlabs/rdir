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

-- Legion Language Entry Point

local ast = require("regent/ast")
local builtins = require("regent/builtins")
local codegen = require("regent/codegen")
local flow_dead_code_elimination = require("regent/flow_dead_code_elimination")
local flow_from_ast = require("regent/flow_from_ast")
local flow_loop_fusion = require("regent/flow_loop_fusion")
local flow_task_fusion = require("regent/flow_task_fusion")
local flow_to_ast = require("regent/flow_to_ast")
local optimize_config_options = require("regent/optimize_config_options")
local optimize_divergence = require("regent/optimize_divergence")
local optimize_futures = require("regent/optimize_futures")
local optimize_inlines = require("regent/optimize_inlines")
local optimize_loops = require("regent/optimize_loops")
local parser = require("regent/parser")
local specialize = require("regent/specialize")
local std = require("regent/std")
local type_check = require("regent/type_check")
local vectorize_loops = require("regent/vectorize_loops")
local inline_tasks = require("regent/inline_tasks")

-- Add Language Builtins to Global Environment

for k, v in pairs(builtins) do
  assert(rawget(_G, k) == nil, "Builtin " .. tostring(k) .. " already defined")
  _G[k] = v
end

-- Add Interface to Helper Functions

_G["regentlib"] = std

-- Compiler

function compile(lex)
  local node = parser:parse(lex)
  local function ctor(environment_function)
    local env = environment_function()
    local ast = specialize.entry(env, node)
    ast = type_check.entry(ast)
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
  return ctor, {node.name}
end

-- Language Definition

-- Note: Keywords marked "reserved for future use" are usually marked
-- as such to indicate that those words are exposed as builtin types,
-- and reserved in case types are ever inducted into the main
-- language.
local language = {
  name = "legion",
  entrypoints = {
    "task",
    "fspace",
    "__demand",
  },
  keywords = {
    "__context",
    "__forbid",
    "__demand",
    "__fields",
    "__parallel",
    "__vectorize",
    "__physical",
    "__raw",
    "__runtime",
    "__inline",
    "__cuda",
    "__unroll",
    "aliased", -- reserved for future use
    "cross_product",
    "disjoint", -- reserved for future use
    "dynamic_cast",
    "index_type", -- reserved for future use
    "isnull",
    "ispace",
    "max",
    "min",
    "new",
    "null",
    "partition",
    "product",
    "ptr", -- reserved for future use
    "reads",
    "reduces",
    "region",
    "static_cast",
    "wild", -- reserved for future use
    "where",
    "writes",
  },
}

-- function language:expression(lex)
--   return function(environment_function)
--   end
-- end

function language:statement(lex)
  return compile(lex)
end

function language:localstatement(lex)
  return compile(lex)
end

return language
