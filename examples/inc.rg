--  Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
-- 
--  Redistribution and use in source and binary forms, with or without
--  modification, are permitted provided that the following conditions
--  are met:
--   * Redistributions of source code must retain the above copyright
--     notice, this list of conditions and the following disclaimer.
--   * Redistributions in binary form must reproduce the above copyright
--     notice, this list of conditions and the following disclaimer in the
--     documentation and/or other materials provided with the distribution.
--   * Neither the name of NVIDIA CORPORATION nor the names of its
--     contributors may be used to endorse or promote products derived
--     from this software without specific prior written permission.
-- 
--  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
--  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
--  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
--  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
--  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
--  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
--  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
--  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
--  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
-- (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
-- OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import "regent"

-- Compile and link helpers
local cinc
do
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = root_dir .. "../../runtime"
  local inc_cc = root_dir .. "inc.cc"
  local inc_so = os.tmpname() .. ".so" -- root_dir .. "inc.so"
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = "-O2 -std=c++0x -Wall -Werror"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                 inc_cc .. " -o " .. inc_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. inc_cc)
    assert(false)
  end
  terralib.linklibrary(inc_so)
  cinc = terralib.includec("inc.h", {"-I", root_dir, "-I", runtime_dir})
end

local c = regentlib.c

-- Utility functions
function raw(t)
  return terra(x : t) return x.__ptr end
end

terra nop() end

terra min(x : int64, y : int64) : int64
  if x < y then
    return x
  else
    return y
  end
end

terra ite(c : bool, x : int64, y : int64) : int64
  if c then
    return x
  else
    return y
  end
end

-- Tasks
__demand(__cuda)
task inc(r: region(int64), y : int64)
where
  reads(r), writes(r)
do
  __demand(__vectorize)
  for x in r do
    @x += y
  end
end

task inc1(r : region(int64), p : partition(disjoint, r), m : int64, y : int64)
where
  reads(r), writes(r)
do
  __demand(__parallel)
  for i = 0, m do
    inc(p[i], y)
  end
end

task inc2(r : region(int64), p : partition(disjoint, r), m : int64, y : int64)
where
  reads(r), writes(r)
do
  __demand(__parallel)
  for i = 0, m do
    inc(p[i], y)
  end
  __demand(__parallel)
  for i = 0, m do
    inc(p[i], y)
  end
end

__demand(__cuda)
task dummy(r : region(int64))
where reads(r) do
  return 0
end

task dummy1(r : region(int64), p : partition(disjoint, r), m : int64)
where reads(r) do
  var x = 0
  __demand(__parallel)
  for i = 0, m do
    x += dummy(p[i])
  end
  return x
end

terra bulk_allocate(runtime : c.legion_runtime_t,
                    ctx : c.legion_context_t,
                    r : c.legion_logical_region_t,
                    n : int64)
  var is = r.index_space
  var a = c.legion_index_allocator_create(runtime, ctx, is)
  c.legion_index_allocator_alloc(a, n)
  c.legion_index_allocator_destroy(a)
end

terra bulk_coloring(n : int64, m : int64) : c.legion_coloring_t
  var ic = c.legion_coloring_create()
  var npercolor = n/m
  for color = 0, m do
    var start = color*npercolor + min(color, n%m)
    var pieces = npercolor + ite(color < n%m, 1, 0)
    c.legion_coloring_add_range(
      ic, color,
      c.legion_ptr_t { value = start },
      c.legion_ptr_t { start+pieces-1 })
  end
  return ic
end

terra wait_for(x : int)
  return x
end

task test(n: int64, m : int64)
  var r = region(ispace(ptr, n), int64)

  c.printf("allocating...\n")
  bulk_allocate(__runtime(), __context(), __raw(r), n)

  c.printf("coloring...\n")
  var rc = bulk_coloring(n, m)
  c.printf("partitioning...\n")
  var p = partition(disjoint, r, rc)
  c.legion_coloring_destroy(rc)

  c.printf("initializing...\n")
  var i2 = 0
  for x in r do
    @x = i2
    nop() -- FIXME: Codegen messes up without this.
    i2 += 1
  end

  -- Warmup
  c.printf("warmup...\n")
  inc1(r, p, m, 0)
  nop() -- FIXME: Avoid task fusion here.
  inc2(r, p, m, 0)
  nop() -- FIXME: Avoid task fusion here.
  wait_for(dummy1(r, p, m))

  -- Timed runs
  c.printf("timing inc1...\n")
  var start_time = c.legion_get_current_time_in_micros()/1.e6
  inc1(r, p, m, 100)
  nop() -- FIXME: Avoid task fusion here.
  wait_for(dummy1(r, p, m))
  var end_time = c.legion_get_current_time_in_micros()/1.e6
  var total_time = end_time - start_time
  c.printf("inc1 elapsed time = %.6e\n", total_time)
  c.printf("inc1 bandwidth = %.6e\n", n*[sizeof(int64)]*2 / total_time)
  c.printf("inc1 iops = %.6e\n", n / total_time)

  c.printf("timing inc2...\n")
  start_time = c.legion_get_current_time_in_micros()/1.e6
  inc2(r, p, m, 10)
  nop() -- FIXME: Avoid task fusion here.
  wait_for(dummy1(r, p, m))
  end_time = c.legion_get_current_time_in_micros()/1.e6
  total_time = end_time - start_time
  c.printf("inc2 elapsed time = %.6e\n", total_time)
  c.printf("inc2 bandwidth = %.6e\n", n*[sizeof(int64)]*2 / total_time)
  c.printf("inc2 iops = %.6e\n", n*2 / total_time)

  c.printf("validating...\n")
  var i3 : int64 = 0
  for x in r do
    var y = @x
    var z = 120 + i3
    if y ~= z then c.printf("output %ld is %ld\n", i3, y) end
    regentlib.assert(y == z, "test failed")
    i3 += 1
  end
end

task main()
  -- test(16777216, 2) -- 128 MB
  -- test(33554432, 4) -- 256 MB
  -- test(67108864, 8) -- 512 MB
  -- test(134217728, 16) -- 1 GB
  test(268435456, 32) -- 2 GB
  -- test(536870912, 64) -- 4 GB
end
cinc.register_mappers()
regentlib.start(main)
