<h3>RDIR Compiler Plug-in</h3> 

```
RDIR is a dataflow intermediate representation for optimizing
compilers for parallel programming models. This compiler plug-in adds
RDIR support to Regent, a compiler for task-based parallel programs,
with a translator (both AST->RDIR and RDIR->AST) and a number of
sample optimizations which demonstrate how to use the plug-in.

```
  * Regent: https://github.com/StanfordLegion/legion/tree/regent-0.0/language
  * Legion: https://github.com/StanfordLegion/legion
  * Terra: https://github.com/zdevito/terra

<h3>Installation</h3>

```
 1. Install LLVM *with headers*. (Tested with LLVM 3.5.)
 2. Download and install Regent:

        git clone https://github.com/StanfordLegion/legion.git
        cd legion/language
        git checkout -b rdir regent-0.0
        ./install.py --debug
        export REGENT=$PWD

    Note: In some cases, Terra may fail to auto-detect CUDA. If so
    (and assuming you want to use CUDA), recompile Terra with CUDA
    enabled.

        cd terra
        make clean
        CUDA_HOME=.../path/to/cuda ENABLE_CUDA=1 make

 3. Install RDIR plug-in:

        git clone .../rdir.git
        cd rdir
        cp -r plugin/src $REGENT

<h3>Usage</h3>

```
The following test is included which exercises the optimizations
included with RDIR. Both CPU and GPU code paths should be
working. (GPUs are enabled with the flag -ll:gpu N where N is the
number of GPUs to use.)

Running the following commands will produce output reporting bandwidth
and compute throughput on a simple streaming benchmark. In general,
inc1 and inc2 should achieve roughly the same bandwidth. When the RDIR
optimizations are functioning propertly, inc2 will achieve 2x the
compute throughput as inc1 (due to loop and task fusion).

    $REGENT/regent.py examples/inc.rg -fcuda 1 -ll:csize 4096
    $REGENT/regent.py examples/inc.rg -fcuda 1 -ll:gpu 1 -ll:csize 4096 -ll:fsize 4096

<h3>Open Source License</h3>

```
 Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of NVIDIA CORPORATION nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
