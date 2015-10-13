# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# RDIR

This document describes the formal properties of RDIR.

## Structure

The RDIR format consists of a directed graph of labeled nodes and
edges. In the full implementation of RDIR, edges connect to nodes at
numbered ports. However, most interesting properties of the
representation can be modeled without numbered ports. This document
will concern itself with this simplified graph model, and with a
reduced set of labels.

For the purposes of this document, nodes may be labeled with one of
the following labels:

  * Task
  * Open
  * Close
  * Region (with a region name, see below)

Collectively, Task, Open and Close nodes are called Compute nodes, and
Region nodes are called Data nodes.

Edges may be labeled with the following labels:

  * Read
  * Discard
  * Write
  * Reduce (with a commutative reduction operator such as + or *)
  * HappensBefore (added as a separate step, see below)

## Well-formedness

Certain properties must be obeyed for a labeled graph to be considered
well-formed RDIR.

  * The graph must be acyclic.

  * All Read and Discard edges must flow from a Data node to a Compute
    node.

  * All Write and Reduce edges must flow from a compute node to a data
    node.

## Data-race freedom

While the well-formedness properties of RDIR ensure that graphs are
sensible, they allow for graphs which contain data-races and other bad
properties. Some additional properties are required to ensure that
graphs avoid these pitfalls.

  * Every Data node must have at most one incoming Write edges, or one
    or more incoming Reduce edges (with the same reduction operator),
    but not both. This ensures that writes are race-free.

  * For every pair of Data nodes with edges that might alias (see
    below), one node must be reachable from the other (or vice
    versa). This ensures that each Region (or set of potentially
    aliased Regions) has a consistent version history.

In addition, several constraints are placed on each of the types of
Compute nodes:

### Task

  * For every Write edge connected to a Task node, the node must have
    a Read or Discard edge for the same Region.

  * The set of nodes connected to a Task by Read edges must not alias.

### Open

  * Each Open node must Read exactly one Region node (the Read set).

  * The Write set of an Open node must be subregions of the Read set.

  * Open nodes may not have Discard or Reduce edges.

### Close

  * Each Close node must Write exactly one Region node (the Write set).

  * The Read set of a Close node must be subregions of the Write set.

  * Close nodes may not have Discard or Reduce edges.

## Serializability

Every graph satisfying the above properties is data-race free and has
a straightforward semantics. However, not all valid graphs produce
valid Legion programs. This is because not all valid graphs are
serializable. Consider the following graph:

Regions A, B -> Task t1 -> Region A
Regions A, B -> Task t2 -> Region B

This graph is valid and data-race free, if we assume that separate
memory is allocated for each of the tasks and both copies are
initialized with the state of regions A and B prior to the start of
either task. However, there is no way to serialize this graph into a
sequence of task calls such that the Legion runtime can reconstruct
the dataflow graph at runtime.

We solve this by adding extra HappensBefore edges to the graph, which
constrain execution to ensure that it is serializable. In cases such
as the one above, the addition of HappensBefore edges will introduce a
cycle into the graph, allowing the graph to be rejected. Graphs which
do not contain cycles after the addition of HappensBefore edges are
serializable and may be translated to well-formed Legion programs.

HappensBefore edges are added to the graph as follows:

  * For every Compute node C which Writes a Region R, find the set of
    Reads which may potentially alias with R (call this set S). Add a
    HappensBefore Edge to C from any operation (other than C) which
    Reads any Region in the set S.

## Region Graph

Every RDIR graph is accompanied by a Region Graph which describe
several properties of the Region nodes in the dataflow graph:

  * Which Regions may be potentially aliased?
  * Which Regions must be setsets of other Regions?

There are several ways to encode such a graph. One way is to maintain
a constraint graph with statically provable Subregion and Disjointness
information on Regions. (The absense of Disjointness information in
the graph indicates that two nodes may alias.)

## Code Generation

Code generation from RDIR is straightforward:

 1. Perform a topological sort on the graph.

 2. For each Task node, walk the set of connected Data nodes (and edge
 labels) to determine what parameters to pass to the task.
