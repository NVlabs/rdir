%%  Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
%% 
%%  Redistribution and use in source and binary forms, with or without
%%  modification, are permitted provided that the following conditions
%%  are met:
%%   * Redistributions of source code must retain the above copyright
%%     notice, this list of conditions and the following disclaimer.
%%   * Redistributions in binary form must reproduce the above copyright
%%     notice, this list of conditions and the following disclaimer in the
%%     documentation and/or other materials provided with the distribution.
%%   * Neither the name of NVIDIA CORPORATION nor the names of its
%%     contributors may be used to endorse or promote products derived
%%     from this software without specific prior written permission.
%% 
%%  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
%%  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%%  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
%%  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
%%  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
%%  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
%%  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
%%  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
%%  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%%  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% Runs with SWI-Prolog and Yap
% swipl -g main -s graph.prolog graph 4 2
% yap -l graph.prolog -g 'unix(argv(Args)), main(Args), halt.' -- graph 4 2

:- use_module(library(apply)).
:- use_module(library(assoc)).
:- use_module(library(lists)).
:- use_module(library(pairs)).
:- use_module(library(ugraphs)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Yap Compatibility
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

is_dialect(X) :-
    catch(current_prolog_flag(dialect, X), _, fail).
:- if(is_dialect(yap)).

% Throw an error if a predicate is undefined.
user:unknown_predicate_handler(G, _, _) :-
    format('Undefined predicate: ~w~n',[G]), error.

% Compatibility for missing apply library predicates.
foldl(Goal, List, V0, V) :-
    foldl_(List, Goal, V0, V).
foldl_([], _, V, V).
foldl_([H|T], Goal, V0, V) :-
    call(Goal, H, V0, V1),
    foldl_(T, Goal, V1, V).

:- endif.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helpers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generic helpers.
pairs(X, Y, X-Y).
flipped_pairs(Y, X, X-Y).

is_first(X, X-_).
is_second(X, _-X).

% Graph helpers.
incoming_edges(Vertex, Graph, Edges) :-
    edges(Graph, E),
    include(is_second(Vertex), E, Edges).

immediate_predecessors(Vertex, Graph, Vertices) :-
    incoming_edges(Vertex, Graph, E),
    pairs_keys(E, V1),
    sort(V1, Vertices).

is_acyclic(Graph) :-
    vertices(Graph, Vertices),
    empty_assoc(Empty),
    maplist(trace_cycle(Graph, Empty), Vertices).

trace_cycle(Graph, Visited, Vertex) :-
    \+ get_assoc(Vertex, Visited, _),
    put_assoc(Vertex, Visited, true, Visited1),
    neighbors(Vertex, Graph, Vertices),
    maplist(trace_cycle(Graph, Visited1), Vertices).

% Flowgraph helpers.
empty_flowgraph(Graph-VertexTypes-VertexOps-VertexRegions-EdgeTypes) :-
    vertices_edges_to_ugraph([], [], Graph),
    empty_assoc(VertexTypes),
    empty_assoc(VertexOps),
    empty_assoc(VertexRegions),
    empty_assoc(EdgeTypes).

flowgraph(Graph, VertexTypes, VertexOps, VertexRegions, EdgeTypes,
          Graph-VertexTypes-VertexOps-VertexRegions-EdgeTypes).
flowgraph_graph(Flowgraph, Graph) :-
    flowgraph(Graph, _, _, _, _, Flowgraph).
flowgraph_vertex_types(Flowgraph, VertexTypes) :-
    flowgraph(_, VertexTypes, _, _, _, Flowgraph).
flowgraph_vertex_ops(Flowgraph, VertexOps) :-
    flowgraph(_, _, VertexOps, _, _, Flowgraph).
flowgraph_vertex_regions(Flowgraph, VertexRegions) :-
    flowgraph(_, _, _, VertexRegions, _, Flowgraph).
flowgraph_edge_types(Flowgraph, EdgeTypes) :-
    flowgraph(_, _, _, _, EdgeTypes, Flowgraph).

vertex_type(Flowgraph, Vertex, VertexType) :-
    flowgraph_vertex_types(Flowgraph, VertexTypes),
    get_assoc(Vertex, VertexTypes, VertexType).
vertex_op(Flowgraph, Vertex, VertexOp) :-
    flowgraph_vertex_ops(Flowgraph, VertexOps),
    get_assoc(Vertex, VertexOps, VertexOp).
vertex_region(Flowgraph, Vertex, VertexRegion) :-
    flowgraph_vertex_regions(Flowgraph, VertexRegions),
    get_assoc(Vertex, VertexRegions, VertexRegion).
edge_type(Flowgraph, Edge, EdgeType) :-
    flowgraph_edge_types(Flowgraph, EdgeTypes),
    get_assoc(Edge, EdgeTypes, EdgeType).

has_vertex_type(Flowgraph, VertexType, Vertex) :-
    vertex_type(Flowgraph, Vertex, VertexType).
has_vertex_op(Flowgraph, VertexOp, Vertex) :-
    vertex_op(Flowgraph, Vertex, VertexOp).
has_vertex_region(Flowgraph, VertexRegion, Vertex) :-
    vertex_region(Flowgraph, Vertex, VertexRegion).
has_edge_type(Flowgraph, EdgeType, Edge) :-
    edge_type(Flowgraph, Edge, EdgeType).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Region Tree
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For now, the region tree is manually hard-coded with predicates. It
% would be a combinatorial explosion to make it dynamic anyway.

% Set of possible regions. (For now, use regions only and no
% partitions, to avoid combinatorial explosion.)
region(rA).
%% region(rB). % partition
region(rC). % partition
%% region(rD).
region(rE).
region(rF).

% Is it a region node or aliased partition?
region_aliased(rA).
region_aliased(rB).
%% region_aliased(rC).
region_aliased(rD).
region_aliased(rE).
region_aliased(rF).

% Who is the parent of region?
%% region_parent(rB, rA).
region_parent(rC, rA).
%% region_parent(rD, rB).
region_parent(rE, rC).
region_parent(rF, rC).

region_root(X, R) :-
    region_parent(X, P) ->
        region_root(P, R);
    R = X.

region_roots(R) :-
    findall(Y, (region(X), region_root(X, Y)), Z),
    sort(Z, R).

region_siblings(X, S) :-
    findall(C, (region_parent(X, P), region_parent(C, P), X \= C), S).

region_ancestor(X, A) :-
    region_parent(X, X0),
    (X0 = A; region_ancestor(X0, A)).

% Computing the lowest common ancestor.
region_ancestors(X, A) :- region_ancestors(X, [], A).
region_ancestors(X, A0, A) :-
    region_parent(X, X0) ->
    region_ancestors(X0, [X0|A0], A);
    A = A0.

region_path(X, A) :- region_ancestors(X, [X], A).

compare_prefix([Z0|XS], [Z0|YS], Z) :-
    compare_prefix(Z0, XS, YS, Z).

compare_prefix(_, [Z0|XS], [Z0|YS], Z) :-
    !, compare_prefix(Z0, XS, YS, Z).
compare_prefix(Z, _, _, Z).

region_lowest_common_ancestor(X, Y, Z) :-
    region_ancestors(X, [X], AX),
    region_ancestors(Y, [Y], AY),
    compare_prefix(AX, AY, Z).

% Who can alias with the region?
region_can_alias(X, Y) :-
    (X = Y, !);
    (region_ancestor(X, Y), !);
    (region_ancestor(Y, X), !);
    (region_lowest_common_ancestor(X, Y, Z),
     region_aliased(Z)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Unlabeled Graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Returns a graph Graph with vertices Vertices and at least MinEdges
% edges.
graph(Vertices, MinEdges, Graph) :-
    edges_from_triangular(Vertices, Vertices, [], Edges),
    length(Edges, Count),
    Count >= MinEdges,
    vertices_edges_to_ugraph(Vertices, Edges, Graph).

% Note: There are two ways to enumerate all edges, either by
% enumerating the complete matrix, or just the upper triangle (minus
% diagonal). The upper triangular matrix produces only acyclic graphs
% and also significantly reduces the search space.

edges_from_triangular([], _, E, E).
edges_from_triangular([F1|FS], [_|TS], E0, E) :-
    edges_to(F1, TS, E0, E1),
    edges_from_triangular(FS, TS, E1, E).

edges_from_complete([], _, E, E).
edges_from_complete([F1|FS], TS, E0, E) :-
    edges_to(F1, TS, E0, E1),
    edges_from_complete(FS, TS, E1, E).

edges_to(_, [], E, E).
edges_to(F, [T1|TS], E0, E) :-
    edges_to(F, TS, [F-T1|E0], E);
    edges_to(F, TS, E0, E).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Labeled Graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate wellformed vertex-lablings of the graph Graph.
assign_vertex_labels(Graph, VertexTypes, VertexOps, VertexRegions) :-
    empty_assoc(Empty),
    vertices(Graph, Vertices),
    foldl(assign_vlabels(Graph), Vertices,
          Empty-Empty-Empty-Empty,
          _-VertexTypes-VertexOps-VertexRegions).

% Walk each (potential) root in the graph and assign vertex labels.
assign_vlabels(Graph, Root, Visited0-Types0-Ops0-Regions0,
               Visited-Types-Ops-Regions) :-
    get_assoc(Root, Visited0, _) ->
        Visited-Types-Ops-Regions = Visited0-Types0-Ops0-Regions0;
    (vtype(RootType),
     alternate_vlabels(Graph, RootType, Root,
                       Visited0-Types0-Ops0-Regions0,
                       Visited-Types-Ops-Regions)).

% Assign alternating colors to progressive levels in the graph.
alternate_vlabels(Graph, Type0, Vertex0,
                  Visited0-Types0-Ops0-Regions0,
                  Visited-Types-Ops-Regions) :-
    get_assoc(Vertex0, Visited0, _) ->
        % Check that Vertex0 matches what we expect.
        (get_assoc(Vertex0, Types0, Type0),
         Visited-Types-Ops-Regions = Visited0-Types0-Ops0-Regions0);

    % Choose an op and region for Vertex0 and update maps.
    (vop(Type0, Op0),
     vregion(Type0, Region0),
     put_assoc(Vertex0, Visited0, true, Visited1),
     put_assoc(Vertex0, Types0, Type0, Types1),
     put_assoc(Vertex0, Ops0, Op0, Ops1),
     put_assoc(Vertex0, Regions0, Region0, Regions1),

     % Apply opposite type to neighbors.
     neighbors(Vertex0, Graph, Vertices0),
     negate_vtype(Type0, Type1),
     foldl(alternate_vlabels(Graph, Type1), Vertices0,
           Visited1-Types1-Ops1-Regions1,
           Visited-Types-Ops-Regions)).

% Set of possible vertex types.
vtype(compute).
vtype(data).

negate_vtype(compute, data).
negate_vtype(data, compute).

% Set of possible vertex ops.
vop(compute, task).
vop(compute, open).
vop(compute, close).
vop(data, none).

% Set of possible vertex regions.
vregion(compute, none).
vregion(data, R) :- region(R).

% Generate mostly wellformed edge-lablings EdgeTypes of the graph
% Graph. Edge labels will be wellformed with respect to their source
% and destination types. Writes may not be properly wellformed with
% respect to the existence of corresponding read edges.
assign_edge_labels(Graph, VertexTypes, EdgeTypes) :-
    edges(Graph, Edges),
    maplist(assign_etype(VertexTypes), Edges, EdgeTypes0),
    maplist(pairs, Edges, EdgeTypes0, EdgeTypePairs),
    list_to_assoc(EdgeTypePairs, EdgeTypes).

assign_etype(VertexTypes, Vertex1-_, EdgeType) :-
    get_assoc(Vertex1, VertexTypes, VertexType),
    etype(VertexType, EdgeType).

% Set of possible edge types for a given source vertex types.
etype(data, read).
etype(compute, write).

% Generate all possible labeled graphs Graph, VertexTypes, EdgeTypes,
% from a set of vertices V.
labeled_graph(Vertices, MinEdges, Flowgraph) :-
    graph(Vertices, MinEdges, Graph),
    assign_vertex_labels(Graph, VertexTypes, VertexOps, VertexRegions),
    assign_edge_labels(Graph, VertexTypes, EdgeTypes),
    flowgraph(Graph, VertexTypes, VertexOps, VertexRegions, EdgeTypes,
              Flowgraph).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Well-formed Graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Returns true when the labeled graph Flowgraph is well-formed.
wellformed(Flowgraph) :-
    flowgraph_graph(Flowgraph, Graph),
    edges(Graph, E),
    maplist(wellformed_write_edge(Flowgraph), E).

% Returns true if edge E is a read or a properly formed write.
wellformed_write_edge(Flowgraph, V1-V2) :-
    flowgraph_graph(Flowgraph, Graph),
    (edge_type(Flowgraph, V1-V2, write),
     vertex_op(Flowgraph, V1, task)) ->
        (immediate_predecessors(V1, Graph, VPred),
         maplist(vertex_region(Flowgraph), VPred, VPredRegion),
         vertex_region(Flowgraph, V2, V2Region),
         memberchk(V2Region, VPredRegion));
    true.

% Generate all possible wellformed labeled graphs Graph, VertexTypes,
% EdgeTypes, from a set of vertices Vertices.
wellformed_graph(Vertices, MinEdges, Flowgraph) :-
    labeled_graph(Vertices, MinEdges, Flowgraph),
    wellformed(Flowgraph).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Data-race Free Graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Returns true when the well-formed graph Flowgraph is
% data-race free.
racefree(Flowgraph) :-
    flowgraph_graph(Flowgraph, Graph),
    vertices(Graph, V),
    include(has_vertex_type(Flowgraph, data), V, VData),
    maplist(racefree_single_write_vertex(Flowgraph), VData),
    self_product(VData, VDataPairs),
    maplist(racefree_connected_vertex_pair(Flowgraph), VDataPairs),
    include(has_vertex_op(Flowgraph, task), V, VTask),
    maplist(racefree_task_vertex(Flowgraph), VTask),
    include(has_vertex_op(Flowgraph, open), V, VOpen),
    maplist(racefree_open_vertex(Flowgraph), VOpen),
    include(has_vertex_op(Flowgraph, close), V, VClose),
    maplist(racefree_close_vertex(Flowgraph), VClose).

% Each data node has at most one write.
racefree_single_write_vertex(Flowgraph, Vertex) :-
    flowgraph_graph(Flowgraph, Graph),
    incoming_edges(Vertex, Graph, E),
    include(has_edge_type(Flowgraph, write), E, EWrites),
    length(EWrites, Count),
    Count < 2.

% Potentially aliased data nodes are connected.
racefree_connected_vertex_pair(Flowgraph, V1-V2) :-
    flowgraph_graph(Flowgraph, Graph),
    can_alias(Flowgraph, V1, V2) ->
        ((reachable(V1, Graph, V1Reach), memberchk(V2, V1Reach));
         (reachable(V2, Graph, V2Reach), memberchk(V1, V2Reach)));
    true.

% Each compute node's read set (backwards reachable in one hop)
% contains no pairs of data nodes which might potentially alias.
%
% Note: This is almost certainly stricter than it needs to be,
% particularly is the quality of aliasing information is poor. A
% potentially more lenient rule might be that potentially aliased data
% nodes would need to have been produced by the same operation. This
% would allow e.g. closes to produce multiple regions in a
% plausibly-consistent manner.
racefree_task_vertex(Flowgraph, V) :-
    flowgraph_graph(Flowgraph, Graph),
    immediate_predecessors(V, Graph, VRead),
    self_product(VRead, VReadPairs),
    maplist(cannot_alias(Flowgraph), VReadPairs).

racefree_open_vertex(Flowgraph, V) :-
    % Compute read and write sets.
    flowgraph_graph(Flowgraph, Graph),
    immediate_predecessors(V, Graph, [VRead]),
    vertex_region(Flowgraph, VRead, VReadRegion),
    neighbors(V, Graph, VWrite),
    maplist(vertex_region(Flowgraph), VWrite, [R|Rs]),

    % Check that the read set dominates the write set.
    foldl(region_lowest_common_ancestor, Rs, R, A),
    (is_ancestor(VReadRegion, A) -> true;
     (VReadRegion = A, \+ memberchk(VReadRegion, [R|Rs]))).

racefree_close_vertex(Flowgraph, V) :-
    % Compute read and write sets.
    flowgraph_graph(Flowgraph, Graph),
    immediate_predecessors(V, Graph, VRead),
    maplist(vertex_region(Flowgraph), VRead, [R|Rs]),
    neighbors(V, Graph, [VWrite]),
    vertex_region(Flowgraph, VWrite, VWriteRegion),

    % Check that the write set dominates the read set.
    foldl(region_lowest_common_ancestor, Rs, R, VWriteRegion),

    % Check that the close doesn't touch multiple versions of any region.
    sort([R|Rs], RSet),
    length(RSet, UniqueCount),
    length([R|Rs], Count),
    UniqueCount = Count.

is_ancestor(A, R) :- region_ancestor(R, A).

can_alias(Flowgraph, V1, V2) :-
    vertex_region(Flowgraph, V1, R1),
    vertex_region(Flowgraph, V2, R2),
    region_can_alias(R1, R2).

cannot_alias(Flowgraph, V1, V2) :- \+ can_alias(Flowgraph, V1, V2).
cannot_alias(Flowgraph, V1-V2) :- cannot_alias(Flowgraph, V1, V2).

self_product(L, P) :- triangular(L, L, [], P).

triangular([], _, E, E).
triangular([F1|FS], [_|TS], E0, E) :-
    triangular2(F1, TS, E0, E1),
    triangular(FS, TS, E1, E).

triangular2(_, [], E, E).
triangular2(F, [T1|TS], E0, E) :-
    triangular2(F, TS, [F-T1|E0], E).

% Generate all possible data-race free labeled graphs Flowgraph, from
% a set of vertices V.
racefree_graph(Vertices, MinEdges, Flowgraph) :-
    wellformed_graph(Vertices, MinEdges, Flowgraph),
    racefree(Flowgraph).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Serializable Graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

serializable(Flowgraph0, Flowgraph) :-
    augment_happens_before_graph(Flowgraph0, Flowgraph),
    flowgraph_graph(Flowgraph, Graph),
    is_acyclic(Graph).

% Auguments the well-formed graph Flowgraph0 with happens-before edges
% to create a Flowgraph. Note that the resulting graph may contain
% cycles.
augment_happens_before_graph(Flowgraph0, Flowgraph) :-
    flowgraph(Graph0, VertexTypes0, VertexOps0, VertexRegions0, EdgeTypes0,
              Flowgraph0),
    edges(Graph0, E),
    include(has_edge_type(Flowgraph0, write), E, EWrites),
    foldl(add_happens_before_write(Flowgraph0), EWrites, [], EHappensBefore),
    add_edges(Graph0, EHappensBefore, Graph),
    foldl(put_edge_type(happens_before), EHappensBefore, EdgeTypes0, EdgeTypes),
    flowgraph(Graph, VertexTypes0, VertexOps0, VertexRegions0, EdgeTypes,
              Flowgraph).

add_happens_before_write(Flowgraph, VCompute-VWrite, Edges0, Edges) :-
    flowgraph_graph(Flowgraph, Graph),
    immediate_predecessors(VCompute, Graph, VReads),
    foldl(add_happens_before_write_for_read(Flowgraph, VCompute, VWrite),
          VReads, Edges0, Edges).

add_happens_before_write_for_read(Flowgraph, VCompute, VWrite, VRead, Edges0, Edges) :-
    can_alias(Flowgraph, VWrite, VRead) ->
        (flowgraph_graph(Flowgraph, Graph),
         neighbors(VRead, Graph, VReaders),
         subtract(VReaders, [VCompute], VOtherReaders),
         foldl(add_edge_to(Graph, VCompute), VOtherReaders, Edges0, Edges));
    Edges = Edges0.

add_edge_to(Graph, V2, V1, Edges0, Edges) :-
    (reachable(V1, Graph, V1Reachable),
     \+ memberchk(V2, V1Reachable),
     reachable(V2, Graph, V2Reachable),
     \+ memberchk(V1, V2Reachable)) ->
        Edges = [V1-V2|Edges0];
    Edges = Edges0.

put_edge_type(EdgeType, Edge, EdgeTypes0, EdgeTypes) :-
    put_assoc(Edge, EdgeTypes0, EdgeType, EdgeTypes).

serializable_graph(Vertices, MinEdges, Flowgraph) :-
    racefree_graph(Vertices, MinEdges, Flowgraph0),
    serializable(Flowgraph0, Flowgraph).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Miscellaneous Graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wellformed_not_racefree_graph(Vertices, MinEdges, Flowgraph) :-
    wellformed_graph(Vertices, MinEdges, Flowgraph),
    \+ racefree(Flowgraph).

racefree_not_serializable_graph(Vertices, MinEdges, Flowgraph) :-
    racefree_graph(Vertices, MinEdges, Flowgraph0),
    augment_happens_before_graph(Flowgraph0, Flowgraph),
    flowgraph_graph(Flowgraph, Graph),
    \+ is_acyclic(Graph).

% Proxy for the property of interest.
valid_graph(Vertices, MinEdges, Flowgraph) :-
    serializable_graph(Vertices, MinEdges, Flowgraph).
    %% wellformed_not_racefree_graph(Vertices, MinEdges, Flowgraph).
    %% racefree_not_serializable_graph(Vertices, MinEdges, Flowgraph).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Programs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A program is a list of tasks.
% A task is a list of region requirements.
% A region requirement is a pair of a region and a privilege.

% Returns all programs Program with Tasks tasks where each task takes
% up to MaxArgs arguments.
program(Tasks, MaxArgs, Program) :-
    length(Program, Tasks),
    maplist(create_task(MaxArgs), Program).

create_task(MaxArgs, Task) :-
    between(1, MaxArgs, Args),
    length(Task, Args),
    foldl(create_arg, Task, [], _).

create_arg(Region-Privilege, Regions0, Regions) :-
    region(Region),
    privilege(Privilege),
    include(region_can_alias(Region), Regions0, Aliased),
    length(Aliased, NAliased),
    NAliased = 0,
    Regions = [Region|Regions0].

privilege(read_only).
privilege(read_write).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Translation from Programs to Graphs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

program_to_graph(Program, Flowgraph) :-
    empty_flowgraph(Flowgraph0),
    empty_region_tree(RegionTree0),
    NextVertex0 = 1,
    foldl(translate_task, Program, Flowgraph0-RegionTree0-NextVertex0,
          Flowgraph-_-_).

translate_task(Task, Flowgraph0-RegionTree0-NextVertex0,
               Flowgraph-RegionTree-NextVertex) :-
    foldl(read_arg, Task, Flowgraph0-RegionTree0-NextVertex0-[],
          Flowgraph1-RegionTree1-NextVertex1-VReads),
    create_node(compute, task, none, Flowgraph1-RegionTree1-NextVertex1,
                Vertex, Flowgraph2-RegionTree2-NextVertex2),
    foldl(attach_read_arg(Vertex), VReads, Flowgraph2-RegionTree2-NextVertex2,
          Flowgraph3-RegionTree3-NextVertex3),
    include(arg_is_write, Task, Writes),
    foldl(write_arg, Writes, Flowgraph3-RegionTree3-NextVertex3-[],
          Flowgraph4-RegionTree4-NextVertex4-VWrites),
    foldl(attach_write_arg(Vertex), VWrites, Flowgraph4-RegionTree4-NextVertex4,
          Flowgraph-RegionTree-NextVertex).

arg_is_write(_-read_write).

read_arg(Region-Privilege, Flowgraph0-RegionTree0-NextVertex0-Regions0,
         Flowgraph-RegionTree-NextVertex-[Vertex|Regions0]) :-
    (arg_is_write(Region-Privilege) -> Mode = write; Mode = read),
    open_region_tree(Region, Mode, Flowgraph0-RegionTree0-NextVertex0,
                     Vertex, Flowgraph-RegionTree-NextVertex).

write_arg(Region-_, Flowgraph0-RegionTree0-NextVertex0-Regions0,
         Flowgraph-RegionTree-NextVertex-[Vertex|Regions0]) :-
    mutate_region_tree(Region, write, Flowgraph0-RegionTree0-NextVertex0,
                      Vertex, Flowgraph-RegionTree-NextVertex).

attach_read_arg(VCompute, VRead, Flowgraph0-RegionTree0-NextVertex0,
         Flowgraph-RegionTree-NextVertex) :-
    create_edge(VRead, VCompute, read, Flowgraph0-RegionTree0-NextVertex0,
                Flowgraph-RegionTree-NextVertex).

attach_write_arg(VCompute, VWrite, Flowgraph0-RegionTree0-NextVertex0,
         Flowgraph-RegionTree-NextVertex) :-
    create_edge(VCompute, VWrite, write, Flowgraph0-RegionTree0-NextVertex0,
                Flowgraph-RegionTree-NextVertex).

% Graph manipulation.
create_node(VertexType, VertexOp, VertexRegion,
            Flowgraph0-RegionTree0-Vertex,
            Vertex, Flowgraph-RegionTree0-NextVertex) :-
    flowgraph(Graph0, VertexTypes0, VertexOps0, VertexRegions0, EdgeTypes0,
              Flowgraph0),

    Vertex > 0,
    add_vertices(Graph0, [Vertex], Graph),
    put_assoc(Vertex, VertexTypes0, VertexType, VertexTypes),
    put_assoc(Vertex, VertexOps0, VertexOp, VertexOps),
    put_assoc(Vertex, VertexRegions0, VertexRegion, VertexRegions),

    flowgraph(Graph, VertexTypes, VertexOps, VertexRegions, EdgeTypes0,
              Flowgraph),
    NextVertex is Vertex + 1.

create_edge(V1, V2, EdgeType,
            Flowgraph0-RegionTree0-NextVertex0,
            Flowgraph-RegionTree0-NextVertex0) :-
    flowgraph(Graph0, VertexTypes0, VertexOps0, VertexRegions0, EdgeTypes0,
              Flowgraph0),

    V1 > 0,
    V2 > 0,
    add_edges(Graph0, [V1-V2], Graph),
    put_assoc(V1-V2, EdgeTypes0, EdgeType, EdgeTypes),

    flowgraph(Graph, VertexTypes0, VertexOps0, VertexRegions0, EdgeTypes,
              Flowgraph).

% Region tree state manipulation.
empty_region_tree(RegionTree) :-
    findall(R, region(R), Regions),
    Closed = closed-clean-0-0,
    maplist(flipped_pairs(Closed), Regions, RegionTreeState),
    list_to_assoc([none-Closed|RegionTreeState], RegionTree).

open_region_tree(Region, Mode, Flowgraph0-RegionTree0-NextVertex0,
                 Vertex, Flowgraph-RegionTree-NextVertex) :-
    region_path(Region, Path),
    open_region_tree_walk([none|Path], Mode,
                          Flowgraph0-RegionTree0-NextVertex0,
                          Flowgraph-RegionTree-NextVertex),
    get_assoc(Region, RegionTree, _-_-Vertex-_).

open_region_tree_walk([Node], Mode,
                      Flowgraph0-RegionTree0-NextVertex0,
                      Flowgraph-RegionTree-NextVertex) :-
    get_region_tree_state(RegionTree0, Node, State0),
    select_transition_leaf(State0, Mode, State1, Action1),
    perform_action(Action1, Node, none, Flowgraph0-RegionTree0-NextVertex0,
                   Flowgraph-RegionTree1-NextVertex),
    put_region_tree_state(State1, Node, RegionTree1, RegionTree).

open_region_tree_walk([Node, Next|Path], Mode,
                      Flowgraph0-RegionTree0-NextVertex0,
                      Flowgraph-RegionTree-NextVertex) :-
    get_region_tree_state(RegionTree0, Node, State0),
    select_transition_inner(State0, Mode, Node, Next, RegionTree0, State1, Action1),
    perform_action(Action1, Node, Next, Flowgraph0-RegionTree0-NextVertex0,
                   Flowgraph1-RegionTree1-NextVertex1),
    put_region_tree_state(State1, Node, RegionTree1, RegionTree2),
    open_region_tree_walk([Next|Path], Mode,
                          Flowgraph1-RegionTree2-NextVertex1,
                          Flowgraph-RegionTree-NextVertex).

select_transition_leaf(closed, _, closed, create).
select_transition_leaf(read, read, read, nothing).
select_transition_leaf(read, write, closed, close).
select_transition_leaf(write, _, closed, close).

select_transition_inner(closed, Mode, _, _, _, Mode, open).
select_transition_inner(read, read, _, _, _, read, nothing).
select_transition_inner(read, write, Node, _, _, write, Action) :-
    region_aliased(Node) ->
        Action = close_conflicting;
    Action = nothing.
select_transition_inner(write, _, Node, Next, RegionTree, write, Action) :-
    region_aliased(Node) ->
        (region_siblings(Next, Siblings),
         ((include(has_region_tree_state(RegionTree, write), Siblings, [_|_]) -> true;
           include(has_region_tree_clean(RegionTree, dirty), Siblings, [_|_])) ->
              Action = close_and_reopen;
          Action = nothing));
    Action = nothing.

perform_action(nothing, _, _, Flowgraph-RegionTree-NextVertex,
               Flowgraph-RegionTree-NextVertex).
perform_action(create, Node, _, Flowgraph0-RegionTree0-NextVertex0,
               Flowgraph-RegionTree-NextVertex) :-
    create_data_node(Node, Flowgraph0-RegionTree0-NextVertex0,
                     _, Flowgraph-RegionTree-NextVertex).
perform_action(open, Node, _, Flowgraph0-RegionTree0-NextVertex0,
               Flowgraph-RegionTree-NextVertex) :-
    Node = none ->
        (Flowgraph0-RegionTree0-NextVertex0 =
         Flowgraph-RegionTree-NextVertex);
    (create_data_node(Node, Flowgraph0-RegionTree0-NextVertex0,
                      _, Flowgraph1-RegionTree1-NextVertex1),
     create_open_node(Node, Flowgraph1-RegionTree1-NextVertex1,
                      _, Flowgraph-RegionTree-NextVertex)).
perform_action(close, Node, _,
               Flowgraph0-RegionTree0-NextVertex0,
               Flowgraph-RegionTree-NextVertex) :-
    close_region_tree(Node, Flowgraph0-RegionTree0-NextVertex0,
                      Flowgraph-RegionTree-NextVertex).
perform_action(close_and_reopen, Node, Next,
               Flowgraph0-RegionTree0-NextVertex0,
               Flowgraph-RegionTree-NextVertex) :-
    close_region_tree(Node, Flowgraph0-RegionTree0-NextVertex0,
                      Flowgraph1-RegionTree1-NextVertex1),
    perform_action(open, Node, Next, Flowgraph1-RegionTree1-NextVertex1,
                   Flowgraph-RegionTree-NextVertex).
perform_action(close_conflicting, Node, Next,
               Flowgraph0-RegionTree0-NextVertex0,
               Flowgraph-RegionTree-NextVertex) :-
    % FIXME: This is broken when the partition is aliased. You lose a
    % WAR dependence if you allow a writer in an aliased subregion
    % without closing all the way to the root.
    region_aliased(Node),
    region_siblings(Next, Siblings),
    include(has_region_tree_state(RegionTree0, read), Siblings, Reads),
    include(has_region_tree_state(RegionTree0, write), Siblings, Writes),
    union(Reads, Writes, Conflicts),
    foldl(close_region_tree, Conflicts, Flowgraph0-RegionTree0-NextVertex0,
          Flowgraph-RegionTree-NextVertex).

close_region_tree(Region, Flowgraph0-RegionTree0-NextVertex0,
                  Flowgraph-RegionTree-NextVertex) :-
    get_region_tree_state(RegionTree0, Region, State0),
    (State0 = closed ->
         (Flowgraph0-RegionTree0-NextVertex0 =
          Flowgraph-RegionTree-NextVertex);
     (findall(Child, region_parent(Child, Region), Children),
      foldl(close_region_tree, Children, Flowgraph0-RegionTree0-NextVertex0,
            Flowgraph1-RegionTree1-NextVertex1),
      include(has_valid_instance(RegionTree1), Children, Dirty),
      maplist(get_region_tree_vertex(RegionTree1), Dirty, VDirty),
      create_close_node(Region, VDirty, Flowgraph1-RegionTree1-NextVertex1,
                        _, Flowgraph-RegionTree2-NextVertex),
      foldl(put_region_tree_state(closed), Children, RegionTree2, RegionTree3),
      foldl(put_region_tree_vertex(0), Children, RegionTree3, RegionTree4),
      foldl(put_region_tree_open(0), Children, RegionTree4, RegionTree5),
      foldl(put_region_tree_clean(clean), Dirty, RegionTree5, RegionTree6),
      put_region_tree_clean(dirty, Region, RegionTree6, RegionTree))).

has_valid_instance(RegionTree, Region) :-
    get_region_tree_vertex(RegionTree, Region, Vertex),
    Vertex > 0.

mutate_region_tree(Region, write, Flowgraph0-RegionTree0-NextVertex0,
                  Vertex, Flowgraph-RegionTree-NextVertex) :-
    create_node(data, none, Region, Flowgraph0-RegionTree0-NextVertex0,
                Vertex, Flowgraph-RegionTree1-NextVertex),
    get_assoc(Region, RegionTree1, State1-_-_-_),
    put_assoc(Region, RegionTree1, State1-dirty-Vertex-0, RegionTree).

% Helpers for creating dataflow nodes
create_data_node(Node, Flowgraph0-RegionTree0-NextVertex0,
                 Vertex, Flowgraph-RegionTree-NextVertex) :-
    get_region_tree_vertex(RegionTree0, Node, Vertex0),
    (Vertex0 > 0 ->
         (Flowgraph0-RegionTree0-NextVertex0 =
          Flowgraph-RegionTree-NextVertex);
     (create_node(data, none, Node, Flowgraph0-RegionTree0-NextVertex0,
                  Vertex, Flowgraph1-RegionTree1-NextVertex1),
      ((region_parent(Node, Parent),
        get_assoc(Parent, RegionTree1, _-_-_-VOpen1),
        VOpen1 > 0) ->
           create_edge(VOpen1, Vertex, write, Flowgraph1-RegionTree1-NextVertex1,
                       Flowgraph-RegionTree2-NextVertex);
       (Flowgraph1-RegionTree1-NextVertex1 =
        Flowgraph-RegionTree2-NextVertex)),
      put_region_tree_vertex(Vertex, Node, RegionTree2, RegionTree3),
      put_region_tree_open(0, Node, RegionTree3, RegionTree))).

create_open_node(Node, Flowgraph0-RegionTree0-NextVertex0,
                 VOpen, Flowgraph-RegionTree-NextVertex) :-
    get_region_tree_vertex(RegionTree0, Node, Vertex0),
    create_node(compute, open, none, Flowgraph0-RegionTree0-NextVertex0,
                VOpen, Flowgraph1-RegionTree1-NextVertex1),
    create_edge(Vertex0, VOpen, read, Flowgraph1-RegionTree1-NextVertex1,
                Flowgraph-RegionTree2-NextVertex),
    put_region_tree_open(VOpen, Node, RegionTree2, RegionTree).

create_close_node(Node, VSources, Flowgraph0-RegionTree0-NextVertex0,
                 VClose, Flowgraph-RegionTree-NextVertex) :-
    create_node(compute, close, none, Flowgraph0-RegionTree0-NextVertex0,
                VClose, Flowgraph1-RegionTree1-NextVertex1),
    foldl(create_close_edge(VClose), VSources, Flowgraph1-RegionTree1-NextVertex1,
          Flowgraph2-RegionTree2-NextVertex2),
    get_region_tree_vertex(RegionTree2, Node, Vertex0),
    create_close_edge(VClose, Vertex0, Flowgraph2-RegionTree2-NextVertex2,
          Flowgraph3-RegionTree3-NextVertex3),
    create_node(data, none, Node, Flowgraph3-RegionTree3-NextVertex3,
                Vertex, Flowgraph4-RegionTree4-NextVertex4),
    create_edge(VClose, Vertex, write, Flowgraph4-RegionTree4-NextVertex4,
                Flowgraph-RegionTree5-NextVertex),
    put_region_tree_vertex(Vertex, Node, RegionTree5, RegionTree6),
    put_region_tree_open(0, Node, RegionTree6, RegionTree).

create_close_edge(VClose, VSource, Flowgraph0-RegionTree0-NextVertex0,
                  Flowgraph-RegionTree-NextVertex) :-
    create_edge(VSource, VClose, read, Flowgraph0-RegionTree0-NextVertex0,
                Flowgraph-RegionTree-NextVertex).

% Region tree queries
has_region_tree_state(RegionTree, State, Region) :-
    get_assoc(Region, RegionTree, State-_-_-_).
has_region_tree_clean(RegionTree, Clean, Region) :-
    get_assoc(Region, RegionTree, _-Clean-_-_).
has_region_tree_vertex(RegionTree, Vertex, Region) :-
    get_assoc(Region, RegionTree, _-_-Vertex-_).
has_region_tree_open(RegionTree, Open, Region) :-
    get_assoc(Region, RegionTree, _-_-_-Open).

get_region_tree_state(RegionTree, Region, State) :-
    get_assoc(Region, RegionTree, State-_-_-_).
get_region_tree_clean(RegionTree, Region, Clean) :-
    get_assoc(Region, RegionTree, _-Clean-_-_).
get_region_tree_vertex(RegionTree, Region, Vertex) :-
    get_assoc(Region, RegionTree, _-_-Vertex-_).
get_region_tree_open(RegionTree, Region, Open) :-
    get_assoc(Region, RegionTree, _-_-_-Open).

put_region_tree_state(State, Region0, RegionTree0, RegionTree) :-
    get_assoc(Region0, RegionTree0, _-Clean0-Vertex0-Open0),
    put_assoc(Region0, RegionTree0, State-Clean0-Vertex0-Open0, RegionTree).
put_region_tree_clean(Clean, Region0, RegionTree0, RegionTree) :-
    get_assoc(Region0, RegionTree0, State0-_-Vertex0-Open0),
    put_assoc(Region0, RegionTree0, State0-Clean-Vertex0-Open0, RegionTree).
put_region_tree_vertex(Vertex, Region0, RegionTree0, RegionTree) :-
    get_assoc(Region0, RegionTree0, State0-Clean0-_-Open0),
    put_assoc(Region0, RegionTree0, State0-Clean0-Vertex-Open0, RegionTree).
put_region_tree_open(Open, Region0, RegionTree0, RegionTree) :-
    get_assoc(Region0, RegionTree0, State0-Clean0-Vertex0-_),
    put_assoc(Region0, RegionTree0, State0-Clean0-Vertex0-Open, RegionTree).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Translation from Graphs to Programs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graph_to_program(Flowgraph, Program) :-
    flowgraph_graph(Flowgraph, Graph),
    top_sort(Graph, Vertices),
    include(has_vertex_op(Flowgraph, task), Vertices, VTasks),
    foldl(task_vertex_to_task(Flowgraph), VTasks, [], Tasks),
    reverse(Tasks, Program).

task_vertex_to_task(Flowgraph, VTask, Program, [Task|Program]) :-
    flowgraph_graph(Flowgraph, Graph),

    immediate_predecessors(VTask, Graph, VPred),
    include(has_vertex_type(Flowgraph, data), VPred, VReads),
    maplist(vertex_region(Flowgraph), VReads, Reads),

    neighbors(VTask, Graph, VDesc),
    include(has_vertex_type(Flowgraph, data), VDesc, VWrites),
    maplist(vertex_region(Flowgraph), VWrites, Writes),

    sort(Reads, ReadSet),
    sort(Writes, WriteSet),
    subtract(ReadSet, WriteSet, ReadOnlySet),

    maplist(flipped_pairs(read_only), ReadOnlySet, ReadOnly),
    maplist(flipped_pairs(read_write), WriteSet, ReadWrite),
    append(ReadOnly, ReadWrite, Task).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pretty Printer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Covert the graph Flowgraph to an atom String prettyprinted in
% GraphViz format.
prettyprint_graph(Flowgraph, String) :-
    list_to_assoc([], Map),
    prettyprint_graph(Flowgraph, 0, Map, String).

prettyprint_graph(Flowgraph, Cluster, Map, String) :-
    flowgraph_graph(Flowgraph, Graph),
    atom_number(SC, Cluster),
    vertices(Graph, V),
    maplist(prettyprint_vertex(Flowgraph, Map), V, SVs),
    edges(Graph, E),
    maplist(prettyprint_edge(Flowgraph, Map), E, SEs),
    append([['subgraph cluster', SC, ' { '], SVs, SEs, ['}\n']], Ss),
    foldl(atom_concat_back, Ss, '', String).

prettyprint_vertex(Flowgraph, Map, Vertex, SV) :-
    (get_assoc(Vertex, Map, VMap) -> true; Vertex = VMap),
    atom_number(ID, VMap),
    vertex_type(Flowgraph, Vertex, VertexType),
    (VertexType = compute ->
         (vertex_op(Flowgraph, Vertex, VertexOp),
          Label = VertexOp,
          (VertexOp = task -> Shape = 'rectangle'; Shape = 'diamond'));
     (vertex_region(Flowgraph, Vertex, VertexRegion),
      Label = VertexRegion,
      Shape = 'ellipse')),
    foldl(atom_concat_back,
          [ID, ' [ label = ', Label, ', shape = ', Shape, ' ]; '],
          '', SV).

prettyprint_edge(Flowgraph, Map, V1-V2, SE) :-
    edge_type(Flowgraph, V1-V2, ET),
    (get_assoc(V1, Map, V1Map) -> true; V1 = V1Map),
    (get_assoc(V2, Map, V2Map) -> true; V2 = V2Map),
    atom_number(SV1Map, V1Map),
    atom_number(SV2Map, V2Map),
    (ET = happens_before -> Style = 'dotted'; Style = 'solid'),
    foldl(atom_concat_back,
          [SV1Map, ' -> ', SV2Map,
           ' [ label = ', ET, ', style = ', Style, ' ]; '],
          '', SE).

atom_concat_back(S2, S1, S3) :- atom_concat(S1, S2, S3).

prettyprint_subgraph(Flowgraph, I1-S1, I2-S2) :-
    flowgraph_graph(Flowgraph, Graph),
    vertices(Graph, Vertices),
    length(Vertices, Count),
    I2 is I1 + Count,

    Start is I1,
    End is I2 - 1,
    numlist(Start, End, VNew),
    maplist(pairs, Vertices, VNew, VPairs),
    list_to_assoc(VPairs, Map),
    prettyprint_graph(Flowgraph, I1, Map, S),
    atom_concat(S1, S, S2).

prettyprint_graphs(Flowgraphs, S) :-
    foldl(prettyprint_subgraph, Flowgraphs, 0-'', _-S).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

main([]) :- main([graph]).
main([Arg0]) :- main([Arg0, '3']).
main([Arg0, Arg1]) :- main([Arg0, Arg1, '0']).
main([Arg0, Arg1, Arg2|_]) :-
    (Arg0 = graph; Arg0 = program),
    atom_number(Arg1, NArg1),
    atom_number(Arg2, NArg2),
    (Arg0 = graph ->
         (numlist(1, NArg1, Vertices),
          findall(Flowgraph,
                  valid_graph(Vertices, NArg2, Flowgraph),
                  Flowgraphs));
     findall(Flowgraph,
             (program(NArg1, NArg2, Program),
              (program_to_graph(Program, Flowgraph0) -> true; throw(Program)),
              (racefree(Flowgraph0) -> true; throw(Program)),
              (serializable(Flowgraph0, Flowgraph) -> true; throw(Program))),
             Flowgraphs)),
    writeln('digraph {'),
    writeln('rankdir = LR;'),
    writeln('node [ margin = "0.055,0.0275" ];'),
    prettyprint_graphs(Flowgraphs, S),
    writeln(S),
    writeln('}').

% Generating graphs:
% graph([1, 2, 3], 0, G).
% labeled_graph([1, 2, 3], 0, G-VT-VO-VR-ET).
% wellformed_graph([1, 2, 3], 0, G-VT-VO-VR-ET).
% racefree_graph([1, 2, 3], 0, G-VT-VO-VR-ET).
% serializable_graph([1, 2, 3], 0, G-VT-VO-VR-ET).
% valid_graph([1, 2, 3], 0, G-VT-VO-VR-ET).

% Counting generated graphs:
% aggregate_all(count, graph([1, 2, 3], 0, G), C).
% aggregate_all(count, labeled_graph([1, 2, 3], 0, F), C).
% aggregate_all(count, wellformed_graph([1, 2, 3], 0, F), C).

% Generating programs:
% program(2, 2, P), program_to_graph(P, F), serializable(F, F1), graph_to_program(F1, P1).
% valid_graph([1, 2, 3], 0, F), graph_to_program(F, P).
