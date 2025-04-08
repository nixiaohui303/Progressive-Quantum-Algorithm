#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, ZZ, RX,RZ,X,I,Measure
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer,MQLayer,MQAnsatzOnlyOps
from mindquantum.simulator import Simulator
from mindspore.common.initializer import Normal,initializer
from mindspore import Tensor,ops
from mindspore import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.nn import Adam, TrainOneStepCell                   

import networkx as nx
import mindspore.nn as nn
import mindspore as ms
import mindquantum as mq
import seaborn as sns
from math import pi


import matplotlib.pyplot as plt
import numpy as np
from math import pi
from mpl_toolkits.mplot3d import Axes3D




import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import time


import warnings
warnings.filterwarnings("ignore")

import logging
import sys
import datetime
 
def init_logger(filename, logger_name):
    '''
    @brief:
        initialize logger that redirect info to a file just in case we lost connection to the notebook
    @params:
        filename: to which file should we log all the info
        logger_name: an alias to the logger
    '''
 
    # get current timestamp
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(message)s',
#         format='%(message)s',
        handlers=[
            logging.FileHandler(filename=filename,encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
 
    # Test
    logger = logging.getLogger(logger_name)
   #logger.info('### Init. Logger {} ###'.format(logger_name))
    return logger


# Initialize
my_logger = init_logger("data/AQA/n = 14/ER/new, 3-regular/test_open.log", "ml_logger")


# In[2]:


# Compute the circuit depth

# right 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""DAG Circuit."""
import typing

from mindquantum.core import Circuit, gates
from mindquantum.utils.type_value_check import _check_input_type

# pylint: disable=invalid-name


class DAGNode:
    """
    Basic node in Directed Acyclic Graph.

    A DAG node has local index, which label the index of leg of node, and child nodes and father nodes.
    """

    def __init__(self):
        """Initialize a DAGNode object."""
        self.child: typing.Dict[int, "DAGNode"] = {}  # key: local index, value: child DAGNode
        self.father: typing.Dict[int, "DAGNode"] = {}  # key: local index, value: father DAGNode
        self.local: typing.List[int] = []

    def clean(self):
        """Clean node and set it to empty."""
        self.child = {}
        self.father = {}
        self.local = []

    def insert_after(self, other_node: "DAGNode"):
        """
        Insert other node after this dag node.

        Args:
            other_node (:class:`~.algorithm.compiler.DAGNode`): other DAG node.
        """
        _check_input_type("other_node", DAGNode, other_node)
        for local in self.local:
            if local in other_node.local:
                other_node.father[local] = self
                if local in self.child:
                    other_node.child[local] = self.child.get(local)
                    self.child.get(local).fathre[local] = other_node
                self.child[local] = other_node

    def insert_before(self, other_node: "DAGNode"):
        """
        Insert other node before this dag node.

        Args:
            other_node (:class:`~.algorithm.compiler.DAGNode`): other DAG node.
        """
        _check_input_type("other_node", DAGNode, other_node)
        for local in self.local:
            if local in other_node.local:
                other_node.child[local] = self
                if local in self.father:
                    other_node.father[local] = self.father.get(local)
                    self.father.get(local).child[local] = other_node
                self.father[local] = other_node


def connect_two_node(father_node: DAGNode, child_node: DAGNode, local_index: int):
    """
    Connect two DAG node through given local_index.

    Args:
        father_node (DAGNode): The father DAG node.
        child_node (DAGNode): The child DAG node.
        local_index (int): which leg you want to connect.
    """
    if local_index not in father_node.local or local_index not in child_node.local:
        raise ValueError(
            f"local_index {local_index} not in father_node" f" {father_node} or not in child_node {child_node}."
        )
    father_node.child[local_index] = child_node
    child_node.father[local_index] = father_node


class DAGQubitNode(DAGNode):
    """
    DAG node that work as quantum qubit.

    Args:
        qubit (int): id of qubit.
    """

    def __init__(self, qubit: int):
        """Initialize a DAGQubitNode object."""
        super().__init__()
        _check_input_type("qubit", int, qubit)
        self.qubit = qubit
        self.local = [qubit]

    def __str__(self):
        """Return a string representation of qubit node."""
        return f"q{self.qubit}"

    def __repr__(self):
        """Return a string representation of qubit node."""
        return self.__str__()


class GateNode(DAGNode):
    """
    DAG node that work as quantum gate.

    Args:
        gate (:class:`~.core.gates.BasicGate`): Quantum gate.
    """

    def __init__(self, gate: gates.BasicGate):
        """Initialize a GateNode object."""
        super().__init__()
        _check_input_type("gate", gates.BasicGate, gate)
        self.gate = gate
        self.local = gate.obj_qubits + gate.ctrl_qubits

    def __str__(self):
        """Return a string representation of gate node."""
        return str(self.gate)

    def __repr__(self):
        """Return a string representation of gate node."""
        return self.__str__()


class BarrierNode(GateNode):
    """DAG node that work as barrier."""

    def __init__(self, gate: gates.BasicGate, all_qubits: typing.List[int]):
        """Initialize a BarrierNode object."""
        super().__init__(gate)
        self.local = all_qubits


class DAGCircuit:
    """
    A Directed Acyclic Graph of a quantum circuit.

    Args:
        circuit (:class:`~.core.circuit.Circuit`): the input quantum circuit.

    Examples:
    from mindquantum.algorithm.compiler import DAGCircuit
    from mindquantum.core.circuit import Circuit
    circ = Circuit().h(0).x(1, 0)
    dag_circ = DAGCircuit(circ)
    dag_circ.head_node[0]
        q0
    dag_circ.head_node[0].child
        {0: H(0)}
    """

    def __init__(self, circuit: Circuit):
        """Initialize a DAGCircuit object."""
        _check_input_type("circuit", Circuit, circuit)
        self.head_node = {i: DAGQubitNode(i) for i in sorted(circuit.all_qubits.keys())}
        self.final_node = {i: DAGQubitNode(i) for i in sorted(circuit.all_qubits.keys())}
        for i in self.head_node:
            self.head_node[i].insert_after(self.final_node[i])
        for gate in circuit:
            if isinstance(gate, gates.BarrierGate):
                if gate.obj_qubits:
                    self.append_node(BarrierNode(gate, sorted(gate.obj_qubits)))
                else:
                    self.append_node(BarrierNode(gate, sorted(circuit.all_qubits.keys())))
            else:
                self.append_node(GateNode(gate))
        self.global_phase = gates.GlobalPhase(0)

    @staticmethod
    def replace_node_with_dag_circuit(node: DAGNode, coming: "DAGCircuit"):
        """
        Replace a node with a DAGCircuit.

        Args:
            node (:class:`~.algorithm.compiler.DAGNode`): the original DAG node.
            coming (:class:`~.algorithm.compiler.DAGCircuit`): the coming DAG circuit.

        Examples:
        from mindquantum.algorithm.compiler import DAGCircuit
        from mindquantum.core.circuit import Circuit
        circ = Circuit().x(1, 0)
        circ
            q0: ────■─────
                    ┃
                  ┏━┻━┓
            q1: ──┨╺╋╸┠───
                  ┗━━━┛
        dag_circ = DAGCircuit(circ)
        node = dag_circ.head_node[0].child[0]
        node
            X(1 <-: 0)
        sub_dag = DAGCircuit(Circuit().h(1).z(1, 0).h(1))
        DAGCircuit.replace_node_with_dag_circuit(node, sub_dag)
        dag_circ.to_circuit()
            q0: ──────────■───────────
                          ┃
                  ┏━━━┓ ┏━┻━┓ ┏━━━┓
            q1: ──┨ H ┠─┨ Z ┠─┨ H ┠───
                  ┗━━━┛ ┗━━━┛ ┗━━━┛
        """
        if set(node.local) != {head.qubit for head in coming.head_node.values()}:
            raise ValueError(f"Circuit in coming DAG is not aligned with gate in node: {node}")
        for local in node.local:
            connect_two_node(node.father[local], coming.head_node[local].child[local], local)
            connect_two_node(coming.final_node[local].father[local], node.child[local], local)

    def append_node(self, node: DAGNode):
        """
        Append a quantum gate node.

        Args:
            node (:class:`~.algorithm.compiler.DAGNode`): the DAG node you want to append.

        Examples:
        from mindquantum.algorithm.compiler import DAGCircuit, GateNode
        from mindquantum.core.circuit import Circuit
        import mindquantum.core.gates as G
        circ = Circuit().h(0).x(1, 0)
        circ
                  ┏━━━┓
            q0: ──┨ H ┠───■─────
                  ┗━━━┛   ┃
                        ┏━┻━┓
            q1: ────────┨╺╋╸┠───
                        ┗━━━┛
        dag_circ = DAGCircuit(circ)
        node = GateNode(G.RX('a').on(0, 2))
        dag_circ.append_node(node)
        dag_circ.to_circuit()
                  ┏━━━┓       ┏━━━━━━━┓
            q0: ──┨ H ┠───■───┨ RX(a) ┠───
                  ┗━━━┛   ┃   ┗━━━┳━━━┛
                        ┏━┻━┓     ┃
            q1: ────────┨╺╋╸┠─────╂───────
                        ┗━━━┛     ┃
                                  ┃
            q2: ──────────────────■───────
        """
        _check_input_type('node', DAGNode, node)
        for local in node.local:
            if local not in self.head_node:
                self.head_node[local] = DAGQubitNode(local)
                self.final_node[local] = DAGQubitNode(local)
                self.head_node[local].insert_after(self.final_node[local])
            self.final_node[local].insert_before(node)

    def depth(self) -> int:
        """
        Return the depth of quantum circuit.

        Examples:
        from mindquantum.core.circuit import Circuit
        from mindquantum.algorithm.compiler import DAGCircuit
        circ = Circuit().h(0).h(1).x(1, 0)
        circ
                  ┏━━━┓
            q0: ──┨ H ┠───■─────
                  ┗━━━┛   ┃
                  ┏━━━┓ ┏━┻━┓
            q1: ──┨ H ┠─┨╺╋╸┠───
                  ┗━━━┛ ┗━━━┛
        DAGCircuit(circ).depth()
            2
        """
        return len(self.layering())

    def find_all_gate_node(self) -> typing.List[GateNode]:
        """
        Find all gate node in this :class:`~.algorithm.compiler.DAGCircuit`.

        Returns:
            List[:class:`~.algorithm.compiler.GateNode`], a list of all :class:`~.algorithm.compiler.GateNode`
            of this :class:`~.algorithm.compiler.DAGCircuit`.

        Examples:
        from mindquantum.algorithm.compiler import DAGCircuit
        from mindquantum.core.circuit import Circuit
        circ = Circuit().h(0).x(1, 0)
        dag_circ = DAGCircuit(circ)
        dag_circ.find_all_gate_node()
            [H(0), X(1 <-: 0)]
        """
        found = set(self.head_node.values())

        def _find(current_node: DAGNode, found):
            if current_node not in found:
                found.add(current_node)
                for node in current_node.father.values():
                    _find(node, found)
                for node in current_node.child.values():
                    _find(node, found)

        for head_node in self.head_node.values():
            for current_node in head_node.child.values():
                _find(current_node, found)
        return [i for i in found if not isinstance(i, DAGQubitNode)]

    def layering(self) -> typing.List[Circuit]:
        r"""
        Layering the quantum circuit.

        Returns:
            List[:class:`~.core.circuit.Circuit`], a list of layered quantum circuit.

        Examples:
        from mindquantum.algorithm.compiler import DAGCircuit
        from mindquantum.utils import random_circuit
        circ = random_circuit(3, 5, seed=42)
        circ
                  ┏━━━━━━━━━━━━━┓   ┏━━━━━━━━━━━━━┓
            q0: ──┨             ┠─╳─┨ RY(-6.1944) ┠───────────────────
                  ┃             ┃ ┃ ┗━━━━━━┳━━━━━━┛
                  ┃ Rxx(1.2171) ┃ ┃        ┃        ┏━━━━━━━━━━━━━┓
            q1: ──┨             ┠─┃────────╂────────┨             ┠───
                  ┗━━━━━━━━━━━━━┛ ┃        ┃        ┃             ┃
                  ┏━━━━━━━━━━━━┓  ┃        ┃        ┃ Rzz(-0.552) ┃
            q2: ──┨ PS(2.6147) ┠──╳────────■────────┨             ┠───
                  ┗━━━━━━━━━━━━┛                    ┗━━━━━━━━━━━━━┛
        dag_circ = DAGCircuit(circ)
        for idx, c in enumerate(dag_circ.layering()):
            ...     print(f"layer {idx}:")
            ...     print(c)
            layer 0:
                  ┏━━━━━━━━━━━━━┓
            q0: ──┨             ┠───
                  ┃             ┃
                  ┃ Rxx(1.2171) ┃
            q1: ──┨             ┠───
                  ┗━━━━━━━━━━━━━┛
                  ┏━━━━━━━━━━━━┓
            q2: ──┨ PS(2.6147) ┠────
                  ┗━━━━━━━━━━━━┛
            layer 1:
            q0: ──╳───
                  ┃
                  ┃
            q2: ──╳───
            layer 2:
                  ┏━━━━━━━━━━━━━┓
            q0: ──┨ RY(-6.1944) ┠───
                  ┗━━━━━━┳━━━━━━┛
                         ┃
            q2: ─────────■──────────
            layer 3:
                  ┏━━━━━━━━━━━━━┓
            q1: ──┨             ┠───
                  ┃             ┃
                  ┃ Rzz(-0.552) ┃
            q2: ──┨             ┠───
                  ┗━━━━━━━━━━━━━┛
        """

        def _layering(current_node: GateNode, depth_map):
            """Layering the quantum circuit."""
            if current_node.father:
                prev_depth = []
                for father_node in current_node.father.values():
                    if father_node not in depth_map:
                        _layering(father_node, depth_map)
                    prev_depth.append(depth_map[father_node])
                depth_map[current_node] = max(prev_depth) + 1
            for child in current_node.child.values():
                if not isinstance(child, DAGQubitNode):
                    if child not in depth_map:
                        _layering(child, depth_map)

        depth_map = {i: 0 for i in self.head_node.values()}
        for current_node in self.head_node.values():
            _layering(current_node, depth_map)
        layer = [Circuit() for _ in range(len(set(depth_map.values())) - 1)]
        for k, v in depth_map.items():
            if v != 0:
                if not isinstance(k, BarrierNode):
                    layer[v - 1] += k.gate
        return [c for c in layer if len(c) != 0]

    def to_circuit(self) -> Circuit:
        """
        Convert :class:`~.algorithm.compiler.DAGCircuit` to quantum circuit.

        Returns:
            :class:`~.core.circuit.Circuit`, the quantum circuit of this DAG.

        Examples:
        from mindquantum.core.circuit import Circuit
        from mindquantum.algorithm.compiler import DAGCircuit
        circ = Circuit().h(0).h(1).x(1, 0)
        circ
                  ┏━━━┓
            q0: ──┨ H ┠───■─────
                  ┗━━━┛   ┃
                  ┏━━━┓ ┏━┻━┓
            q1: ──┨ H ┠─┨╺╋╸┠───
                  ┗━━━┛ ┗━━━┛
        dag_circ = DAGCircuit(circ)
        dag_circ.to_circuit()
                  ┏━━━┓
            q0: ──┨ H ┠───■─────
                  ┗━━━┛   ┃
                  ┏━━━┓ ┏━┻━┓
            q1: ──┨ H ┠─┨╺╋╸┠───
                  ┗━━━┛ ┗━━━┛
        """
        circuit = Circuit()
        considered_node = set(self.head_node.values())

        def adding_current_node(current_node, circuit, considered):
            if all(i in considered for i in current_node.father.values()) and not isinstance(
                current_node, DAGQubitNode
            ):
                circuit += current_node.gate
                considered.add(current_node)
            else:
                for node in current_node.father.values():
                    if node not in considered:
                        adding_current_node(node, circuit, considered)
                for node in current_node.child.values():
                    if node not in considered:
                        adding_current_node(node, circuit, considered)

        for current_node in self.final_node.values():
            adding_current_node(current_node, circuit, considered_node)
        return circuit


# In[3]:


# Function to retrieve the neighboring nodes information for each node in graph g
def get_info_neighbors(g):
    info = {}  # Dictionary to store each node and its corresponding neighbors
    
    n = len(g.nodes())  # Get the number of nodes in the current subgraph
    # Loop through each node in the graph
    for k in g.nodes():
        neighbors = []  # List to store the neighbors of node k
        # Loop through each edge in the graph
        for u, v in g.edges:
            # If node k is the second node in the edge, append the first node as neighbor
            if v == k:
                neighbors.append(u)
            # If node k is the first node in the edge, append the second node as neighbor
            if u == k:
                neighbors.append(v)
        # Store node k and its neighbors in the dictionary 'info'
        info[k] = neighbors
#     my_logger.info('The information of nodes and their neighboring nodes is: {}'.format(info))
    return info





# Find the nodes with the minimum degree
def find_min_degree(target_graph):
    info = get_info_neighbors(target_graph)
    
    d = [] # Store the degree of each node
    for key,value in info.items():
        d.append(len(value))
    min_d = min(d)
    
    # Find the nodes with the minimum degree
    nodes = []
    for key,value in info.items():
        if len(value) == min_d:
            nodes.append(key)
#     my_logger.info('min_d = {}, nodes = {}'.format(min_d,nodes))
    return nodes


# In[4]:


# Create a new graph by adding one new node and its associated edges to the old graph.
# Required parameters:
#   old_graph: the existing subgraph,
#   new_nodes: a list of new nodes to be added,
#   target_graph: the target graph from which the connectivity (degree) information is obtained.
def create_new_graph(old_graph, new_nodes, target_graph):
    new_E = []  # List to store the edges of the new graph.
    new_V = []  # List to store the vertices of the new graph.
    
    # Add all nodes from the old graph to new_V.
    for node in old_graph.nodes():
        new_V.append(node)
    
    # Add the new nodes and their associated edges.
    for i in range(len(new_nodes)):
        new_V.append(new_nodes[i])
    new_V.sort()  # Sort the vertices in ascending order.
    
    # Iterate over each edge in the target graph.
    for edge in target_graph.edges():
        u0 = edge[0]
        v0 = edge[1]  # The two vertices of the current edge.
        
        # If both endpoints of the edge exist in new_V, add the edge to new_E.
        if u0 in new_V and v0 in new_V:
            new_E.append(edge)
    
    my_logger.info('new_E = {}, new_V = {}'.format(new_E, new_V))
    
    # Create a new graph.
    new_graph = nx.Graph()
    new_graph.add_nodes_from(new_V)
    new_graph.add_edges_from(new_E)
    
    # Draw the generated graph.
    pos = nx.circular_layout(new_graph)
    options = {
        "with_labels": True,
        "font_size": 16,
        "font_weight": "bold",
        "font_color": "white",
        "node_size": 1000,
        "width": 2
    }
    nx.draw_networkx(new_graph, pos, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
    
    return new_graph, new_E, new_V


# In[5]:


# Calculate m0 for candidate nodes, where m0 is the number of edges connecting a candidate node 'a'
# with the nodes already present in the current subgraph.

def calculate_m0(candidate_nodes, subgraph_nodes, target_graph):
    m0_edges = []  # List to store the m0 value corresponding to each candidate node.
    
    # For each candidate node, compute the count of edges (m0) that connect it to nodes in the subgraph.
    for j in range(0, len(candidate_nodes)):
        a = candidate_nodes[j]
        m0 = 0
        # Iterate through each edge in the target graph.
        for u, v in target_graph.edges():
            # Check if candidate node 'a' is incident to the edge.
            if u == a or v == a:
                # If one of the endpoints belongs to the current subgraph, count this edge.
                if u in subgraph_nodes or v in subgraph_nodes:
                    m0 = m0 + 1
        m0_edges.append(m0)
        
    # Determine the minimum m0 value among the candidate nodes.
    min_m0 = min(m0_edges)
    
    # Collect all candidate nodes whose m0 equals the minimum value.
    min_a = []
    for j in range(0, len(m0_edges)):
        if m0_edges[j] == min_m0:
            min_a.append(candidate_nodes[j])
    
    return min_m0, min_a


# In[6]:


def find_new_node(old_graph, target_graph):
    """
    Finds a new node to add to the current subgraph (old_graph) based on the connection criterion.
    
    
    Inputs:
        old_graph:  A NetworkX graph representing the current subgraph.
        target_graph: A NetworkX graph representing the full graph from which a new node is to be selected.
    
    Outputs:
        - new_add_node: the selected node from target_graph to add to old_graph.
        - min_m0: the m0 value (number of edges connecting the candidate node to the subgraph)
                    corresponding to the chosen new node.
    """

    # Retrieve the nodes of the current subgraph (old_graph)
    subgraph_nodes = []
    for node in old_graph.nodes():
        subgraph_nodes.append(node)
    
    # Construct the candidate set: nodes in the target graph not currently in the subgraph.
    # For example, if target_graph has nodes [0, 1, 2] and subgraph_nodes = [0],
    # then candidate_nodes will be [1, 2].
    candidate_nodes = []
    for i in target_graph.nodes():
        if i not in subgraph_nodes:
            candidate_nodes.append(i)
    # my_logger.info('subgraph_nodes = {}, candidate_nodes = {}'.format(subgraph_nodes, candidate_nodes))
    
    # Calculate, for each candidate node, the m0 value (the number of edges connecting it
    # with any node in the current subgraph) using the calculate_m0 function.
    # min_m0 is the minimum m0 value and min_a is the list of candidate nodes achieving that m0.
    min_m0, min_a = calculate_m0(candidate_nodes, subgraph_nodes, target_graph)
    
    # If only one candidate node achieves the minimum m0, choose that node.
    if len(min_a) == 1:
        new_add_node = min_a[0]
    
    # If multiple candidate nodes have the same min_m0, perform a further analysis.
    else:
        # my_logger.info('Multiple candidate nodes have min_m0 = {}, details: min_a = {}'.format(min_m0, min_a))
        next_m0 = []  # To store the subsequent m0 values when each candidate from min_a is added to the subgraph.
        
        # For each candidate node in min_a, simulate adding it to the subgraph and
        # compute the resulting m0 (next_min_m0) for the new candidate set.
        for i0 in range(0, len(min_a)):
            # Reset subgraph_nodes to the original nodes in old_graph
            subgraph_nodes = []
            for node in old_graph.nodes():
                subgraph_nodes.append(node)
            
            # Append the candidate node from min_a to the subgraph
            subgraph_nodes.append(min_a[i0])
            
            # Construct a new candidate set: nodes in target_graph not in the updated subgraph.
            candidate_nodes = []
            for i in target_graph.nodes():
                if i not in subgraph_nodes:
                    candidate_nodes.append(i)
            
            # Calculate the m0 for the updated candidate set with the new subgraph.
            min_m0, next_min_a = calculate_m0(candidate_nodes, subgraph_nodes, target_graph)
            # print('next_min_m0 = {}, next_min_a = {}'.format(min_m0, next_min_a))
            next_m0.append(min_m0)
        
        # Determine the minimum next-step m0 among the simulated choices.
        min_next_m0 = min(next_m0)
        # my_logger.info('For candidates in min_a = {} the corresponding next m0 values are {}'.format(min_a, next_m0))
        
        # Store candidate nodes that achieve the minimum next-step m0.
        s = []
        for index in range(0, len(next_m0)):
            if next_m0[index] == min_next_m0:
                s.append(min_a[index])
        # my_logger.info('Candidates satisfying both criteria (min_m0 and next m0) are: s = {}'.format(s))
        
        # If there is only one candidate satisfying both criteria, select it.
        if len(s) == 1:
            new_add_node = s[0]
        else:
            # If multiple candidates remain, select one at random.
            index0 = random.randint(0, len(s) - 1)
            new_add_node = s[index0]
            
    return new_add_node, min_m0


# In[7]:


def construct_graph(old_graph, target_graph):
    """
    Incrementally constructs a new graph by adding one vertex at a time from the target_graph 
    to the current subgraph (old_graph).
    
    
    Inputs:
        old_graph:   The current subgraph (a NetworkX graph) that is being incrementally constructed.
        target_graph: The target graph (a NetworkX graph) from which vertices and edge connectivity
                      information are derived.
    
    Outputs:
        Returns new_graph, the updated graph after the addition of one new vertex. 
    """
    # Get the number of nodes in the current graph (old_graph)
    old_num_nodes = old_graph.number_of_nodes()  
    # Get the total number of nodes in the target graph
    target_num_nodes = target_graph.number_of_nodes()  
    new_nodes = []  # Initialize list to store the new node(s) to be added

    # Check if the current graph already equals the target graph
    if old_num_nodes == target_num_nodes:
        my_logger.info('The current graph is the target graph.')
    else:
        # Obtain the next node to add (based on connection criterion) 
        new_add_node, min_m0 = find_new_node(old_graph, target_graph)
        new_nodes.append(new_add_node)
    
    # Construct a new graph by adding the new nodes (and associated edges) to the old graph
    new_graph, new_E, new_V = create_new_graph(old_graph, new_nodes, target_graph)
    return new_graph


# In[8]:


# Build the initial subgraph from the target graph.
def build_initial_graph(t):
    """
    Constructs an initial subgraph by selecting one or more nodes from the target graph.
    The process starts by choosing a node with the minimum degree as the starting point,
    and then incrementally adds nodes to the subgraph based on criteria designed to minimize
    the connection (edge count) between the new node and the existing subgraph. 
    
    Inputs:
        t: The desired number of nodes for the initial subgraph.
    
    Outputs:
        V0: A sorted list of nodes in the constructed initial subgraph.
        E0: The edges of the constructed initial subgraph.
    """

    # Check if the number of nodes in the target graph is not equal to t.
    if t != len(V):
        # Find the node(s) in the target graph with the smallest degree to use as the initial node.
        nodes = find_min_degree(target_graph)
        # If there is only one node with the minimum degree, choose it.
        if len(nodes) == 1:
            first_node = nodes[0]
        else:
            # If there are multiple nodes with the same minimum degree, further select the one that,
            # when added, results in the fewest additional edges (lowest m0 value) in subsequent steps.
            min_m = []  # To store the next-step m0 values corresponding to each candidate as the first node.
            for node in range(0, len(nodes)):
                old_V = []
                old_E = []
                # Create a temporary subgraph containing the candidate node.
                old_V.append(nodes[node]) 
                old_graph = nx.Graph()
                old_graph.add_nodes_from(old_V)
                old_graph.add_edges_from(old_E)
                # Calculate the m0 value if the candidate node is added.
                new_add_node, min_m0 = find_new_node(old_graph, target_graph)
                # my_logger.info('If node = {} is chosen as the first node, then after two steps min_m0 = {}'.format(nodes[node], min_m0))
                # my_logger.info('\n')
                min_m.append(min_m0)

            # From the candidate nodes with minimum degree, select those that yield the smallest m0.
            s = []  
            for index in range(0, len(min_m)):
                if min_m[index] == min(min_m):
                    s.append(nodes[index])  # Candidate nodes satisfying both the minimum degree and minimum m0 criteria.
            # my_logger.info('The candidate nodes satisfying minimum degree and minimum m0 criteria are: s = {}'.format(s))
            # my_logger.info('\n')

            # If only one candidate remains, select it; otherwise, randomly choose one.
            if len(s) == 1:
                first_node = s[0]
            else:
                index0 = random.randint(0, len(s) - 1)
                first_node = s[index0]

        # With the selected first node, build the initial subgraph.
        # The chosen first node satisfies:
        #   (1) It has the minimum degree in the target graph.
        #   (2) Its addition minimizes the number of edges in further steps of expanding the subgraph.
        old_V = []
        old_E = []
        old_V.append(first_node)
        old_graph = nx.Graph()
        old_graph.add_nodes_from(old_V)
        old_graph.add_edges_from(old_E)  

        t0 = len(old_V)  # The current number of nodes in the subgraph.

        # Incrementally add nodes until the subgraph reaches the desired size t.
        for i in range(0, t - t0):
            new_graph = construct_graph(old_graph, target_graph)
            old_graph = new_graph
    else:
        # If the desired number of nodes equals the number of nodes in the target graph,
        # the subgraph is simply the target graph.
        old_graph = nx.Graph()
        old_graph = target_graph
    
    # Get the list of nodes from the final subgraph and sort them.
    V0 = list(old_graph.nodes())
    V0.sort()  # Ensure nodes are sorted in ascending order.
    
    # Retrieve the edges of the final subgraph.
    E0 = old_graph.edges()
    
    return V0, E0


# In[9]:


# Create initial state
def create_encoder():
    encoder = Circuit()
    return encoder



# Build the circuit U_HD corresponding to the target Hamiltonian H_D, with parameter gamma.
def build_U_HD(layer, g, target_graph):
    """
    Constructs the U_HD circuit for the target Hamiltonian with a given gamma parameter at a specified layer.

    Function:
        For each node in the provided subgraph g, an RZ rotation gate is applied with a parameter
        named 'gamma{layer}'. A barrier is added after all such gates.

    Inputs:
        layer: An integer specifying the current QAOA layer.
        g: A NetworkX graph or an object that has a property 'nodes' representing the nodes of the subgraph.
        target_graph: The target graph from which the Hamiltonian is derived. (Not used directly in this function)

    Outputs:
        Returns a tuple (cir_HD, RZ_gates) where:
            - cir_HD: The constructed circuit (of type Circuit) with RZ gates applied to all nodes.
            - RZ_gates: The number of RZ gates used.
    """
    RZ_gates = 0  # Counter for the number of RZ gates used.
    
    # Initialize an empty quantum circuit.
    cir_HD = Circuit()
    
    # Apply an RZ gate with parameter 'gamma{layer}' on each node in the graph g.
    for v in g.nodes:
        cir_HD += RZ('gamma{}'.format(layer)).on(v)
        RZ_gates += 1
        
    # Add a barrier for circuit separation.
    cir_HD.barrier()
    
    return cir_HD, RZ_gates


# Build the circuit U_HM corresponding to the mixing Hamiltonian H_M, with parameter beta.
def build_U_HM(layer, info, g, target_graph):
    """
    Constructs the U_HM circuit for the mixing Hamiltonian with a given beta parameter at a specified layer.

    Function:
        This function uses the neighboring node information provided in 'info' to build a multi-qubit controlled 
        RX gate for each node with at least one neighbor. The gate is applied conditionally: only when the states of 
        all the neighboring nodes (controlled by applying X gates beforehand) are 1. If a node has no neighbors (an isolated node),
        a standard RX gate is applied directly.
    
    Inputs:
        layer: An integer specifying the current QAOA layer.
        info: A dictionary containing neighboring node information for each node (obtained from get_info_neighbors).
        g: A NetworkX graph or similar object representing the current subgraph.
        target_graph: The target graph from which the Hamiltonian information is derived.

    Outputs:
        Returns a tuple (cir_HM, multi_gates, RX_gates) where:
            - cir_HM: The constructed circuit (of type Circuit) for the mixing Hamiltonian.
            - multi_gates: The number of multi-qubit controlled RX gates used.
            - RX_gates: The number of single-qubit RX gates used.
    """
    n = len(target_graph.nodes())
    
    # Counters for the number of multi-qubit controlled gates and single-qubit RX gates.
    multi_gates = 0
    RX_gates = 0
    
    # Initialize an empty quantum circuit for n qubits.
    # Note: The underlying quantum framework (mindquantum) allows directly implementing controlled single-qubit rotations without ancillary qubits.
    cir_HM = Circuit()
    
    # For each node (key) and its neighbor information (value) in the info dictionary:
    for key, value in info.items():
        if len(value) != 0:
            # Flip the state of neighboring qubits using X gates.
            for i in range(0, len(value)):
                cir_HM += X.on(value[i])
            
            # Apply a multi-qubit controlled RX gate on 'key' with parameter 'beta{layer}'.
            # The RX gate is executed only if all the flipped neighboring qubits are in the |1> state.
            cir_HM += RX('beta{}'.format(layer)).on(key, value)
            multi_gates += 1
            
            # Revert the X gate flips on the neighboring qubits.
            for i in range(0, len(value)):
                cir_HM += X.on(value[i])
            
            # Add a barrier after processing this node.
            cir_HM.barrier()
        else:
            # For an isolated node (no neighbors), apply a simple RX gate with parameter 'beta{layer}'.
            cir_HM += RX('beta{}'.format(layer)).on(key)
            cir_HM.barrier()
            RX_gates += 1
              
    return cir_HM, multi_gates, RX_gates


# Build a P-layer QAOA ansatz.
def build_ansatz(p, g, target_graph):
    """
    Constructs a QAOA ansatz with p layers.

    Function:
        This function first prepares an initial feasible state (encoder) and retrieves neighboring node information.
        Then, for each layer from 1 to p, it constructs the target Hamiltonian circuit U_HD (with RZ gates) 
        and the mixing Hamiltonian circuit U_HM (with RX gates and multi-qubit controlled rotations) and
        appends them to the overall ansatz. It also keeps track of the total number of RX, RZ, and multi-qubit gates used.
    
    Inputs:
        p: An integer representing the number of QAOA layers.
        g: The current subgraph (used for constructing the circuits).
        target_graph: The target graph from which Hamiltonian information is derived.
    
    Outputs:
        Returns a tuple (encoder, ansatz, RX_gates, RZ_gates, multi_gates) where:
            - encoder: The initial state preparation circuit.
            - ansatz: The full QAOA circuit ansatz with p layers.
            - RX_gates: Total count of RX gates used.
            - RZ_gates: Total count of RZ gates used.
            - multi_gates: Total count of multi-qubit controlled gates used.
    """
    print(g.nodes())
    # Prepare the initial encoder state; any feasible state is acceptable.
    encoder = create_encoder()
    
    # Retrieve the neighboring node information for constructing the circuits.
    info = get_info_neighbors(g)
    
    # Counters for gates.
    RX_gates = 0
    RZ_gates = 0
    multi_gates = 0
    
    # Initialize an empty circuit for the QAOA ansatz.
    ansatz = Circuit()
    
    # Build the QAOA circuit layer by layer.
    for layer in range(1, p + 1):
        # Construct the target Hamiltonian circuit for the current layer.
        circ_HD, RZ_gates0 = build_U_HD(layer, g, target_graph)
        # Construct the mixing Hamiltonian circuit for the current layer.
        circ_HM, multi_gates0, RX_gates0 = build_U_HM(layer, info, g, target_graph)
        ansatz += circ_HD
        ansatz += circ_HM
        
        # Accumulate the gate counts.
        RX_gates += RX_gates0
        RZ_gates += RZ_gates0
        multi_gates += multi_gates0
        
        # Add a barrier after each layer.
        ansatz.barrier()

    return encoder, ansatz, RX_gates, RZ_gates, multi_gates






class MQAnsatzOnlyLayer(nn.Cell):
    def __init__(self, expectation_with_grad,params, weight='normal'):
        """Initialize a MQAnsatzOnlyLayer object."""
        super().__init__()
        self.evolution = MQAnsatzOnlyOps(expectation_with_grad)
        weight_size = len(self.evolution.expectation_with_grad.ansatz_params_name)
#         print('weight_size = {} '.format(weight_size))

        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get f{weight.shape}.")
        
        
        self.weight =  Parameter(params.astype(np.float32), name='ansatz_weight')
        my_logger.info('weight = {}'.format(self.weight.asnumpy()))
        

    def construct(self):
        """Construct a MQAnsatzOnlyLayer node."""
        return self.evolution(self.weight)


# In[10]:


def calculate_initial_expectation_value(g, layer, beta, gamma):
    """
    Calculates the expectation value of the target Hamiltonian for the constructed QAOA circuit,
    given the parameters beta and gamma.


    Inputs:
        g: The current subgraph (NetworkX graph) on which the Hamiltonian is defined.
        layer: The current QAOA layer count (or number of layers) to build the ansatz.
        beta: A list of beta parameters used in constructing the mixing Hamiltonian part of the ansatz.
        gamma: A list of gamma parameters used in constructing the target Hamiltonian part of the ansatz.

    Outputs:
        Returns the negative real part of the expectation value of the Hamiltonian (a real number).
    
    Parameter Details:
        - beta and gamma: The variational parameters for QAOA; these are interleaved into one parameter list.
        - g: The graph structure which determines the Hamiltonian.
        - layer: Determines the depth of the QAOA ansatz circuit.
    """
    # Combine parameters: store gamma parameters first, followed by beta parameters.
    params = []
    for i in range(len(beta)):
        params.append(gamma[i])
        params.append(beta[i])
        
    # Build the QAOA ansatz circuit using the specified number of layers, the current subgraph g,
    # and the target graph (assumed to be defined globally as target_graph).
    encoder, ansatz, RX_gates, RZ_gates, multi_gates = build_ansatz(layer, g, target_graph)
    # Concatenate the encoder circuit with the QAOA ansatz circuit.
    circ = encoder + ansatz
    circ.as_ansatz()  # Mark the circuit as an ansatz (if required by the framework).
    
    # Initialize the simulator; here using 'mqvector' backend for simulating the circuit,
    # and using the number of qubits in 'circ'.
    sim = Simulator('mqvector', circ.n_qubits)
    
    # Build the Hamiltonian corresponding to graph g.
    ham = build_ham(g)
    print(ham)
    
    # Create a parameter resolver: a dictionary that maps each parameter name in the circuit
    # to its corresponding value from the params list.
    pr = dict(zip(circ.params_name, params))
    
    # Apply the circuit on the simulator with the resolved parameters.
    sim.apply_circuit(circ, pr=pr)
    # Optionally, one could print the quantum state using:
    # print(sim.get_qs(True))
    
    # Compute the expectation value of the Hamiltonian.
    expectation = sim.get_expectation(ham)
    
    # Return the negative real part of the expectation value.
    return -1 * (expectation.real)


# In[11]:


def execute_function(g, layer, beta, gamma):
    """
    Execute the optimization of the QAOA circuit by iteratively updating the variational parameters
    until convergence based on a convergence error criterion.


    Inputs:
        g:           The current subgraph (a NetworkX graph or similar) used to build the Hamiltonian.
        layer:       An integer representing the number of QAOA layers (depth) to construct the ansatz circuit.
        beta:        A list of beta parameters for the mixing Hamiltonian part.
        gamma:       A list of gamma parameters for the target Hamiltonian part.

    Outputs:
        Returns a tuple containing:
            - result:         The measurement result from sampling the optimized circuit (after adding measurements).
            - gamma_opt:      The optimized gamma parameters (list).
            - beta_opt:       The optimized beta parameters (list).
            - final_loss:     The final (negative) expectation value of the Hamiltonian after optimization.
            - loss0:          A list storing the evolution of the loss (expectation value) during training.
            - qubits0:        A list containing the number of qubits used (which equals the number of nodes in g).
            - circuit_depth:  A list containing the circuit depth (estimated by DAGCircuit for shallow circuits).
            - quantum_gates:  A list of lists recording counts of [RX_gates, RZ_gates, multi-qubit controlled gates] used.
    """
    
    # Set the learning rate and maximum iteration count. ITR is the maximum number of iterations;
    # if convergence error is not reached within ITR iterations, optimization stops.
    lr = 0.05
    ITR = 600
    
    # Combine parameters: store gamma parameters first, then beta parameters.
    params = []
    for i in range(0, len(beta)):
        params.append(gamma[i])
        params.append(beta[i])
    
    # Initialize lists to record resource metrics:
    quantum_gates = []  # List to store the count of quantum gates used.
    circuit_depth = []  # List to store the estimated circuit depth.
    qubits0 = []        # List to store the number of qubits used.
    
    # Set MindSpore context to PYNATIVE_MODE, targeting CPU.
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    
    # Build the QAOA circuit:
    # First, create the QAOA ansatz by calling build_ansatz.
    # (Note: The commented code for generating a uniform superposition state via H gates is not used.)
    encoder, ansatz, RX_gates, RZ_gates, multi_gates = build_ansatz(layer, g, target_graph)
    # Concatenate the encoder (initial state preparation) circuit with the ansatz circuit.
    circ = encoder + ansatz
    print(circ.summary())
    # Optionally, one can export the circuit diagram as an SVG file.
    # circ.svg().to_file(filename='Circuit_{}.svg'.format(len(g.nodes())))
    
    # Record resource metrics:
    qubits0.append(len(list(g.nodes())))  # The number of qubits equals the number of nodes in g.
    if layer <= 6:
        # If the circuit is shallow enough, estimate its depth using DAGCircuit.
        circuit_depth.append(DAGCircuit(circ).depth())
    else:
        circuit_depth.append(1000)  # Use 1000 as a placeholder for "depth cannot be estimated" for deep circuits.
    quantum_gates.append([RX_gates, RZ_gates, multi_gates])
    
    # Convert the parameters list to a NumPy array.
    params = np.array(params)
    
    # Create a simulator using the 'mqvector' backend for the number of qubits in circ.
    sim = Simulator('mqvector', circ.n_qubits)
    # Build the Hamiltonian corresponding to graph g.
    ham = build_ham(g)
    
    # Calculate the initial expectation value of the Hamiltonian for the given parameters.
    initial_expectation_value = calculate_initial_expectation_value(g, layer, beta, gamma)
    my_logger.info('initial_ex = {}'.format(initial_expectation_value))
    
    # Obtain the expectation value and gradient operators for the Hamiltonian and circuit.
    # The 'parallel_worker' parameter specifies the number of parallel workers to use.
    grad_ops = sim.get_expectation_with_grad(ham, circ, parallel_worker=2)
    # Wrap the gradient operators and initial parameters into the QuantumNet model.
    QuantumNet = MQAnsatzOnlyLayer((grad_ops), params)
    
    # Set up the Adam optimizer for the trainable parameters of QuantumNet with the specified learning rate.
    opti = Adam(QuantumNet.trainable_params(), learning_rate=lr)
    # Create a training cell that performs one optimization step.
    train_net = nn.TrainOneStepCell(QuantumNet, opti)
    
    loss0 = []  # List to store the loss (expectation value) during the training iterations.
    for i in range(0, ITR + 1):
        # Execute one training step; train_net() returns a Tensor containing the updated loss value.
        loss = train_net().asnumpy()[0]
        loss0.append(loss)
        
        if i >= 2:
            l = len(loss0)
            delta1 = abs(loss0[l - 1] - loss0[l - 2])
            delta2 = abs(loss0[l - 2] - loss0[l - 3])
            # Check convergence: if the change in loss over two successive steps is below 0.001,
            # consider the optimization as converged.
            if delta1 <= 0.001 and delta2 <= 0.001:
                my_logger.info('Convergence reached after {} iterations, with loss evolution: {}'.format(len(loss0), loss0))
                break
            else:
                # Every 50 iterations, log the current training step and loss value.
                if i % 50 == 0:
                    my_logger.info("train_step = {}, loss = {}".format(i, round(loss, 5)))
    
    # Retrieve the optimized parameters from the model.
    beta_opt = []
    gamma_opt = []
    params = []
    
    # Get the circuit parameters from QuantumNet's weights, mapping the parameter names to their values.
    pr = dict(zip(ansatz.params_name, QuantumNet.weight.asnumpy()))
    for key, value in pr.items():
        params.append(value)
    my_logger.info('Optimized circuit parameters: params = {}'.format(params))
    
    # Separate the combined parameter list into gamma_opt and beta_opt.
    for i in range(0, len(params)):
        if i % 2 == 0:
            gamma_opt.append(params[i])
        else:
            beta_opt.append(params[i])
       
    # Append measurement operations to each qubit (node) in the graph.
    for i in g.nodes():
        circ += Measure('q_{}'.format(i)).on(i)  # Apply a measurement on qubit corresponding to node i.
    
    # Optionally, one can export the final circuit diagram as an SVG file.
    # circ.svg().to_file(filename='Circuit{}.svg'.format(len(g.nodes())))
    
    # Sample the circuit with the optimized parameters, taking 1000 shots.
    result = sim.sampling(circ, pr=pr, shots=1000)
    
    # Return the sampling result, the optimized gamma and beta parameters,
    # the final loss (converted to positive by multiplying by -1), the loss evolution list,
    # and the resource metrics: number of qubits, circuit depth, and gate counts.
    return result, gamma_opt, beta_opt, -round(loss, 5), loss0, qubits0, circuit_depth, quantum_gates


# In[12]:


def global_training(ham,g,p,SEED):
    minval = Tensor(0, ms.float32)
    maxval = Tensor(np.pi, ms.float32)
    shape = tuple([1])
    
    # 存储初始参数的列表
    initial_beta = []
    initial_gamma = []

    # 随机初始化初始参数
    for i in range(0,p) :
        param = ops.uniform(shape,minval,maxval,seed= SEED,dtype=ms.float32)
        initial_beta.append(param.asnumpy()[0])

        param = ops.uniform(shape,minval,maxval,seed= SEED,dtype=ms.float32)
        initial_gamma.append(param.asnumpy()[0])
    my_logger.info('SEED = {},initial_beta = {},initial_gamma = {}'.format(SEED,initial_beta,initial_gamma))
        
    # Parameter optimization
    # Get the optimized parameters
    result,gamma_opt,beta_opt,loss,loss0,qubits0,circuit_depth,quantum_gates = execute_function(g,p,initial_beta,initial_gamma)
    return result,gamma_opt,beta_opt,loss,loss0


# In[13]:


# Construct the target Hamiltonian
def build_ham(g):
    ham = QubitOperator()
    for i in g.nodes:
        ham += QubitOperator(f'Z{i}',0.5)
        ham += QubitOperator(f'Z{i} Z{i}',-0.5) # I operator
    ham = Hamiltonian(ham)
    return ham




def search_optimized_parameters(g, p, counts, SEED):
    """
    Searches for the optimized parameters under a fixed circuit depth by performing multiple
    global random initializations, and returns the parameters corresponding to the maximum loss,
    as well as the associated expectation value and iteration statistics.
    
    Function:
        1. Builds the Hamiltonian for the current graph g.
        2. Repeatedly (for 'counts' times) performs a global random initialization and subsequent training
           (via global_training) to optimize the QAOA parameters.
        3. For each random initialization, collects the final expectation value (loss), optimized parameters
           (beta_opt and gamma_opt), measurement result, and number of iterations consumed.
        4. Computes the average loss and average iterations over all random initializations.
        5. Identifies the maximum loss value and extracts the corresponding optimized parameters and seed.
    
    Inputs:
        g:       The current subgraph (a NetworkX graph) for which the Hamiltonian is built.
        p:       The number of QAOA layers (or circuit depth) used in the ansatz.
        counts:  An integer specifying the number of global random initializations to be performed.
        SEED:    A list of seed values to be used for each random initialization.
    
    Outputs:
        Returns a tuple (params_opt, max_loss, avg, avg_iterations) where:
            - params_opt:  The optimized parameters [beta_opt, gamma_opt] corresponding to the maximum loss.
            - max_loss:    The maximum expectation (loss) value achieved among all initializations.
            - avg:         The average expectation (loss) value across all runs.
            - avg_iterations: The average number of iterations consumed during the optimization process.
    """
    
    ham = build_ham(g)  # Build the target Hamiltonian for the graph g.

    value = []           # List to store the expectation values (loss) after optimization for each initialization.
    params = []          # List to store the optimized parameters [beta_opt, gamma_opt] for each run.
    measure_result = []  # List to store the measurement results.
    consumed_iterations = []  # List to store the number of iterations consumed in each run.

    # Perform global random initialization 'counts' times.
    for i in range(1, counts + 1):
        my_logger.info('\n\n')
        my_logger.info('The {}-th global random initialization'.format(i))
        # Run global training with the specified Hamiltonian, graph, QAOA layers, and seed.
        result, gamma_opt, beta_opt, loss, loss0 = global_training(ham, g, p, SEED[i - 1])
        
        value.append(loss)
        params.append([beta_opt, gamma_opt])
        measure_result.append(result.data)
        consumed_iterations.append(len(loss0))

    my_logger.info('value = {}'.format(value))
    my_logger.info('\n\n')
    
    my_logger.info('params = [[beta_optimized, gamma_optimized], ...] = {}'.format(params))
    my_logger.info('\n\n')
    
    my_logger.info('measure_result = {}'.format(measure_result))
    my_logger.info('\n\n')
    
    my_logger.info('consumed_iterations = {}'.format(consumed_iterations))
    my_logger.info('\n\n')
    
    # Calculate the average expectation (loss) value.
    avg = 0
    for index in range(0, len(value)):
        avg += value[index]
    avg = avg / len(value)
    avg = round(avg, 5)
    my_logger.info('avg_loss = {}'.format(avg))
    
    # Calculate the average number of iterations consumed.
    avg_iterations = 0
    for j in range(0, len(consumed_iterations)):
        avg_iterations += consumed_iterations[j]
    avg_iterations = avg_iterations / len(consumed_iterations)
    avg_iterations = round(avg_iterations, 5)
    my_logger.info('Average iterations consumed across global random initializations: avg_iterations = {}'.format(avg_iterations))
    
    # Identify the maximum expectation (loss) value obtained from the multiple initializations.
    max_loss = max(value)
    my_logger.info('max_loss = {}'.format(max_loss))
    params_opt = []  # List to store the optimized parameters corresponding to max_loss.
    SEED_opt = []    # List to store the seed values that produced max_loss.

    for j in range(0, len(value)):
        if value[j] == max_loss:
            params_opt.append(params[j])
            SEED_opt.append(SEED[j])
    my_logger.info('Optimized parameters corresponding to max_loss: params_opt = {}'.format(params_opt))
    my_logger.info('\n\n')
    my_logger.info('SEED_opt = {}'.format(SEED_opt))
    
    return params_opt, max_loss, avg, avg_iterations


# In[14]:


# n = 14，prob = 0.4
# E = [(0, 1), (0, 3), (0, 5), (0, 6), (0, 7), (0, 10), (0, 11), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 11), (1, 12), (2, 3), (2, 5), (2, 6), (2, 7), (2, 10), (2, 11), (3, 8), (3, 10), (3, 12), (3, 13), (4, 6), (4, 8), (4, 9), (4, 11), (4, 12), (4, 13), (5, 6), (5, 7), (5, 8), (5, 10), (5, 11), (5, 13), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (7, 11), (7, 12), (7, 13), (8, 9), (8, 11), (9, 10), (9, 11), (9, 12), (9, 13), (10, 12), (10, 13), (11, 12), (12, 13)] # graph

# E = [(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (1, 3), (1, 5), (1, 6), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (2, 4), (2, 7), (2, 8), (2, 10), (3, 4), (3, 7), (3, 13), (4, 8), (4, 9), (4, 12), (4, 13), (5, 6), (5, 7), (5, 9), (5, 10), (5, 11), (5, 12), (6, 7), (6, 8), (6, 9), (6, 11), (6, 12), (6, 13), (7, 8), (7, 9), (7, 10), (7, 11), (7, 13), (8, 9), (8, 12), (10, 11), (10, 12), (11, 12), (11, 13), (12, 13)] # graph2, m = 52

# E = [(0, 1), (0, 2), (0, 5), (0, 6), (0, 8), (0, 10), (0, 11), (0, 12), (1, 2), (1, 4), (1, 5), (1, 7), (1, 8), (1, 9), (1, 10), (1, 13), (2, 3), (2, 6), (2, 8), (3, 6), (3, 7), (3, 9), (3, 10), (3, 12), (3, 13), (4, 5), (4, 6), (4, 7), (4, 10), (4, 11), (4, 12), (4, 13), (5, 6), (5, 7), (5, 8), (5, 10), (5, 11), (5, 13), (6, 10), (6, 11), (6, 13), (7, 8), (7, 10), (7, 11), (7, 12), (8, 9), (8, 10), (8, 12), (9, 10), (9, 13), (10, 13), (11, 12), (12, 13)] # graph3, m = 53

# E = [(0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (1, 2), (1, 3), (1, 5), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4), (2, 6), (2, 9), (2, 11), (2, 12), (2, 13), (3, 4), (3, 5), (3, 6), (3, 8), (3, 10), (3, 11), (3, 12), (4, 5), (4, 6), (4, 7), (4, 9), (4, 11), (4, 12), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 12), (5, 13), (6, 7), (6, 9), (6, 12), (6, 13), (7, 8), (7, 9), (7, 10), (7, 11), (7, 13), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (9, 10), (9, 11), (9, 13), (10, 11), (10, 12), (10, 13), (11, 12), (12, 13)] # graph4, m = 66

# E = [(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 12), (1, 3), (1, 5), (1, 6), (1, 9), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (3, 5), (3, 8), (3, 10), (3, 12), (4, 8), (4, 9), (4, 11), (4, 12), (5, 7), (5, 9), (5, 11), (5, 12), (6, 7), (6, 10), (6, 11), (6, 12), (7, 10), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (9, 10), (9, 11), (9, 13), (10, 12), (10, 13), (11, 13), (12, 13)] # graph5, m = 51


# In[15]:


# 新生成的15个14顶点的ER，prob=0.4图 ，beta = [4, 3, 4, 4, 4,    4, 6, 4, 5, 4,   4, 4, 4, 4, 4]
# E0 = [[(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 13), (1, 5), (1, 6), (1, 7), (1, 9), (1, 10), (1, 11), (2, 3), (2, 4), (2, 6), (2, 7), (2, 11), (2, 13), (3, 5), (3, 8), (3, 9), (3, 11), (3, 12), (3, 13), (4, 5), (4, 10), (4, 12), (5, 6), (5, 7), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (6, 7), (6, 9), (6, 10), (6, 11), (7, 8), (7, 9), (7, 10), (7, 12), (8, 9), (8, 13), (9, 10), (9, 11), (9, 12), (10, 12), (11, 12), (12, 13)], [(0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 10), (1, 11), (1, 12), (1, 13), (2, 5), (2, 7), (2, 9), (2, 10), (2, 12), (2, 13), (3, 4), (3, 5), (3, 8), (3, 9), (3, 10), (3, 12), (4, 5), (4, 8), (4, 9), (4, 12), (4, 13), (5, 7), (5, 8), (5, 11), (5, 12), (5, 13), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 13), (7, 9), (7, 10), (7, 12), (8, 10), (8, 11), (8, 12), (8, 13), (9, 10), (9, 12), (9, 13), (10, 12), (10, 13), (11, 13)], [(0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (1, 2), (1, 3), (1, 5), (1, 9), (1, 10), (1, 12), (1, 13), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 10), (2, 11), (2, 12), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (4, 5), (4, 7), (4, 8), (4, 10), (4, 11), (4, 12), (5, 7), (5, 8), (5, 10), (5, 11), (5, 13), (6, 7), (6, 8), (6, 10), (6, 12), (6, 13), (7, 12), (7, 13), (8, 9), (8, 10), (8, 12), (8, 13), (9, 10), (9, 12), (10, 11), (11, 12), (12, 13)], [(0, 2), (0, 5), (0, 6), (0, 8), (0, 9), (0, 10), (0, 11), (1, 2), (1, 4), (1, 5), (1, 12), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9), (2, 13), (3, 4), (3, 5), (3, 7), (3, 10), (3, 12), (4, 6), (4, 11), (4, 13), (5, 6), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (6, 8), (6, 9), (6, 10), (6, 11), (6, 13), (7, 8), (7, 9), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (9, 11), (9, 12), (9, 13), (10, 11), (10, 13), (11, 12), (12, 13)], [(0, 2), (0, 3), (0, 4), (0, 6), (0, 7), (0, 8), (0, 10), (0, 12), (0, 13), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 12), (1, 13), (2, 7), (2, 10), (2, 11), (2, 12), (2, 13), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (4, 5), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (5, 6), (5, 7), (5, 11), (5, 13), (6, 9), (6, 10), (6, 11), (6, 12), (7, 8), (7, 11), (7, 13), (8, 9), (8, 10), (8, 12), (9, 10), (9, 12), (10, 12), (11, 13), (12, 13)], [(0, 1), (0, 2), (0, 3), (0, 5), (0, 11), (0, 12), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 13), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9), (2, 10), (2, 12), (2, 13), (3, 4), (3, 6), (3, 8), (3, 9), (3, 10), (3, 11), (3, 13), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 13), (5, 8), (5, 11), (5, 12), (5, 13), (6, 8), (6, 9), (6, 10), (7, 10), (7, 11), (7, 13), (8, 11), (8, 12), (8, 13), (9, 10), (9, 13), (10, 11), (10, 12), (10, 13), (11, 12), (11, 13)], [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 12), (1, 2), (1, 3), (1, 4), (1, 7), (1, 8), (1, 9), (1, 10), (1, 12), (1, 13), (2, 3), (2, 6), (2, 13), (3, 8), (3, 10), (3, 12), (4, 6), (4, 11), (4, 12), (5, 8), (5, 10), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (7, 8), (7, 11), (8, 9), (8, 12), (8, 13), (9, 11), (10, 11), (11, 12), (11, 13), (12, 13)], [(0, 2), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 13), (1, 2), (1, 6), (1, 8), (1, 9), (1, 12), (2, 4), (2, 6), (2, 8), (2, 11), (2, 12), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 11), (3, 12), (4, 5), (4, 6), (4, 8), (4, 9), (4, 12), (4, 13), (5, 6), (5, 7), (5, 8), (5, 10), (5, 12), (6, 8), (7, 8), (7, 9), (7, 10), (7, 13), (8, 9), (8, 11), (8, 12), (8, 13), (9, 10), (10, 11), (10, 12), (11, 12), (11, 13), (12, 13)], [(0, 2), (0, 5), (0, 10), (0, 11), (0, 12), (0, 13), (1, 2), (1, 3), (1, 4), (1, 6), (1, 8), (1, 11), (1, 12), (1, 13), (2, 4), (2, 8), (2, 9), (2, 12), (3, 4), (3, 5), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (4, 5), (4, 6), (4, 8), (4, 11), (4, 13), (5, 6), (5, 7), (5, 8), (5, 10), (5, 11), (5, 12), (6, 8), (6, 12), (7, 8), (7, 12), (7, 13), (8, 10), (8, 13), (9, 11), (9, 12), (9, 13), (10, 12), (10, 13), (11, 12), (11, 13), (12, 13)], [(0, 2), (0, 4), (0, 7), (0, 8), (0, 10), (0, 12), (1, 2), (1, 3), (1, 4), (1, 7), (1, 10), (1, 12), (1, 13), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 12), (4, 8), (4, 11), (4, 12), (4, 13), (5, 7), (5, 9), (5, 12), (6, 9), (6, 10), (6, 11), (6, 12), (7, 10), (7, 11), (8, 9), (8, 10), (8, 11), (8, 12), (9, 11), (9, 13), (10, 11), (10, 12), (10, 13), (11, 13), (12, 13)], [(0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 8), (0, 9), (0, 11), (0, 13), (1, 2), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 5), (2, 8), (2, 9), (2, 10), (2, 13), (3, 4), (3, 6), (3, 8), (3, 9), (4, 5), (4, 6), (4, 7), (4, 8), (4, 11), (4, 12), (5, 6), (5, 7), (5, 8), (5, 10), (5, 11), (6, 8), (6, 9), (6, 10), (6, 12), (6, 13), (7, 9), (7, 11), (7, 12), (7, 13), (8, 9), (8, 10), (9, 10), (9, 12), (9, 13), (10, 11), (11, 13)], [(0, 2), (0, 3), (0, 4), (0, 6), (0, 8), (0, 10), (0, 12), (0, 13), (1, 2), (1, 11), (1, 12), (1, 13), (2, 4), (2, 5), (2, 6), (2, 7), (2, 9), (2, 10), (2, 11), (3, 4), (3, 8), (3, 9), (3, 10), (3, 11), (4, 5), (4, 8), (4, 10), (4, 11), (4, 12), (5, 7), (5, 8), (5, 10), (5, 13), (6, 7), (6, 13), (7, 8), (7, 10), (7, 12), (8, 9), (8, 13), (9, 11), (9, 12), (9, 13), (10, 12), (10, 13), (11, 12), (12, 13)], [(0, 1), (0, 2), (0, 4), (0, 7), (0, 8), (0, 12), (0, 13), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4), (2, 5), (2, 10), (2, 11), (2, 13), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 12), (3, 13), (4, 6), (4, 7), (4, 9), (4, 10), (4, 12), (4, 13), (5, 6), (5, 7), (5, 8), (5, 10), (5, 11), (5, 12), (6, 7), (6, 10), (6, 11), (7, 8), (7, 9), (7, 10), (8, 9), (8, 10), (8, 12), (8, 13), (9, 10), (9, 11), (9, 13), (10, 11), (10, 12), (11, 13), (12, 13)], [(0, 1), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 10), (0, 12), (0, 13), (1, 2), (1, 4), (1, 6), (1, 10), (1, 12), (2, 3), (2, 5), (2, 10), (2, 12), (2, 13), (3, 5), (3, 7), (3, 12), (4, 5), (4, 6), (4, 7), (4, 8), (4, 11), (4, 12), (4, 13), (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (6, 10), (6, 11), (7, 8), (7, 12), (8, 10), (8, 11), (8, 12), (8, 13), (9, 10), (9, 11), (9, 12), (10, 11), (10, 12)], [(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (0, 9), (0, 10), (0, 12), (0, 13), (1, 4), (1, 5), (1, 6), (1, 8), (1, 9), (2, 3), (2, 5), (2, 8), (2, 10), (3, 5), (3, 6), (3, 8), (3, 9), (3, 10), (3, 12), (4, 5), (4, 6), (4, 7), (4, 8), (4, 10), (4, 13), (5, 7), (5, 8), (5, 10), (5, 11), (5, 12), (5, 13), (6, 7), (6, 8), (6, 10), (6, 11), (6, 12), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (8, 9), (8, 10), (8, 11), (8, 12), (9, 10), (9, 12), (9, 13), (10, 11), (10, 12), (10, 13), (11, 12), (11, 13), (12, 13)]]

# 15个ER，prob=0.5, beta = [4, 5, 5, 5, 5,    5, 5, 4, 4, 4,    4, 4, 4, 5, 5]
# E0 = [[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 8), (0, 10), (0, 12), (1, 2), (1, 4), (1, 5), (1, 6), (1, 9), (1, 10), (1, 12), (1, 13), (2, 4), (2, 7), (2, 9), (2, 10), (2, 13), (3, 4), (3, 6), (3, 8), (3, 10), (3, 11), (4, 6), (4, 8), (4, 9), (4, 10), (5, 6), (5, 7), (5, 10), (5, 11), (5, 12), (5, 13), (6, 7), (6, 11), (6, 12), (6, 13), (7, 11), (7, 12), (8, 9), (9, 12), (9, 13), (10, 11), (10, 13), (12, 13)], [(0, 2), (0, 3), (0, 4), (0, 10), (0, 12), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 9), (1, 10), (1, 11), (1, 13), (2, 3), (2, 4), (2, 6), (2, 7), (2, 9), (2, 11), (3, 5), (3, 6), (3, 11), (3, 12), (3, 13), (4, 5), (4, 6), (4, 8), (4, 12), (4, 13), (5, 7), (5, 8), (5, 12), (6, 7), (6, 8), (6, 13), (7, 8), (7, 10), (7, 11), (8, 10), (9, 12), (9, 13), (10, 11), (11, 13), (12, 13)], [(0, 1), (0, 3), (0, 4), (0, 6), (0, 7), (0, 8), (0, 10), (1, 3), (1, 5), (1, 8), (1, 11), (1, 12), (1, 13), (2, 3), (2, 4), (2, 8), (2, 11), (2, 12), (3, 4), (3, 6), (3, 7), (3, 10), (3, 12), (4, 6), (4, 8), (4, 10), (4, 12), (4, 13), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (6, 8), (6, 9), (6, 10), (6, 11), (6, 13), (7, 8), (7, 11), (8, 9), (8, 11), (8, 12), (9, 12), (9, 13), (10, 11), (10, 12), (10, 13), (11, 12), (11, 13)], [(0, 1), (0, 5), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13), (1, 5), (1, 6), (1, 7), (1, 9), (1, 10), (1, 13), (2, 3), (2, 4), (2, 6), (2, 10), (2, 13), (3, 4), (3, 6), (3, 7), (3, 10), (3, 11), (3, 12), (3, 13), (4, 5), (4, 6), (4, 7), (4, 10), (4, 12), (4, 13), (5, 8), (5, 10), (5, 12), (6, 7), (6, 9), (6, 12), (7, 9), (7, 12), (7, 13), (8, 9), (8, 10), (9, 11), (9, 13), (10, 11), (10, 13), (11, 13), (12, 13)], [(0, 2), (0, 6), (0, 7), (0, 8), (0, 9), (0, 11), (0, 13), (1, 2), (1, 3), (1, 8), (1, 10), (1, 11), (1, 13), (2, 4), (2, 7), (2, 10), (2, 11), (2, 12), (2, 13), (3, 5), (3, 8), (3, 9), (3, 11), (4, 5), (4, 9), (4, 12), (4, 13), (5, 11), (5, 12), (6, 9), (6, 11), (6, 12), (6, 13), (7, 9), (7, 12), (8, 10), (8, 13), (9, 10), (9, 11), (10, 11), (10, 13), (11, 12), (11, 13), (12, 13)], [(0, 1), (0, 3), (0, 9), (0, 10), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 10), (1, 11), (1, 12), (2, 4), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (3, 4), (3, 6), (3, 7), (3, 11), (3, 12), (4, 5), (4, 7), (4, 8), (4, 10), (4, 11), (4, 12), (4, 13), (5, 10), (5, 11), (5, 13), (6, 8), (6, 9), (6, 11), (6, 12), (6, 13), (7, 8), (8, 12), (9, 11), (9, 12), (10, 12), (11, 12), (12, 13)], [(0, 5), (0, 7), (0, 8), (0, 9), (0, 12), (1, 2), (1, 3), (1, 5), (1, 8), (1, 9), (1, 12), (2, 4), (2, 5), (2, 6), (2, 7), (2, 9), (2, 10), (3, 4), (3, 7), (4, 5), (4, 7), (4, 8), (4, 10), (4, 11), (4, 12), (4, 13), (5, 6), (5, 11), (5, 13), (6, 8), (6, 9), (6, 11), (6, 12), (6, 13), (7, 9), (7, 10), (7, 12), (8, 9), (8, 11), (8, 12), (8, 13), (9, 11), (9, 13), (10, 13), (11, 12), (11, 13)], [(0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 2), (1, 3), (1, 7), (1, 10), (1, 12), (1, 13), (2, 7), (2, 8), (2, 9), (2, 12), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9), (3, 11), (3, 12), (4, 5), (4, 6), (4, 7), (4, 12), (4, 13), (5, 7), (5, 10), (6, 7), (6, 8), (6, 11), (6, 12), (6, 13), (7, 8), (7, 11), (7, 12), (7, 13), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (9, 10), (9, 11), (9, 12), (10, 11), (10, 12), (11, 12), (11, 13), (12, 13)], [(0, 1), (0, 2), (0, 3), (0, 5), (0, 6), (0, 7), (0, 10), (0, 13), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 9), (2, 10), (2, 13), (3, 5), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 13), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (5, 6), (5, 7), (5, 8), (5, 11), (5, 13), (6, 7), (6, 8), (6, 10), (6, 11), (6, 12), (6, 13), (7, 9), (7, 10), (7, 12), (8, 11), (8, 12), (8, 13), (9, 10), (11, 12)], [(0, 2), (0, 4), (0, 5), (0, 7), (0, 8), (0, 10), (0, 12), (1, 3), (1, 6), (1, 7), (1, 8), (1, 9), (1, 11), (1, 12), (1, 13), (2, 4), (2, 7), (2, 11), (2, 12), (3, 4), (3, 5), (3, 6), (3, 7), (3, 9), (4, 5), (4, 6), (4, 10), (4, 11), (4, 12), (5, 6), (5, 7), (5, 8), (6, 9), (6, 11), (6, 12), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (8, 11), (9, 11), (9, 13), (10, 13), (11, 12), (11, 13), (12, 13)], [(0, 1), (0, 2), (0, 4), (0, 6), (0, 8), (0, 9), (0, 10), (0, 12), (0, 13), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8), (1, 10), (1, 13), (2, 5), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (3, 6), (3, 10), (3, 12), (4, 7), (4, 11), (4, 13), (5, 6), (5, 8), (5, 9), (5, 12), (6, 9), (7, 8), (7, 10), (7, 11), (7, 12), (8, 9), (8, 10), (8, 11), (8, 13), (9, 10), (9, 13), (10, 11), (10, 13), (11, 13), (12, 13)], [(0, 2), (0, 5), (0, 6), (0, 8), (0, 9), (0, 12), (1, 2), (1, 3), (1, 4), (1, 7), (1, 8), (1, 9), (1, 11), (2, 6), (2, 7), (2, 9), (2, 12), (2, 13), (3, 5), (3, 6), (3, 8), (3, 10), (3, 12), (3, 13), (4, 5), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (5, 8), (5, 9), (5, 10), (5, 12), (5, 13), (6, 7), (6, 8), (6, 10), (6, 13), (7, 8), (7, 11), (7, 12), (8, 9), (8, 12), (9, 10), (9, 13), (10, 11), (10, 12), (10, 13)], [(0, 2), (0, 3), (0, 6), (0, 8), (0, 10), (0, 11), (0, 12), (1, 2), (1, 3), (1, 4), (1, 8), (1, 13), (2, 5), (2, 6), (2, 8), (2, 9), (2, 12), (2, 13), (3, 4), (3, 5), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (4, 6), (4, 7), (4, 8), (4, 10), (4, 11), (4, 12), (4, 13), (5, 7), (5, 8), (5, 13), (6, 7), (6, 8), (6, 10), (6, 11), (6, 13), (7, 10), (8, 9), (8, 12), (9, 11), (9, 12), (10, 11), (10, 12), (11, 12)], [(0, 2), (0, 3), (0, 4), (0, 5), (0, 8), (0, 11), (0, 12), (1, 4), (1, 5), (1, 8), (1, 10), (2, 3), (2, 6), (2, 11), (2, 13), (3, 4), (3, 5), (3, 6), (3, 10), (4, 5), (4, 7), (4, 9), (4, 10), (4, 11), (5, 6), (5, 7), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (6, 8), (6, 10), (6, 11), (6, 13), (7, 8), (7, 9), (7, 10), (7, 12), (7, 13), (8, 9), (8, 10), (8, 11), (9, 10), (9, 12), (9, 13), (10, 13)], [(0, 1), (0, 3), (0, 5), (0, 6), (0, 8), (0, 9), (0, 11), (0, 12), (1, 2), (1, 4), (1, 6), (1, 7), (1, 11), (1, 12), (1, 13), (2, 3), (2, 5), (2, 6), (2, 8), (3, 4), (3, 6), (3, 8), (3, 13), (4, 5), (4, 6), (4, 7), (4, 8), (5, 9), (5, 10), (5, 12), (5, 13), (6, 9), (7, 10), (7, 12), (8, 9), (8, 10), (8, 11), (8, 13), (9, 11), (9, 12), (9, 13), (10, 11), (11, 12), (12, 13)]]

# 15个2-regular, beta  = [7, 7, 6, 7, 6,    6, 7, 7, 6, 7,   7, 6, 7, 7, 6]
# E0 = [[(2, 6), (2, 3), (6, 8), (5, 10), (5, 0), (10, 1), (8, 9), (12, 13), (12, 0), (13, 7), (7, 9), (3, 11), (11, 4), (1, 4)], [(7, 8), (7, 5), (8, 11), (5, 10), (10, 6), (2, 9), (2, 3), (9, 1), (11, 12), (6, 1), (0, 13), (0, 12), (13, 4), (4, 3)], [(8, 13), (8, 6), (13, 4), (6, 10), (5, 11), (5, 1), (11, 2), (10, 9), (2, 12), (0, 7), (0, 3), (7, 3), (9, 1), (4, 12)], [(4, 7), (4, 5), (7, 8), (3, 12), (3, 10), (12, 6), (2, 9), (2, 8), (9, 0), (5, 1), (10, 0), (1, 11), (6, 13), (13, 11)], [(5, 9), (5, 6), (9, 11), (1, 3), (1, 7), (3, 7), (6, 8), (8, 10), (10, 0), (4, 13), (4, 12), (13, 2), (2, 0), (11, 12)], [(0, 1), (0, 11), (1, 4), (6, 9), (6, 10), (9, 4), (5, 10), (5, 11), (8, 13), (8, 7), (13, 7), (3, 12), (3, 2), (12, 2)], [(2, 6), (2, 11), (6, 5), (3, 12), (3, 5), (12, 4), (4, 9), (9, 10), (8, 13), (8, 1), (13, 7), (7, 1), (11, 0), (10, 0)], [(0, 1), (0, 3), (1, 12), (5, 9), (5, 2), (9, 4), (2, 7), (7, 11), (3, 12), (4, 6), (8, 13), (8, 10), (13, 11), (6, 10)], [(9, 13), (9, 12), (13, 12), (2, 6), (2, 0), (6, 4), (5, 10), (5, 8), (10, 7), (1, 3), (1, 4), (3, 11), (8, 11), (0, 7)], [(0, 1), (0, 7), (1, 9), (2, 6), (2, 9), (6, 4), (8, 13), (8, 4), (13, 10), (3, 12), (3, 5), (12, 7), (10, 11), (5, 11)], [(2, 7), (2, 11), (7, 6), (4, 10), (4, 0), (10, 12), (6, 8), (8, 9), (3, 13), (3, 5), (13, 1), (1, 5), (11, 9), (12, 0)], [(4, 10), (4, 2), (10, 7), (6, 9), (6, 12), (9, 5), (5, 8), (0, 12), (0, 3), (3, 8), (2, 13), (13, 11), (7, 1), (1, 11)], [(1, 2), (1, 10), (2, 6), (4, 10), (4, 13), (6, 9), (9, 0), (0, 13), (7, 12), (7, 11), (12, 5), (5, 8), (3, 11), (3, 8)], [(9, 13), (9, 11), (13, 6), (1, 3), (1, 8), (3, 10), (4, 5), (4, 12), (5, 7), (2, 8), (2, 11), (10, 12), (7, 0), (0, 6)], [(4, 9), (4, 11), (9, 3), (8, 12), (8, 2), (12, 10), (5, 6), (5, 0), (6, 1), (2, 13), (10, 13), (3, 0), (1, 7), (7, 11)]]

# 15个3-regular, beta = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
E0 = [[(1, 3), (1, 6), (1, 13), (3, 13), (3, 4), (10, 11), (10, 7), (10, 5), (11, 6), (11, 8), (7, 12), (7, 9), (12, 2), (12, 5), (8, 9), (8, 0), (9, 2), (6, 13), (2, 4), (5, 0), (0, 4)], [(10, 11), (10, 3), (10, 9), (11, 9), (11, 13), (5, 6), (5, 2), (5, 12), (6, 1), (6, 7), (1, 2), (1, 8), (2, 12), (12, 0), (7, 13), (7, 8), (4, 9), (4, 0), (4, 3), (3, 13), (0, 8)], [(10, 11), (10, 8), (10, 4), (11, 0), (11, 12), (2, 5), (2, 1), (2, 0), (5, 8), (5, 3), (8, 1), (1, 4), (6, 7), (6, 12), (6, 0), (7, 9), (7, 13), (4, 3), (3, 13), (13, 9), (12, 9)], [(5, 9), (5, 4), (5, 0), (9, 10), (9, 1), (1, 3), (1, 0), (3, 11), (3, 7), (11, 6), (11, 2), (4, 12), (4, 10), (12, 13), (12, 6), (7, 2), (7, 8), (13, 2), (13, 6), (8, 10), (8, 0)], [(1, 3), (1, 4), (1, 12), (3, 0), (3, 9), (8, 13), (8, 10), (8, 11), (13, 2), (13, 9), (7, 12), (7, 6), (7, 10), (12, 5), (0, 10), (0, 9), (2, 5), (2, 6), (5, 11), (6, 4), (11, 4)], [(4, 7), (4, 8), (4, 13), (7, 12), (7, 10), (1, 3), (1, 2), (1, 12), (3, 9), (3, 2), (8, 5), (8, 10), (5, 6), (5, 11), (6, 2), (6, 13), (12, 9), (0, 10), (0, 11), (0, 13), (11, 9)], [(5, 9), (5, 2), (5, 12), (9, 10), (9, 1), (10, 11), (10, 1), (11, 3), (11, 0), (2, 8), (2, 4), (8, 1), (8, 7), (3, 0), (3, 12), (4, 12), (4, 6), (0, 13), (6, 7), (6, 13), (7, 13)], [(6, 9), (6, 7), (6, 13), (9, 8), (9, 1), (10, 11), (10, 0), (10, 12), (11, 7), (11, 8), (4, 8), (4, 0), (4, 3), (5, 13), (5, 2), (5, 1), (13, 12), (2, 12), (2, 3), (0, 1), (3, 7)], [(7, 11), (7, 6), (7, 8), (11, 0), (11, 13), (5, 8), (5, 1), (5, 0), (8, 3), (4, 9), (4, 13), (4, 1), (9, 1), (9, 12), (6, 10), (6, 3), (10, 12), (10, 2), (12, 0), (13, 2), (2, 3)], [(4, 7), (4, 9), (4, 10), (7, 0), (7, 2), (0, 8), (0, 9), (5, 8), (5, 1), (5, 11), (8, 1), (1, 2), (2, 3), (9, 13), (12, 13), (12, 6), (12, 11), (13, 3), (3, 10), (10, 6), (6, 11)], [(6, 9), (6, 1), (6, 4), (9, 8), (9, 13), (1, 3), (1, 12), (3, 10), (3, 5), (0, 7), (0, 5), (0, 2), (7, 10), (7, 2), (8, 10), (8, 11), (12, 13), (12, 5), (13, 4), (4, 11), (11, 2)], [(1, 3), (1, 11), (1, 13), (3, 7), (3, 9), (7, 12), (7, 8), (12, 4), (12, 5), (5, 13), (5, 11), (13, 9), (8, 9), (8, 6), (4, 0), (4, 2), (0, 10), (0, 6), (10, 6), (10, 2), (11, 2)], [(6, 9), (6, 1), (6, 10), (9, 8), (9, 2), (8, 13), (8, 12), (13, 12), (13, 3), (0, 7), (0, 10), (0, 4), (7, 3), (7, 2), (1, 11), (1, 5), (10, 4), (3, 4), (2, 5), (5, 11), (11, 12)], [(5, 9), (5, 6), (5, 2), (9, 8), (9, 7), (10, 11), (10, 3), (10, 6), (11, 3), (11, 13), (6, 0), (3, 4), (0, 7), (0, 1), (7, 1), (8, 12), (8, 1), (4, 12), (4, 13), (12, 2), (2, 13)], [(6, 9), (6, 3), (6, 11), (9, 11), (9, 13), (1, 3), (1, 2), (1, 10), (3, 2), (8, 13), (8, 4), (8, 0), (13, 5), (4, 10), (4, 5), (7, 12), (7, 0), (7, 10), (12, 2), (12, 0), (5, 11)]]


# In[16]:


# prob = 0.5, beta_G = 5,4,4,4,5
# E = [(0, 3), (0, 4), (0, 7), (0, 10), (0, 11), (1, 3), (1, 5), (1, 8), (1, 9), (1, 13), (2, 4), (2, 10), (2, 11), (3, 4), (3, 5), (3, 6), (3, 11), (3, 12), (4, 5), (4, 6), (4, 7), (4, 8), (4, 10), (4, 12), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (6, 7), (6, 11), (6, 12), (6, 13), (7, 10), (7, 11), (7, 12), (7, 13), (8, 10), (8, 13), (9, 12), (9, 13), (11, 12), (12, 13)]#graph1,m=45
# E = [(0, 3), (0, 6), (0, 7), (0, 10), (0, 12), (0, 13), (1, 2), (1, 4), (1, 5), (1, 9), (1, 12), (1, 13), (2, 4), (2, 5), (2, 7), (2, 8), (2, 11), (2, 13), (3, 4), (3, 6), (3, 8), (3, 13), (4, 5), (4, 7), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (5, 7), (5, 9), (5, 10), (5, 11), (5, 13), (6, 9), (6, 12), (7, 8), (7, 10), (7, 11), (7, 12), (7, 13), (8, 11), (8, 12), (9, 10), (9, 11), (10, 12), (10, 13), (11, 12), (11, 13), (12, 13)] #graph2,m=50
# E = [(0, 1), (0, 2), (0, 6), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (1, 2), (1, 4), (1, 6), (1, 8), (1, 9), (1, 10), (1, 12), (1, 13), (2, 4), (2, 12), (2, 13), (3, 7), (3, 9), (3, 10), (3, 13), (4, 5), (4, 6), (4, 7), (4, 8), (4, 11), (5, 8), (5, 9), (5, 10), (5, 11), (5, 13), (6, 7), (6, 8), (6, 9), (6, 10), (6, 13), (7, 8), (7, 9), (7, 11), (7, 12), (7, 13), (8, 9), (8, 10), (8, 12), (8, 13), (9, 12), (10, 12), (10, 13), (12, 13)] # graph3,m=51
# E = [(0, 1), (0, 4), (0, 5), (0, 7), (0, 8), (0, 10), (0, 11), (0, 13), (1, 2), (1, 7), (1, 11), (1, 12), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9), (2, 12), (3, 8), (3, 12), (3, 13), (4, 5), (4, 6), (4, 8), (4, 10), (4, 13), (5, 6), (5, 7), (5, 8), (5, 10), (5, 12), (5, 13), (6, 7), (6, 9), (6, 10), (6, 12), (6, 13), (7, 8), (7, 10), (7, 11), (7, 12), (7, 13), (8, 10), (9, 10), (10, 11), (11, 13), (12, 13)] # graph4, m = 47
# E = [(0, 3), (0, 6), (0, 7), (0, 9), (0, 12), (0, 13), (1, 4), (1, 6), (1, 9), (1, 11), (1, 12), (2, 4), (2, 8), (2, 10), (2, 11), (2, 12), (3, 4), (3, 6), (3, 7), (3, 9), (3, 11), (3, 13), (4, 5), (4, 6), (4, 9), (4, 11), (4, 13), (5, 6), (5, 7), (5, 13), (6, 7), (6, 9), (6, 10), (6, 13), (7, 8), (7, 9), (7, 10), (7, 11), (9, 10), (9, 11), (9, 12), (10, 12), (10, 13), (12, 13)] # graph5,m=44


# 14个顶点，2-regular graph
# E = [(5, 9), (5, 2), (9, 10), (4, 7), (4, 11), (7, 10), (1, 3), (1, 6), (3, 12), (12, 0), (6, 8), (8, 2), (0, 13), (13, 11)] # graph1
# E = [(1, 3), (1, 11), (3, 4), (8, 13), (8, 5), (13, 7), (2, 9), (2, 10), (9, 11), (7, 12), (12, 6), (5, 6), (10, 0), (0, 4)] # graph2
# E = [(2, 9), (2, 8), (9, 3), (6, 11), (6, 5), (11, 8), (12, 13), (12, 0), (13, 4), (5, 3), (0, 4), (7, 10), (7, 1), (10, 1)] #graph3
# E = [(7, 8), (7, 1), (8, 4), (9, 13), (9, 3), (13, 11), (4, 2), (3, 10), (10, 6), (6, 0), (0, 12), (12, 11), (1, 5), (5, 2)] # graph4
# E = [(10, 11), (10, 3), (11, 8), (8, 7), (3, 6), (4, 13), (4, 0), (13, 2), (2, 5), (0, 5), (1, 9), (1, 7), (9, 12), (6, 12)] # graph5

# 14个顶点，3-regular graph
# E = [(6, 9), (6, 11), (6, 13), (9, 11), (9, 12), (5, 13), (5, 2), (5, 1), (13, 3), (0, 7), (0, 10), (0, 8), (7, 10), (7, 8), (4, 12), (4, 3), (4, 2), (12, 10), (2, 1), (1, 11), (3, 8)] #graph1
# E = [(6, 9), (6, 0), (6, 13), (9, 2), (9, 0), (5, 13), (5, 2), (5, 8), (13, 3), (3, 11), (3, 12), (11, 8), (11, 4), (2, 7), (8, 7), (10, 12), (10, 7), (10, 1), (12, 1), (0, 4), (4, 1)] #graph2
# E = [(7, 12), (7, 11), (7, 5), (12, 2), (12, 13), (8, 9), (8, 11), (8, 6), (9, 1), (9, 0), (2, 5), (2, 1), (0, 10), (0, 11), (10, 3), (10, 4), (5, 4), (1, 13), (13, 3), (3, 6), (6, 4)] #graph3
# E = [(4, 7), (4, 11), (4, 3), (7, 12), (7, 5), (10, 11), (10, 6), (10, 1), (11, 0), (12, 13), (12, 9), (2, 8), (2, 5), (2, 6), (8, 9), (8, 5), (9, 1), (0, 3), (0, 13), (3, 6), (13, 1)] #graph4
# E = [(5, 6), (5, 11), (5, 4), (6, 3), (6, 4), (3, 11), (3, 13), (11, 12), (4, 12), (12, 8), (0, 10), (0, 1), (0, 9), (10, 8), (10, 7), (1, 2), (1, 7), (2, 9), (2, 13), (9, 13), (8, 7)] #graph5


# In[17]:


# target graph
n = 14
V = []
E = E0[12]
for node in range(0,n):
    V.append(node)

target_graph = nx.Graph()
target_graph.add_nodes_from(V)
target_graph.add_edges_from(E)


# In[18]:


target_graph = nx.Graph()
target_graph.add_nodes_from(V)
target_graph.add_edges_from(E)

pos = nx.circular_layout(target_graph)
options = {
    "with_labels": True,
    "font_size": 16,
    "font_weight": "bold",
    "font_color": "white",
    "node_size": 1000,
    "width": 2
}
nx.draw_networkx(target_graph, pos, **options)
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()


# In[20]:


# Retrieve the expectation values obtained from 20*p rounds of AQA at a given level depth,
# as well as the average number of iterations consumed.
def AQA(p):
    """
    Runs the AQA optimization process for a given level depth p by performing multiple rounds
    of progressive learning (20*p rounds in total). This function collects the final expectation
    values, computes the average number of iterations consumed in each round, and measures the total classical
    simulation time. Additionally, it gathers quantum resource information such as the number of qubits,
    circuit depth, and gate counts.
    
    Process:
      1. Set convergence threshold (delta) and determine the number of progressive learning runs 
         (times) and random initializations per run (counts).
      2. Generate a list of random seed values for all runs.
      3. For each progressive learning run:
            a. Build the initial subgraph.
            b. Initialize the subgraph and record its initial loss (expectation value) after optimizing parameters,
               along with the number of iterations consumed.
            c. If the subgraph's parameters can be successfully transferred (i.e., the initial expectation value on the new subgraph
               is acceptable), perform further parameter optimization and update the current subgraph by adding new nodes.
            d. Stop the progressive learning process if either the parameter transfer fails or the improvement in the objective
               function becomes negligible, indicating that further optimization is unlikely to improve the result.
      4. Log and record quantum resource usage (qubits, circuit depth, gate counts) and the optimization statistics.
      5. Compute average metrics such as average loss, average iterations, and average quantum resources over all runs.
      6. Return the final expectation values at level depth p, the overall average iterations, total simulation time,
         and the average quantum resources consumed.
    
    Inputs:
        p: An integer specifying the QAOA ansatz depth (level depth).
    
    Outputs:
        Returns a tuple:
            - final_value: A list of the final expectation (loss) values for each progressive learning run.
            - avg: The average number of iterations consumed per AQA run.
            - delta_t: The total classical simulation time for the entire process.
            - avg_qubits: The average number of qubits used.
            - avg_depth: The average circuit depth.
            - avg_RX: The average number of RX gates used.
            - avg_RZ: The average number of RZ gates used.
            - avg_multi: The average number of multi-qubit controlled gates used.
    """
    
    delta = 0.1  # Convergence threshold for the change in loss.
    value = []  # List to store the final expectation values (final_value) for different runs.
    average_iterations = []  # List to store the average number of iterations consumed per progressive learning run.
    counts = 5  # Number of random initializations per progressive learning run.
    times = 10  # Number of progressive learning runs.

    # Generate a list of random seeds to be used for all runs.
    total_SEED = [] 
    for i in range(0, counts * times):
        total_SEED.append(random.randint(1, 25000))
    my_logger.info('p= {}, Random seeds total_SEED = {}'.format(p, total_SEED))

    start_time = time.time()  # Record the start time of the optimization process.
    start = 0  # Index marking the start of seeds for the first progressive learning run.
    
    for t in range(1, times + 1):
        # Build the initial subgraph.
        old_V, old_E = build_initial_graph(2)  # Create an initial subgraph of a given size.
        my_logger.info('initial_subgraph_V = {}, initial_subgraph_E = {}'.format(old_V, old_E))
    
        initial_n = len(old_V)  # Size of the initial subgraph.
        # Initialize the subgraph.
        sub_graph = nx.Graph()
        sub_graph.add_nodes_from(old_V)
        sub_graph.add_edges_from(old_E)
        
        # Record the expectation values for the current progressive learning run.
        sub_loss = []

        # Determine the list of seeds to be used for the current progressive learning run.
        SEED = []
        end = t * counts  # End index for the seed values of the current run.
        for seed in range(start, end):
            SEED.append(total_SEED[seed])
        start = end  # Update the seed index for the next run.
        my_logger.info('{}-th progressive learning run, SEED = {}'.format(t, SEED))

        # Loss_p will store the optimized expectation values obtained at a given circuit depth p.
        loss_p = []

        # Compute the number of times 'construct_graph' is called to evolve from the initial subgraph to the target graph.
        sub_num_nodes = sub_graph.number_of_nodes()       # Number of nodes in the current subgraph.
        target_num_nodes = target_graph.number_of_nodes()   # Total nodes in the target graph.
        consumed_iter = []  # List to store the iterations consumed during each optimization phase.

        # Find the optimal parameters for the initial subgraph.
        params_opt, max_loss, avg, avg_iterations = search_optimized_parameters(sub_graph, p, counts, SEED)
        loss_p.append(max_loss)
        consumed_iter.append(avg_iterations)

        # Initialize the old_graph with the current subgraph.
        old_graph = sub_graph
        # Pre-check: if the current subgraph is not the target graph, create a new graph.
        if len(old_graph.nodes()) != len(target_graph.nodes()):
            new_graph = construct_graph(old_graph, target_graph)

        for i in range(0, target_num_nodes - sub_num_nodes):
            # Evaluate whether the current parameters perform well on the new subgraph.
            ham = build_ham(new_graph)
            initial_expectation_value = calculate_initial_expectation_value(new_graph, p, params_opt[0][0], params_opt[0][1])
            my_logger.info('Transferring optimized parameters params_opt = {} from old_V = {} and old_E = {} to new subgraph (new_V = {}, new_E = {}), initial_expectation_value = {}'.format(
                params_opt, old_graph.nodes(), old_graph.edges(), new_graph.nodes(), new_graph.edges(), initial_expectation_value))

            # If the parameter transfer is not effective (i.e., the initial expectation value on the new subgraph
            # is less than half of the previous loss), abandon further parameter optimization.
            value0 = loss_p[len(loss_p) - 1]  # The last optimized expectation value.
            if initial_expectation_value < value0 / 2:
                my_logger.info('Parameter transfer effect is poor; abandoning further optimization for this progressive learning run.')
                my_logger.info('\n')
                break

            else:
                # If parameter transfer is successful, perform further parameter optimization.
                result, gamma_opt, beta_opt, loss, loss0, qubits0, circuit_depth, quantum_gates = execute_function(new_graph, p, params_opt[0][0], params_opt[0][1])
                loss_p.append(loss)
                consumed_iter.append(len(loss0))  # Record iterations consumed in this optimization.
                my_logger.info('new_V = {}, new_E = {}, loss after parameter transfer optimization = {}'.format(new_graph.nodes(), new_graph.edges(), loss))
                
                # Update optimized parameters.
                params_opt = []
                params_opt.append([beta_opt, gamma_opt])
                my_logger.info('Current loss_p = {}'.format(loss_p))
                my_logger.info('\n')

                # Check whether to continue adding nodes based on the change in expectation value.
                index = len(loss_p)
                a = loss_p[index - 1]
                b = loss_p[index - 2]
                if a <= b - 1:
                    my_logger.info('Post-optimization, the expectation value degraded significantly; terminating this progressive learning run.')
                    my_logger.info('\n')
                    break
                else:
                    index = len(loss_p)
                    if index >= 3:
                        delta1 = abs(loss_p[index - 1] - loss_p[index - 2])
                        delta2 = abs(loss_p[index - 2] - loss_p[index - 3])
                        if delta1 <= delta and delta2 <= delta and len(new_graph.nodes()) < len(target_graph.nodes()):
                            my_logger.info('Subgraph size increased, but the optimized expectation value remains nearly unchanged. Current subgraph size: {}'.format(len(new_graph.nodes())))
                            my_logger.info('Recommending the optimized parameters for the current subgraph (params_opt = {}) to the target graph.'.format(params_opt))
                            my_logger.info('Recommended parameters: params_opt = {}'.format(params_opt))
                            my_logger.info('\n')
                            break
                        else:
                            # Update old_graph for subsequent vertex additions.
                            old_graph = new_graph
                            if len(old_graph.nodes()) != len(target_graph.nodes()):
                                new_graph = construct_graph(old_graph, target_graph)
                            else:
                                my_logger.info('Current graph size equals target graph size; terminating this progressive learning run.')
                                my_logger.info('\n')
                                break
                    else:
                        old_graph = new_graph
                        if len(old_graph.nodes()) != len(target_graph.nodes()):
                            new_graph = construct_graph(old_graph, target_graph)
                        else:
                            my_logger.info('Current graph size equals target graph size; terminating this progressive learning run.')
                            my_logger.info('\n')
                            break

        my_logger.info('Progressive learning run {}: Expectation value history (sub_loss) = {}'.format(t, loss_p))
        my_logger.info('Progressive learning run {}: Iterations consumed = {}'.format(t, consumed_iter))
        # my_logger.info('Final optimized resource usage for this run: qubits = {}, circuit_depth = {}, quantum_gates = {}'.format(qubits0[0], circuit_depth[0], quantum_gates))
        
        # Record quantum resource usage.
        quantum_resources = []
        quantum_resources.append([qubits0[0], circuit_depth[0], quantum_gates])
        
        # Calculate the total number of iterations consumed in this progressive learning run.
        avg0 = 0
        for j0 in range(0, len(consumed_iter)):
            avg0 += consumed_iter[j0]
        average_iterations.append(avg0)
        value.append(loss_p)
        
        my_logger.info('\n')
        if len(value) % 5 == 0:
            my_logger.info('After {} progressive learning runs, the expectation values are: value = {}'.format(t, value))
    
    # End time and calculation of total simulation time.
    end_time = time.time()
    delta_t = end_time - start_time
    my_logger.info('start_time = {}'.format(start_time))
    my_logger.info('end_time = {}'.format(end_time))
    my_logger.info('delta_t = {}'.format(delta_t))
    
    my_logger.info('After {} progressive learning runs, the expectation values are: value = {}'.format(times, value))
    my_logger.info('\n')
    
    my_logger.info('After {} progressive learning runs, quantum resources used: quantum_resources = {}'.format(times, quantum_resources))
    
    # Compute the average quantum resource usage: qubits, RX_gates, RZ_gates, multi-qubit gates, and circuit depth.
    avg_qubits = 0
    avg_RX = 0
    avg_RZ = 0
    avg_multi = 0
    avg_depth = 0
    
    for i0 in range(0, len(quantum_resources)):
        avg_qubits += quantum_resources[i0][0]
        avg_depth += quantum_resources[i0][1]
        quantum_gates = quantum_resources[i0][2]
        avg_RX += quantum_gates[0][0]
        avg_RZ += quantum_gates[0][1]
        avg_multi += quantum_gates[0][2]
    avg_qubits = round(avg_qubits / len(quantum_resources), 5)
    avg_depth = round(avg_depth / len(quantum_resources), 5)
    avg_RX = round(avg_RX / len(quantum_resources), 5)
    avg_RZ = round(avg_RZ / len(quantum_resources), 5)
    avg_multi = round(avg_multi / len(quantum_resources), 5)
    my_logger.info('In {} rounds of AQA, average resources: avg_qubits = {}, avg_depth = {}, avg_RX = {}, avg_RZ = {}, avg_multi = {}'.format(
        times, avg_qubits, avg_depth, avg_RX, avg_RZ, avg_multi))
    
    # Get the final expectation values for each run at level depth p.
    final_value = []
    for i in range(0, len(value)):
        l = len(value[i])
        final_value.append(value[i][l - 1])
    my_logger.info('final_value = {}'.format(final_value))
    
    my_logger.info('In {} rounds of AQA, the average iterations per run: average_iterations = {}'.format(times, average_iterations))
    
    # Calculate the overall average number of iterations consumed across all runs.
    avg = 0
    for j0 in range(0, len(average_iterations)):
        avg += average_iterations[j0]
    avg = avg / len(average_iterations)
    avg = round(avg, 5)
    my_logger.info('avg_iterations consumed by {} runs of AQA is {}'.format(times, avg))
    
    # Return the final expectation values, average iterations, total simulation time,
    # and average quantum resource usage (qubits, circuit depth, RX, RZ, and multi-qubit gate counts).
    return final_value, avg, delta_t, avg_qubits, avg_depth, avg_RX, avg_RZ, avg_multi



# In[21]:


p = int(input('Please input the maximum circuit depth:')) # level depth

# delta = float(input('Please input the predefined error:'))
data = []           # Stores the final expectation values (final_value) obtained for different values of p.
iterations_avg = [] # Stores the average number of iterations consumed by AQA for each p.
T = []              # Stores the classical simulation time for AQA (simulated p*20 times) for each p.

qubits0 = []
circuit_depth = []
RX_gates = []
RZ_gates = []
multi_gates = []

for depth in range(1,p+1):
    final_value,avg,delta_t,avg_qubits,avg_depth,avg_RX,avg_RZ,avg_multi = AQA(depth)
    data.append(final_value)
    iterations_avg.append(avg)
    T.append(delta_t)
    qubits0.append(avg_qubits)
    circuit_depth.append(avg_depth)
    RX_gates.append(avg_RX)
    RZ_gates.append(avg_RZ)
    multi_gates.append(avg_multi)
    
    
my_logger.info('data = {}'.format(data))
my_logger.info('iterations_avg = {}'.format(iterations_avg))
my_logger.info('T = {}'.format(T))


# In[22]:


my_logger.info('average consumption of qubits : avg_qubits = {}'.format(qubits0))
my_logger.info('average consumption of depth: avg_depth =  {}'.format(circuit_depth))
my_logger.info('average consumption of RX: avg_RX =  {}'.format(RX_gates))
my_logger.info('average consumption of RZ: avg_RZ =  {}'.format(RZ_gates))
my_logger.info('average consumption of multi-controlled gates: multi_gates =  {}'.format(multi_gates))

