---
title: "AODV: Ad-hoc On-Demand Distance Vector Routing Protocol"
description: >
  Reference guide and conceptual overview of the Ad-hoc On-demand Distance Vector
  (AODV) reactive routing protocol for dynamic mobile ad-hoc networks, including
  message types, loop prevention, and Python simulation pseudocode.
created: 2026-11-05
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Archived from knowledge-base/ant-colony/what_is_AODV.md with standardized
          frontmatter and code fence identifiers.
related_files:
  - "[notes_simple_ant_colony_routing.md](./notes_simple_ant_colony_routing.md)"
tags:
  - notes/routing
  - notes/networking
  - notes/archive
  - notes/protocols
  - algorithm/routing
  - guide/networking
  - analysis/routing
  - notes/manet
  - notes/simulation
  - notes/pseudocode
---

# AODV

## Acronym List

- **RERR**: Route Error Report
- **RREQ**: Route Request
- **RREP**: Route Reply
- **AODV**: Ad-hoc On-demand Distance Vector Routing

## AODV: Ad-hoc On-demand Distance Vector Routing

AODV (Ad-hoc On-demand Distance Vector Routing) is a routing protocol for mobile ad hoc networks (MANETs). It is a reactive, on-demand, table-driven, and destination-sequenced distance-vector routing protocol. Proactive protocols, in contrast, maintain a complete list of routes at all times.

Its purpose is for routing mobile networks where nodes are interconnected, communicating with each other without relying on a central infrastructure.

### Use Case

When the arrangement or organization of network nodes, known as network topologies, can change frequently.  
Suited for smaller, dynamic networks.

### Key Features

1. **On-demand**: Routes are created only when needed.
2. **Route maintenance**: Routes are maintained as long as they are needed by the source node.
3. **Reactive**: Nodes react to changes in the network by updating their routing tables.
4. **Sequence Numbers**: AODV uses sequence numbers to ensure the freshness of routes and to detect loops in the network.
5. **Route Error Messages**: When a link break in an active route is detected, a route error (RERR) message is broadcasted to inform other nodes about the broken link.

### Basic Operation

1. **Route Discovery**: When a node needs to send a packet to a destination node, it initiates a Route Request (RREQ) message. This message includes a unique sequence number and a hop count, which is incremented each time the RREQ is forwarded by an intermediate node. Nodes receiving the RREQ compare the sequence number with the one stored in their routing table. If the received sequence number is higher, they update their routing table with the new information and continue forwarding the RREQ. If the sequence number is lower or equal, the message is discarded to prevent routing loops. When a node that has a valid route to the destination or is the destination itself receives the RREQ, it generates a Route Reply (RREP) message. This RREP is sent back to the source node along the reverse path established by the RREQ, ensuring that the route is loop-free and up-to-date.
2. **Route Maintenance**: When a link break in an active route is detected, a Route Error (RERR) message is broadcasted to inform all nodes about the broken link. Nodes receiving the RERR update their routing tables to remove any routes that use the broken link. Additionally, the source node may receive the RERR and can initiate a new route discovery process if it still needs a route to the destination.

> [!NOTE] Broken Link
> In the context of AODV, a broken link refers to a failure in the communication path between two nodes in the network. This can occur due to various reasons such as node mobility, interference, or physical obstructions. When a node detects that a link to a neighboring node is no longer valid, it considers this link as broken. The detection of a broken link triggers the generation of a Route Error (RERR) message, which is used to inform other nodes about the invalid link so they can update their routing tables accordingly.

## Python Implementation Outline

Implementing AODV in Python involves simulating the behavior of nodes in an ad-hoc network. Here is a simplified outline of how you might implement AODV:

1. **Define a Node class**: This class represents a node in the network. Each node has a unique ID, a list of neighbors, and a routing table storing destination nodes and the next hop to reach them. It handles sending RREQ, RREP, and RERR messages.
2. **Routing table**: Data structure storing routes to other nodes in the network (destination node, next hop node, sequence numbers, and cost of the route).
3. **Message handling**: Implement functions to handle RREQ, RREP, and RERR messages. When a node receives an RREQ, it checks if it has a route to the destination; if yes, it replies with an RREP; otherwise, it forwards the RREQ to neighbors. When receiving an RREP, it updates its routing table. When a broken link is detected, it broadcasts an RERR message.
4. **Network simulations**: Simulation environment where nodes communicate (graph representation using NetworkX, socket simulation, or abstract movement simulation).

### Pseudocode and Implementation

```python
class Node:
    def __init__(self, address):
        self.address = address
        self.routing_table = {}  # destination: (next_hop, sequence_number)

    def receive_rreq(self, rreq):
        # Process incoming RREQ
        pass

    def send_rreq(self, destination):
        # Broadcast RREQ
        pass

    def receive_rrep(self, rrep):
        # Process incoming RREP
        pass

    def send_rrep(self, destination):
        # Send RREP back to source
        pass

    def receive_rerr(self, rerr):
        # Process incoming RERR
        pass

    def send_rerr(self, destination):
        # Send RERR to inform of broken link
        pass

# Example usage
node_a = Node('A')
node_b = Node('B')

# Simulate sending a RREQ from A to B
node_a.send_rreq('B')
```

Routing loops occur when a packet is continuously passed through a cycle of nodes without reaching its destination. This can happen due to outdated routing information or incorrect sequence numbers. In AODV, sequence numbers help prevent routing loops by ensuring that nodes only update their routing tables with fresher routes.

### Considerations

- **Network Simulation**: Implementing a full network simulation can be complex. Existing network simulation frameworks or graph libraries help handle lower-level details.
- **Concurrency**: In a real network, nodes operate concurrently. Threading or asynchronous programming may be required.
- **Testing**: Test the implementation in various scenarios, such as dynamic topology changes, to ensure correct routing recovery.
