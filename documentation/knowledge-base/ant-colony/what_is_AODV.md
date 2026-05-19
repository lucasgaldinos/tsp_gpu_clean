# AODV

## acronym list

RERR - Route Error Report
RREQ - Route Request
RREP - Route Reply
AODV - Ad-hoc On-demand Distance Vector Routing

## AODV: Ad-hoc On-demand Distance Vector Routing

AODV (Ad-hoc On-demand Distance Vector Routing) is a routing protocol for mobile ad hoc networks (MANETs). It is a reactive, on-demand, table-driven, and destination-sequenced distance-vector routing protocol. Proactive protocols, in contrast, maintain a complete list of routes all times.

Its purpose is for routing mobile networks where nodes are interconnected, communicating with each other without relying on a central infrastructure.

### Use case

When the arrangement or organization of network nodes, known as network topologies, can change frequently.  
Suited for smaller, dynamic networks.

### Key features

1. On-demand: Routes are created only when needed.
2. Route maintenance: Routes are maintained as long as they are needed by the source node. (? not sure)
3. Reactive: Nodes react to changes in the network by updating their routing tables.
4. Sequence Numbers: AODV uses sequence numbers to ensure the freshness of routes and to detect loops in the network. (? not sure)
5. Route Error Messages: When a link break in an active route is detected, a route error (RERR) message is broadcasted to inform other nodes about the broken link.

### Basic Operation

1. Route Discovery: When a node needs to send a packet to a destination node, it initiates a Route Request (RREQ) message. This message includes a unique sequence number and a hop count, which is incremented each time the RREQ is forwarded by an intermediate node. Nodes receiving the RREQ compare the sequence number with the one stored in their routing table. If the received sequence number is higher, they update their routing table with the new information and continue forwarding the RREQ. If the sequence number is lower or equal, the message is discarded to prevent routing loops. When a node that has a valid route to the destination or is the destination itself receives the RREQ, it generates a Route Reply (RREP) message. This RREP is sent back to the source node along the reverse path established by the RREQ, ensuring that the route is loop-free and up-to-date.
2. Route Maintenance: When a link break in an active route is detected, a Route Error (RERR) message is broadcasted to inform all nodes about the broken link. Nodes receiving the RERR update their routing tables to remove any routes that use the broken link. Additionally, the source node may receive the RERR and can initiate a new route discovery process if it still needs a route to the destination.

> :memo: Broken Link
>
> In the context of AODV, a broken link refers to a failure in the communication path between two nodes in the network. This can occur due to various reasons such as node mobility, interference, or physical obstructions. When a node detects that a link to a neighboring node is no longer valid, it considers this link as broken. The detection of a broken link triggers the generation of a Route Error (RERR) message, which is used to inform other nodes about the invalid link so they can update their routing tables accordingly.

## Python Implementation

Implementing AODV in Python involves simulating the behavior of nodes in an ad-hoc network. Here is a simplified outline of how you might implement AODV

 1. Define a Node class: This class will represent a node in the network. Each node will have a unique ID, a list of neighbors, and a routing table. The routing table will store information about the destination nodes and the next hop to reach them. It needs to handle sending RREQ, RREP and RERR messages.
 2. Routing table: Data structure to store the routes to other nodes in the network. It should include information such as the destination node, the next hop node, sequence numbers and the cost of the route.
 3. Message handling: Implement functions to handle RREQ, RREP and RERR messages. When a node receives a RREQ, it should check if it has a route to the destination. If it does, it should send a RREP message back to the source. If it doesn't, it should forward the RREQ to its neighbors. When a node receives a RREP, it should update its routing table with the new route. When a node detects a broken link, it should send a RERR message to its neighbors.
 4. Network simulations: Create a simulation environment where nodes can communicate with each other. This can be done by creating a list of nodes and simulating their movements and communication (more abstract way), using sockets for communication (more realistic way), or by using a library such as NetworkX to create a graph representation of the network.

**pseudocode**

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

Routing loops occur when a packet is continuously passed through a cycle of nodes without reaching its destination.  
This can happen due to outdated routing information or incorrect sequence numbers. In AODV, sequence numbers help prevent routing loops by ensuring that nodes only update their routing tables with fresher routes.

### Considerations

Network Simulation: Implementing a full network simulation can be complex. You might want to use existing network simulation frameworks or libraries to handle lower-level details.
Concurrency: In a real network, nodes operate concurrently. You may need to use threading or asynchronous programming to simulate this behavior.
Testing: Test your implementation in various scenarios, such as dynamic topology changes, to ensure it behaves correctly.
