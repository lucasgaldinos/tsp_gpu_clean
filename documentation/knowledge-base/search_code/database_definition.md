# What should the database have

Suppose a VRP with a fleet of vehicles and a set of customers. Each vehicle has a capacity and each customer has a demand. The objective is to minimize the total distance traveled by the vehicles while delivering the goods to the customers.

## Customer

Each customer should have an ID, a name, a location (coordinates), and a demand (the total weight and volume of the boxes they need). For this, there can be maybe 100 customers(?) for test purposes.

## Box and products

Each box should have an ID, a size (weight and volume), and a customer ID (to indicate which customer it belongs to).  
Time in deposit may be another constraint. The longer the time, the lower(?) the cost for that item to be delivered.

## Vehicle

Something like 5 types of vehicles, with different capacities. Capacities can be both weight and volume. It should have the fuel consumption by distance of each vehicle. Other costs should also be included.

## Locations

Get city map and location points from OpenStreetMap or other.
