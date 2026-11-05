---
title: "Taxonomy of Variables in Complex Systems and Statistical Optimization"
description: >
  Conceptual taxonomy and mathematical characterization of variable classifications
  in systems modeling, control theory, fluid mechanics, and combinatorial optimization:
  state, control, decision, algebraic, exogenous, parameters, output, and auxiliary variables.
created: 2026-11-05
status: final
author:
  - "[[Lucas Galdino]]"
  - "[[Antigravity]]"
type: notes
scope: global
modifications:
  - date_modified: 2026-11-05
    modifications:
      - description: >
          Standardized variable taxonomy note adhering to repository markdown guidelines,
          adding compliant YAML frontmatter, mathematical equations, cross-domain engineering
          examples, and linking to optimization problem formulations.
related_files:
  - "[guide-statistical-testing-index.md](./guide-statistical-testing-index.md)"
  - "[analysis-complete-statistical-results.md](./analysis-complete-statistical-results.md)"
  - "[problem_context.py](../../src/protocols/problem_context.py)"
tags:
  - notes/theory
  - notes/mathematics
  - notes/architecture
  - analysis/statistical
  - algorithm/tsp
  - benchmark/statistical-tests
  - guide/analysis
  - review/quality
  - notes/vram
  - notes/hardware
---

# Taxonomy of Variables in Complex Systems and Statistical Optimization

- [Taxonomy of Variables in Complex Systems and Statistical Optimization](#taxonomy-of-variables-in-complex-systems-and-statistical-optimization)
  - [1. Fundamental Concepts of System Variables](#1-fundamental-concepts-of-system-variables)
  - [2. Taxonomy of Variable Types](#2-taxonomy-of-variable-types)
    - [2.1 State Variables](#21-state-variables)
    - [2.2 Control Variables](#22-control-variables)
    - [2.3 Decision Variables](#23-decision-variables)
    - [2.4 Algebraic Variables](#24-algebraic-variables)
    - [2.5 Exogenous Variables](#25-exogenous-variables)
    - [2.6 Parameters](#26-parameters)
    - [2.7 Output Variables](#27-output-variables)
    - [2.8 Auxiliary Variables](#28-auxiliary-variables)
  - [3. Key Inter-Variable Relationships and Matrix Mappings](#3-key-inter-variable-relationships-and-matrix-mappings)
  - [4. Cross-Domain Concrete Engineering Examples](#4-cross-domain-concrete-engineering-examples)
    - [4.1 Simple Harmonic Pendulum](#41-simple-harmonic-pendulum)
    - [4.2 Macroeconomic Dynamic System](#42-macroeconomic-dynamic-system)
    - [4.3 Electrical Power Distribution Grid](#43-electrical-power-distribution-grid)
    - [4.4 Supply Chain Logistics Network](#44-supply-chain-logistics-network)
    - [4.5 Climate System Model](#45-climate-system-model)
    - [4.6 Autonomous Vehicle Cruise Control](#46-autonomous-vehicle-cruise-control)
    - [4.7 Fluid Mechanics: Pipe Flow Under Navier-Stokes](#47-fluid-mechanics-pipe-flow-under-navier-stokes)
  - [5. Application to Combinatorial GPU Routing Optimization](#5-application-to-combinatorial-gpu-routing-optimization)
  - [6. Academic and Technical References](#6-academic-and-technical-references)

---

## 1. Fundamental Concepts of System Variables

In dynamical systems theory, control engineering, and mathematical optimization, **variables** represent quantifiable attributes or states that evolve over time or respond to operational stimuli. 

A rigorous classification of variables is essential for:
1. **Mathematical Well-Posedness**: Ensuring that degrees of freedom match the number of governing differential-algebraic equations (DAEs).
2. **Computational Architecture**: Discerning immutable data (parameters) stored in GPU constant or global cache from volatile trajectory states stored in registers or shared memory.
3. **Statistical Inference**: Distinguishing controlled treatment factors from random block effects and exogenous environmental noise.

---

## 2. Taxonomy of Variable Types

### 2.1 State Variables ($\mathbf{x} \in \mathbb{R}^{n_x}$)

State variables represent the minimal set of internal physical quantities that uniquely define the dynamical condition of a system at any given instant $t$. Knowledge of the state vector $\mathbf{x}(t_0)$ together with input trajectories for $t \ge t_0$ completely determines the future trajectory $\mathbf{x}(t)$.

- **Evolution**: Governed by differential or difference state transition equations:
  $$\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = \mathbf{f}(\mathbf{x}(t), \mathbf{u}(t), \boldsymbol{\theta}, t)$$
- **Examples**: Electric charge on a capacitor $q(t)$, kinetic velocity of a vehicle $v(t)$, population genotype frequency vector $\mathbf{p}(t)$.

### 2.2 Control Variables ($\mathbf{u} \in \mathbb{R}^{n_u}$)

Control variables are independent exogenous inputs actively manipulated by an external operator, controller, or algorithm to steer the state trajectory toward an optimal state or objective target.

- **Role**: Free actuators constrained by physical envelopes ($\mathbf{u}_{\text{min}} \le \mathbf{u}(t) \le \mathbf{u}_{\text{max}}$).
- **Examples**: Throttle actuator angle $\theta_{\text{thr}}(t)$, pump rotational frequency $\omega_{\text{pump}}$, mutation rate $p_m$ in an adaptive genetic algorithm.

### 2.3 Decision Variables ($\mathbf{w} \in \mathcal{W}$)

Decision variables are the explicit unknowns solved for in static or combinatorial optimization problems. In optimal control, discrete trajectories of control variables become decision variables.

- **Role**: Determine the configuration of a solution to minimize an objective function $f(\mathbf{w})$ subject to feasibility constraints $\mathbf{g}(\mathbf{w}) \le \mathbf{0}$.
- **Examples**: Binary edge selections in a Traveling Salesperson Problem permutation matrix $\mathbf{X} \in \{0, 1\}^{n \times n}$, capital allocation amounts in portfolio optimization.

### 2.4 Algebraic Variables ($\mathbf{y}_a \in \mathbb{R}^{n_a}$)

Algebraic variables are instantaneous dependent quantities whose values are uniquely determined by static algebraic equations relating state, control, and parameter variables, without independent derivative terms.

- **Equation**:
  $$\mathbf{g}_a(\mathbf{x}(t), \mathbf{u}(t), \mathbf{y}_a(t), \boldsymbol{\theta}) = \mathbf{0}$$
- **Examples**: Instantaneous mechanical shaft power $P = \tau \omega$, Reynolds number $Re = \frac{\rho v D}{\mu}$, ohmic current $I = V / R$.

### 2.5 Exogenous Variables ($\mathbf{d} \in \mathbb{R}^{n_d}$)

Exogenous (environmental or disturbance) variables are external inputs that impinge upon the system dynamics but are fundamentally outside the controller's influence.

- **Role**: Modeled as deterministic time series or stochastic processes ($\mathbf{d}(t) \sim \mathcal{GP}(\mathbf{0}, \mathbf{K})$).
- **Examples**: Ambient road gradient $\theta_{\text{road}}(t)$, wind turbulence vector $\mathbf{v}_{\text{wind}}$, market demand shock $D(t)$.

### 2.6 Parameters ($\boldsymbol{\theta} \in \Theta$)

Parameters are invariant characteristics that specify the physical, geometric, or constitutive properties of a specific system instance. They do not evolve during a simulation run or optimization episode.

- **Role**: Model constants whose sensitivity is evaluated via parametric studies or post-optimality analysis ($\frac{\partial f^*}{\partial \boldsymbol{\theta}}$).
- **Examples**: Gravitational acceleration constant $g = 9.80665\text{ m/s}^2$, pipe internal roughness $\varepsilon$, city coordinate matrix $\mathbf{C} \in \mathbb{R}^{n \times 2}$ in TSP.

### 2.7 Output Variables ($\mathbf{y} \in \mathbb{R}^{n_y}$)

Output variables are the measurable, observable signals extracted from the system state and control inputs:

$$\mathbf{y}(t) = \mathbf{h}(\mathbf{x}(t), \mathbf{u}(t), \boldsymbol{\theta}, t) + \boldsymbol{\eta}(t)$$

where $\boldsymbol{\eta}(t)$ represents sensor measurement noise.

- **Examples**: Speedometer readout on an instrument dashboard, wall-clock time elapsed per generation $t_{\text{gen}}$, total tour length $L(\pi)$.

### 2.8 Auxiliary Variables ($\mathbf{z} \in \mathbb{R}^{n_z}$)

Auxiliary variables are artificial entities introduced to convexify non-linear constraints, linearize objective formulations, or decouple dense computation graphs.

- **Examples**: Slack variables $s_i \ge 0$ in linear programming inequality conversion ($g_i(\mathbf{w}) + s_i = 0$), Miller-Tucker-Zemlin (MTZ) sub-tour elimination potential variables $u_i \in \mathbb{R}$ in integer programming formulations.

---

## 3. Key Inter-Variable Relationships and Matrix Mappings

The table below summarizes the operational contrast between these classes:

| Variable Class | Manipulability | Dynamics | Role in System Architecture | GPU Memory Placement |
| :--- | :--- | :--- | :--- | :--- |
| **State ($\mathbf{x}$)** | Indirect (via $\mathbf{u}$) | Dynamic ($\dot{\mathbf{x}} = \mathbf{f}$) | Internal memory of past trajectory | VRAM Global / Shared Memory |
| **Control ($\mathbf{u}$)** | Direct | Dynamic/Discrete | Actuation drive towards objective | Registers / Kernel Parameters |
| **Decision ($\mathbf{w}$)** | Direct Optimization | Algorithmic steps | Direct solution representation | Device Memory Buffers |
| **Algebraic ($\mathbf{y}_a$)** | Dependent | Instantaneous | Derived intermediate relationships | Register File |
| **Exogenous ($\mathbf{d}$)** | Uncontrolled | Autonomous | Disturbance / Environmental boundary | Global Memory (Read-Only) |
| **Parameter ($\boldsymbol{\theta}$)** | Invariant | Static | System identity / Physical geometry | Constant Cache (`__constant__`) |
| **Output ($\mathbf{y}$)** | Observable | Measured | Performance verification / Telemetry | Host Pinned Memory Buffer |
| **Auxiliary ($\mathbf{z}$)** | Algorithmic | Auxiliary | Mathematical reformulation / Slack | Scratchpad VRAM Buffer |

---

## 4. Cross-Domain Concrete Engineering Examples

### 4.1 Simple Harmonic Pendulum

For a rigid pendulum of length $L$ and bob mass $m$ undergoing planar oscillation:

- **State Variables**: Angular displacement $\theta(t)$ and angular velocity $\omega(t) = \dot{\theta}(t)$.
- **Control Variable**: External applied torque $\tau(t)$.
- **Exogenous Variable**: Ambient aerodynamic damping disturbance.
- **Parameters**: Rod length $L$, bob mass $m$, gravitational constant $g$.
- **Algebraic Variable**: Tangential gravitational restoring force $F_t = -mg \sin \theta$.
- **Output Variable**: Optical sensor position $y(t) = L \sin \theta(t)$.
- **Governing Equation**:
  $$\frac{\mathrm{d}}{\mathrm{d}t} \begin{bmatrix} \theta \\ \omega \end{bmatrix} = \begin{bmatrix} \omega \\ -\frac{g}{L}\sin\theta + \frac{\tau(t)}{m L^2} \end{bmatrix}$$

### 4.2 Macroeconomic Dynamic System

- **State Variables**: Gross Domestic Product $Y(t)$, total capital stock $K(t)$, price inflation rate $\pi(t)$.
- **Control Variables**: Central bank policy interest rate $i(t)$, government expenditure $G(t)$.
- **Exogenous Variables**: Global supply chain shocks, raw commodity import pricing.
- **Parameters**: Marginal propensity to consume $c_1$, capital depreciation coefficient $\delta_K$.
- **Output Variable**: Consumer Price Index (CPI) quarterly headline report.

### 4.3 Electrical Power Distribution Grid

- **State Variables**: Bus voltage magnitudes $|V_k|$ and phase angles $\delta_k$.
- **Control Variables**: Generator active power setpoints $P_{\text{gen},k}$, shunt capacitor bank switching states.
- **Exogenous Variables**: Aggregated consumer load demand fluctuations $P_{\text{load},k}(t)$.
- **Parameters**: Line impedance matrices $\mathbf{Z} = \mathbf{R} + j\mathbf{X}$, transformer tap ratios $a_{ij}$.
- **Algebraic Variables**: Power balance equations derived from Kirchhoff's Current Law:
  $$P_k - \sum_{m} |V_k| |V_m| (G_{km}\cos\delta_{km} + B_{km}\sin\delta_{km}) = 0$$

### 4.4 Supply Chain Logistics Network

- **State Variables**: Warehouse on-hand inventory levels $I_j(t)$, in-transit freight quantities.
- **Control Variables**: Factory replenishment production rates, inter-depot shipping dispatch schedules.
- **Exogenous Variables**: Stochastic daily retail order arrivals.
- **Parameters**: Depot storage capacities $C_j$, unit holding costs $h_j$, delivery transit lead times $\tau_{ij}$.
- **Output Variable**: Fleet on-time delivery percentage, order fill rate.

### 4.5 Climate System Model

- **State Variables**: Global sea surface temperature anomalies $T_{\text{ocean}}(t)$, atmospheric carbon mass $M_{\text{CO}_2}(t)$.
- **Control Variables**: Anthropogenic emission abatement policies, solar geoengineering injection rate.
- **Exogenous Variables**: Solar irradiance solar-cycle variations, volcanic aerosol injection volumes.
- **Parameters**: Planetary Stefan-Boltzmann constant $\sigma_{\text{SB}}$, planetary albedo $\alpha_p$, climate sensitivity parameter $S$.
- **Output Variable**: Mean global surface air temperature departure $\Delta T_{\text{surface}}$.

### 4.6 Autonomous Vehicle Cruise Control

- **State Variables**: Vehicle longitudinal position $s(t)$, velocity $v(t)$, engine shaft rotation speed $\omega_{\text{eng}}(t)$.
- **Control Variable**: Fuel throttle valve command $u_{\text{throttle}}(t) \in [0, 1]$.
- **Exogenous Variables**: Highway incline slope angle $\alpha_{\text{road}}(t)$, headwind airspeed $v_{\text{wind}}(t)$.
- **Parameters**: Total vehicle mass $M_{\text{veh}}$, aerodynamic drag coefficient $C_d$, tire rolling friction $C_{rr}$.
- **Algebraic Variable**: Net aerodynamic drag force:
  $$F_{\text{aero}} = \frac{1}{2} \rho_{\text{air}} C_d A_{\text{frontal}} (v + v_{\text{wind}})^2$$
- **Output Variable**: Speedometer display digital telemetry.

### 4.7 Fluid Mechanics: Pipe Flow Under Navier-Stokes

Consider incompressible Newtonian fluid flow through a pressurized conduit:

1. **State Fields**:
   - Velocity vector field $\mathbf{v}(\mathbf{x}, t) = [u, v, w]^T$.
   - Hydrostatic pressure distribution $p(\mathbf{x}, t)$.
2. **Control Inputs**:
   - Upstream variable frequency drive pump pressure head $H_{\text{pump}}(t)$.
   - Downstream throttling valve aperture percentage $\theta_v(t)$.
3. **Exogenous Disturbance**:
   - Ambient thermal gradient influencing fluid bulk temperature $T_{\infty}$.
4. **Parameters**:
   - Fluid dynamic viscosity $\mu$, fluid density $\rho_0$, conduit length $L$, diameter profile $D(x)$, wall roughness height $\varepsilon$.
5. **Algebraic Non-Dimensional Variables**:
   - Reynolds Number:
     $$Re = \frac{\rho_0 \bar{v} D}{\mu}$$
   - Darcy-Weisbach Friction Factor $f_D$ via Colebrook-White implicit relation:
     $$\frac{1}{\sqrt{f_D}} = -2 \log_{10} \left( \frac{\varepsilon}{3.7 D} + \frac{2.51}{Re \sqrt{f_D}} \right)$$
6. **Governing Navier-Stokes PDE System**:
   $$\nabla \cdot \mathbf{v} = 0$$
   $$\rho_0 \left( \frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} \right) = -\nabla p + \mu \nabla^2 \mathbf{v} + \rho_0 \mathbf{g}$$

---

## 5. Application to Combinatorial GPU Routing Optimization

Mapping this variable taxonomy directly into the `gpu_accelerated_clean` codebase:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GPU Routing Solver: Variable Classification Architecture                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. PARAMETERS (Immutable Problem Definition):                                          │
│    - Node count n, coordinates (X, Y) in constant/texture cache                        │
│    - Static Distance Matrix D \in R^{n x n} in VRAM Global Memory                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. DECISION / STATE VARIABLES (Solution Trajectories):                                 │
│    - Permutation Tours \pi \in S_n (Shape: [P, n] uint16) in Scratchpad VRAM           │
│    - Generation Counter g \in [1, G_max] in GPU block register                         │
│    - Cost Vector f(\pi) \in R^P in device memory                                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. CONTROL VARIABLES (Metaheuristic Hyperparameters):                                  │
│    - Population size P, Mutation rate p_m, Tournament size k, 2-Opt iteration cap K_max│
│    - CUDA Grid launch configuration (<<< gridDim, blockDim, smem >>>)                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. ALGEBRAIC VARIABLES (Instantaneous Metrics):                                        │
│    - Delta 2-opt edge swap gain: \Delta = (D_{i,j} + D_{i+1,j+1}) - (D_{i,i+1} + D_{j,j+1})│
│    - Normalized population diversity metric: Div(P_t) = 1 - SharedEdges / (n * P)      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. OUTPUT VARIABLES (Experimental Observations):                                       │
│    - Best-so-far tour cost f*, wall-clock execution time t_wall, GPU speedup S_GPU     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

By strictly categorizing every variable, the solver avoids memory aliasing, minimizes register spilling, and ensures deterministic, reproducible benchmark evaluation.

---

## 6. Academic and Technical References

1. **Luenberger, D. G. (1979).** *Introduction to Dynamic Systems: Theory, Models, and Applications*. John Wiley & Sons, New York.
2. **Stengel, R. F. (1994).** *Optimal Control and Estimation*. Dover Publications, New York.
3. **White, F. M. (2015).** *Fluid Mechanics* (8th ed.). McGraw-Hill Education, New York.
4. **Bertsekas, D. P. (2016).** *Nonlinear Programming* (3rd ed.). Athena Scientific, Belmont, MA.
5. **Crainic, T. G., & Toulouse, M. (2003).** Parallel Strategies for Meta-heuristics. In F. Glover & G. A. Kochenberger (Eds.), *Handbook of Metaheuristics* (pp. 475–513). Springer, Boston, MA. DOI: [10.1007/0-306-48056-5_17](https://doi.org/10.1007/0-306-48056-5_17).
