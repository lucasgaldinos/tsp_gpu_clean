# GPU Memory Calculation for Large TSP Problems

## Worst-Case Memory Analysis for d15112 (n=15,112)

### Memory Components Breakdown

#### 1. Distance Matrix (Persistent Storage)
```
Size: n × n × sizeof(float64)
d15112: 15,112 × 15,112 × 8 bytes = 1,826,434,816 bytes = 1,741.6 MB
```

#### 2. Working Buffers (The Real Bottleneck)

##### 2-opt Move Buffers:
```
Number of possible 2-opt moves: n × (n-1) / 2
d15112: 15,112 × 15,111 / 2 = 114,125,016 moves

Delta buffer (improvement values): 114,125,016 × 8 bytes = 913.0 MB
Index buffer (i,j pairs): 114,125,016 × 2 × 4 bytes = 913.0 MB  
Reduction workspace: 15,112 × 8 bytes = 0.1 MB

Total working buffers: 913.0 + 913.0 + 0.1 = 1,826.1 MB
```

#### 3. Additional GPU Overhead
```
Tour arrays (current + temp): 15,112 × 2 × 4 bytes = 0.1 MB
Kernel parameters and metadata: ~10 MB
GPU driver overhead: ~50-100 MB
```

### Total Memory Calculation
```
Distance Matrix:    1,741.6 MB
Working Buffers:    1,826.1 MB  
GPU Overhead:          110.0 MB
                    ____________
Subtotal:           3,677.7 MB

Safety Margin (20%):  735.5 MB
                    ____________
TOTAL REQUIRED:     4,413.2 MB = 4.31 GB
```

### GPU Capacity Analysis (GTX 1050 Mobile 4GB)
```
Total VRAM:         4,200 MB (4.1 GB usable)
Safe utilization:   4,200 × 0.65 = 2,730 MB (65% limit)

Required for d15112: 4,413.2 MB
Available safe:     2,730.0 MB
                   ____________
DEFICIT:           -1,683.2 MB ❌ UNSAFE
```

## Maximum Safe Problem Size Calculation

### Formula for Maximum n:
```
Working buffer constraint: n²/2 × 16 bytes ≤ Available_Memory

For 2,730 MB available:
n²/2 × 16 ≤ 2,730 × 10⁶
n² ≤ 341,250,000
n ≤ √341,250,000 ≈ 18,470

BUT distance matrix also needs space:
n² × 8 + n²/2 × 16 ≤ 2,730 × 10⁶
n² × (8 + 8) ≤ 2,730 × 10⁶  
n² × 16 ≤ 2,730 × 10⁶
n² ≤ 170,625,000
n ≤ √170,625,000 ≈ 13,062
```

### **MAXIMUM SAFE PROBLEM SIZE: n = 13,000 nodes**

### Validation for Common Problem Sizes:
```
n=1,000:  Total = 16.0 MB ✅ SAFE (0.6% of capacity)
n=5,000:  Total = 400.0 MB ✅ SAFE (14.7% of capacity)  
n=10,000: Total = 1,600.0 MB ✅ SAFE (58.6% of capacity)
n=13,000: Total = 2,704.0 MB ✅ SAFE (99.0% of capacity)
n=15,112: Total = 4,413.2 MB ❌ UNSAFE (161.6% of capacity)
```

### Memory Scaling Pattern:
```
Memory scales as O(n²) due to working buffers being dominant component.
Distance matrix: O(n²) but smaller coefficient (8 vs 16 bytes per n²)
Working buffers: O(n²) with larger coefficient (dominant term)
```

## Recommendations:

1. **Conservative Limit**: n ≤ 12,000 (90% of theoretical max)
2. **Testing Limit**: n ≤ 10,000 (safe for extended testing)  
3. **Production Limit**: Depends on GPU VRAM available

## Source of Calculation:
- Based on SA_COMPREHENSIVE_ANALYSIS.md Section Q6
- Validated against VRAMCalculator.analyze_problem() function
- Confirmed by actual GPU testing up to n=1,000

