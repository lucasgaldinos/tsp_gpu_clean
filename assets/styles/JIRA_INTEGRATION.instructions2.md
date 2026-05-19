---
title: "JIRA Project PT Integration Requirements"
applyTo: "project_management/**,code/**"
enforced: true
project_key: "PT"
jira_url: "https://this-shall-not-be-taken.atlassian.net"
board_id: "67"
---

# JIRA Project PT Integration Requirements

## MANDATORY JIRA INTEGRATION

### Project Configuration

- **Project Key**: PT (HARDCODED)
- **JIRA Instance**: <https://this-shall-not-be-taken.atlassian.net>
- **Board ID**: 67
- **Project Type**: TCC Academic Research

### Required Custom Fields (MUST BE CONFIGURED)

#### Academic Phase (customfield_10001)

**Type**: Select (Single Choice)
**Options**:

- Literature Review
- Methodology Design
- Implementation
- Experimentation
- Analysis
- Documentation
- Defense Preparation

#### Learning Objective (customfield_10002)

**Type**: Text Field
**Required**: Yes for all Story and Task issues
**Validation**: Minimum 10 characters

#### Algorithm Type (customfield_10003)

**Type**: Select (Single Choice)
**Options**:

- Classical TSP
- Classical VRP
- GPU Optimization
- Metaheuristic
- Exact Algorithm
- Approximation
- Hybrid Approach

#### Performance Target (customfield_10004)

**Type**: Text Field
**Format**: Must include numeric target (e.g., "5x speedup", "O(n²) complexity")

#### RACI Role (customfield_10005)

**Type**: Select (Single Choice)
**Options**:

- Responsible
- Accountable
- Consulted
- Informed

#### Complexity Level (customfield_10006)

**Type**: Select (Single Choice)
**Options**:

- Basic
- Intermediate
- Advanced
- Research

#### Testing Requirement (customfield_10007)

**Type**: Multi-select
**Options**:

- Unit Tests
- Integration Tests
- Performance Benchmarks
- Academic Validation
- Literature Comparison
- Statistical Significance

### Mandatory Issue Types Hierarchy

1. **Theme** (Level 1) - Project themes
2. **Epic** (Level 2) - Academic milestones
3. **Story** (Level 3) - Deliverable features
4. **Task** (Level 4) - Development work
5. **Sub-task** (Level 5) - Atomic work units

### Academic Development Workflow (ENFORCED)

**Workflow Name**: Academic Development
**Statuses**:

1. Literature Review → Design
2. Design → Implementation  
3. Implementation → Testing
4. Testing → Analysis
5. Analysis → Documentation
6. Documentation → Review
7. Review → Complete
8. Review → Implementation (rework)

### Branch Naming Convention (MANDATORY)

```
{type}/PT-{issue_number}-{brief-description}

Examples:
feature/PT-123-implement-nearest-neighbor
bugfix/PT-124-fix-memory-leak
research/PT-125-analyze-gpu-performance
docs/PT-126-update-api-documentation
```

### Commit Message Format (ENFORCED)

```
PT-{issue_number}: {Brief description}

Examples:
PT-123: Implement CuPy backend for distance provider
PT-124: Fix memory leak in GPU allocation
PT-125: Add performance benchmarking for nearest neighbor
```

### Pull Request Requirements

- **Title**: `[PT-{issue_number}] {Description}`
- **Description**: Must link to JIRA issue
- **Labels**: Must include issue type and complexity level
- **Reviewers**: Must include academic supervisor

## AUTOMATED VALIDATION RULES

### Issue Creation Validation

```python
# All issues must have:
required_fields = [
    "Academic Phase",
    "Learning Objective", 
    "Algorithm Type",
    "Complexity Level",
    "Testing Requirement"
]

# Performance-related issues must have:
performance_fields = [
    "Performance Target"
]

# Research-level issues must have:
research_fields = [
    "Literature Context",
    "Research Question",
    "Methodology"
]
```

### Git Integration Validation

```bash
# Pre-commit hook validation
if ! [[ "$BRANCH_NAME" =~ ^(feature|bugfix|research|docs)/PT-[0-9]+-.*$ ]]; then
    echo "❌ Branch name must follow pattern: {type}/PT-{number}-{description}"
    exit 1
fi

if ! [[ "$COMMIT_MSG" =~ ^PT-[0-9]+:.*$ ]]; then
    echo "❌ Commit message must start with PT-{number}:"
    exit 1
fi
```

## JIRA API INTEGRATION

### Authentication Setup

```bash
export JIRA_USER="your-email@example.com"
export JIRA_TOKEN="your-api-token"
export JIRA_URL="https://this-shall-not-be-taken.atlassian.net"
export PROJECT_KEY="PT"
```

### Automated Issue Creation Scripts

```bash
# Create story issue
./scripts/create_jira_issue.py \
    --type "Story" \
    --summary "Implement GPU-Accelerated Nearest Neighbor" \
    --academic-phase "Implementation" \
    --algorithm-type "Classical TSP" \
    --complexity "Intermediate"
```

### Progress Tracking Queries

```jql
# Current sprint work
project = PT AND sprint in openSprints() ORDER BY priority DESC

# Academic phase progress
project = PT AND "Academic Phase" = "Implementation" AND assignee = currentUser()

# High-priority research tasks
project = PT AND "Complexity Level" = "Research" AND priority = High

# Performance optimization work
project = PT AND "Algorithm Type" = "GPU Optimization" AND status != Done
```

## ACADEMIC COMPLIANCE REQUIREMENTS

### Issue Documentation Standards

Every issue must include:

- **Learning Objective**: Clear statement of knowledge gained
- **Academic Context**: How it fits into TCC research
- **Validation Method**: How success will be measured
- **Literature References**: Related academic sources

### Testing Integration

```python
# All implementation issues must have:
test_requirements = {
    "Unit Tests": "pytest coverage >90%",
    "Integration Tests": "Backend compatibility validation", 
    "Performance Benchmarks": "Statistical comparison with baselines",
    "Academic Validation": "Correctness on known problem instances"
}
```

### Review Requirements

- **Code Review**: Minimum 2 approvals for complex algorithms
- **Academic Review**: Supervisor approval for research-level work
- **Performance Review**: Benchmark validation for optimization claims
- **Documentation Review**: Academic writing standards compliance

## BOARD CONFIGURATION

### Kanban Columns (ENFORCED)

1. **Backlog** (Literature Review, Methodology Design)
2. **Ready for Development** (Design complete, ready for Implementation)
3. **In Development** (Implementation in progress)
4. **Testing** (Testing and validation)
5. **Analysis** (Performance analysis, results evaluation)
6. **Documentation** (Academic writing and documentation)
7. **Review** (Peer/supervisor review)
8. **Done** (Complete and validated)

### Swimlanes Configuration

- **By Algorithm Type**: Classical TSP, Classical VRP, GPU Optimization
- **By Complexity**: Basic, Intermediate, Advanced, Research
- **By Academic Phase**: Implementation, Experimentation, Analysis

## REPORTING AND METRICS

### Academic Progress Dashboard

- Issues by Academic Phase
- Completion rate by complexity level
- Performance targets achieved vs. planned
- Literature references and citations tracking

### Development Metrics

- Code coverage by algorithm type
- Performance improvement trends
- Bug resolution time by complexity
- Academic review cycle time

### Sprint Planning Metrics

- Story points by complexity level
- Academic milestone alignment
- Research vs. implementation balance
- Learning objective achievement rate

## INTEGRATION SCRIPTS

### Setup Script

```bash
# Run this to configure JIRA integration
./project_management/jira_templates/configure_pt_project.sh
```

### Daily Automation

```bash
# Update issue status based on Git activity
./scripts/sync_git_to_jira.py

# Generate academic progress report
./scripts/generate_academic_report.py

# Validate academic compliance
./scripts/validate_academic_standards.py
```

### Sprint Review Automation

```bash
# Generate sprint retrospective
./scripts/generate_sprint_review.py --sprint="Sprint 1"

# Export academic metrics
./scripts/export_academic_metrics.py --format=latex
```

---

**These JIRA integration requirements are MANDATORY and automatically enforced through validation scripts and API integration.**
