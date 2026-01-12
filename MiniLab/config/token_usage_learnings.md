# MiniLab Token Usage Learnings

**Last Updated:** 2026-01-08 15:27
**Total Completed Runs:** 32
**Total Incomplete Runs:** 0
**Total Tokens All Time:** 4,317,277

---

## How Agents Should Use This Document

This document provides historical token usage data to help you plan work.
Use these estimates as **guidance** for self-regulation, not hard limits.

**Key principles:**
- Check remaining budget frequently during work
- Prioritize core deliverables when budget is constrained
- Tool calls (especially terminal output) can be expensive
- Literature review and analysis are typically the most token-intensive

---

## Module Statistics (Completed Runs)

| Module | Runs | Mean | Min | Max | Std | Incomplete | Reliability |
|--------|------|------|-----|-----|-----|------------|-------------|
| phase4.task.feature_engineering.execute_analysis | 1 | 479,403 | 479,403 | 479,403 | ±0 | 0 | 🔴 Low |
| execute_analysis | 9 | 323,294 | 20,000 | 608,307 | ±239,281 | 0 | 🟢 High |
| phase3.planning_committee | 3 | 206,578 | 201,098 | 211,834 | ±5,371 | 0 | 🟡 Medium |
| phase4.task.statistical_design.execute_analysis | 1 | 196,871 | 196,871 | 196,871 | ±0 | 0 | 🔴 Low |
| task1 | 1 | 50,000 | 50,000 | 50,000 | ±0 | 0 | 🔴 Low |
| consultation | 13 | 4,177 | 0 | 5,602 | ±1,587 | 0 | 🟢 High |
| phase2.consultation | 4 | 1,826 | 1,733 | 2,021 | ±135 | 0 | 🟡 Medium |
| phase4.task.univariate_analysis.execute_analysis | 0 | 0 | 0 | 0 | ±0 | 0 | 🔴 Low |

---

## Planning Recommendations

### Execute Analysis
- **Recommended budget:** 801,858 tokens (mean + 2σ)
- **Observed range:** 20,000 – 608,307
- **Budget exhaustions:** 0
- **Note:** Highly variable based on data complexity

---

## Token Optimization Tips

### High-Cost Operations
- **Terminal output:** Commands that print large outputs consume many tokens
  - Use `head -50`, `tail -50`, or `grep` to limit output
  - Redirect verbose output to files
  - Check size first: `wc -l file.txt`

- **File reading:** Reading entire large files is expensive
  - Use `head` to preview structure
  - Read specific line ranges when possible

- **Literature searches:** Multiple search queries add up
  - Plan queries before executing
  - Use specific, targeted queries

### Low-Cost Operations
- Creating directories and writing files
- Targeted searches with specific queries
- Concise agent consultations

---

## Complexity Adjustments

| Complexity | Multiplier | When to Use |
|------------|------------|-------------|
| Simple | 0.7x | Single file, clear objective |
| Medium | 1.0x | Multiple files, some iteration |
| Complex | 1.3x | Multi-modal data, novel analysis |

---

*This document is auto-generated after each run.*
*Statistics improve with more completed runs.*