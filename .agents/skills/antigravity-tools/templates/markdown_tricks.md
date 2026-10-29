# Antigravity Markdown Tricks Reference

This file serves as a template and technical reference for the custom Markdown components supported in this environment.

---

## 1. Carousel

The Carousel allows for multiple sliding panels of content. It is ideal for comparing alternative implementations or showing step-by-step UI progressions.

### Usage

````markdown
```carousel
Slide 1 content (Text, Tables, Images, etc.)
<!-- slide -->
Slide 2 content
<!-- slide -->
Slide 3 content
```
````

### Example

```carousel
### Phase 1: Planning
Initial thought process and design.
<!-- slide -->
### Phase 2: Implementation
The actual code changes.
<!-- slide -->
### Phase 3: Verification
Logs and test results.
```

---

## 2. render_diffs

The `render_diffs` macro is a powerful shorthand that automatically generates visual diffs for modified files. **Note: This only works when placed inside an Artifact.**

### Usage

```markdown
render_diffs(file:///absolute/path/to/your/file.ext)
```

---

## 3. Artifact Rendering

Any Markdown file written to the system-managed artifacts directory is automatically treated as a "Rich Artifact." These files gain a special UI with a navigation sidebar, action buttons (like "Apply" for diffs), and a clean, documentation-style layout.

### Usage

I trigger this automatically when I use `write_to_file` with `IsArtifact: true` or when saving to the project's artifact directory.

---

## 4. Active File Links

Standard file URI syntax is enhanced in this environment to interact directly with your IDE.

### Usage

- **Open File**: `[Link Text](file:///path/to/file.ext)`
- **Jump to Lines**: `[Link Text](file:///path/to/file.ext#L50-L100)`

---

## 5. Walkthrough UI

A "Walkthrough" is a special type of artifact (usually named `walkthrough.md`) designed to guide you through complex changes. It renders diffs as interactive steps that you can verify and apply individually.

### Usage

The file must contain standard Markdown headings and `diff` blocks. The system recognizes the structure and provides "Apply Change" buttons for each block.

```markdown
# Implementation Walkthrough

## Step 1: Update Config
```diff
- old_value: false
+ old_value: true
```

```
