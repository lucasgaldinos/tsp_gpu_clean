---
name: antigravity-tools
description: "list of tools available for google's antigravity"
argument-hint: "Whenever generating an answer"
---

# antigravity's chat capabilities

## 🎨 Markdown Components (Special Formatting)

These are custom rendering features I use to present information clearly and interactively.

### 🚀 Custom "Markdown Tricks"

| Trick | Syntax / Description | Reference |
| :--- | :--- | :--- |
| **Carousel** | ```` ```carousel ```` with `<!-- slide -->`. Custom UI for sliding panels. | [Reference](file:///home/lucas_galdino/chimera/gpu_accelerated/.agents/skills/antigravity-tools/templates/markdown_tricks.md#L7) |
| **render_diffs** | `render_diffs(file_uri)`. Shorthand for visual diffs in artifacts. | [Reference](file:///home/lucas_galdino/chimera/gpu_accelerated/.agents/skills/antigravity-tools/templates/markdown_tricks.md#L35) |
| **Artifact Rendering** | Advanced UI for `.md` files in the artifact directory. | [Reference](file:///home/lucas_galdino/chimera/gpu_accelerated/.agents/skills/antigravity-tools/templates/markdown_tricks.md#L45) |
| **Active File Links** | `file:///` links that scroll your IDE to specific lines. | [Reference](file:///home/lucas_galdino/chimera/gpu_accelerated/.agents/skills/antigravity-tools/templates/markdown_tricks.md#L53) |
| **Walkthrough UI** | Special interactive template for multi-step code changes. | [Reference](file:///home/lucas_galdino/chimera/gpu_accelerated/.agents/skills/antigravity-tools/templates/markdown_tricks.md#L62) |

---

## 🛠️ Full Tool List (Capability Inventory)

These are the underlying functions I can execute. They are organized by the "Server" providing them.

### **1. Core System Tools (Default API)**

| Tool | Description |
| :--- | :--- |
| **`list_dir`** | Lists files and subdirectories in an absolute path. |
| **`view_file`** | Reads file content (supports text, images, and videos). |
| **`write_to_file`** | Creates new files and parent directories. |
| **`replace_file_content`** | Replaces a single contiguous block of text in a file. |
| **`multi_replace_file_content`** | Performs multiple non-contiguous edits in one file. |
| **`grep_search`** | High-performance text/regex search across directories (`ripgrep`). |
| **`run_command`** | Runs bash commands in a Linux environment. |
| **`command_status`** | Checks the status and output of background commands. |
| **`send_command_input`** | Sends STDIN or termination signals to active processes. |
| **`search_web`** | Performs a web search and returns summaries with citations. |
| **`read_url_content`** | Fetches and converts a URL's HTML to Markdown. |
| **`browser_subagent`** | An autonomous agent for complex web navigation and UI interaction. |
| **`generate_image`** | Creates or edits images based on descriptive prompts. |

### **2. GitHub MCP Server**

| Tool | Description |
| :--- | :--- |
| **`search_code`** | Precision search across all GitHub repositories. |
| **`search_issues`** / **`search_pull_requests`** | Filter and find GitHub issues or PRs. |
| **`search_repositories`** / **`search_users`** | Discover projects and contributors. |
| **`create_pull_request`** / **`update_pull_request`** | Manage the PR lifecycle. |
| **`push_files`** | Commits and pushes multiple files in a single action. |
| **`pull_request_review_write`** | Submit approvals, comments, or change requests. |
| **`sub_issue_write`** | Manage hierarchical sub-issues. |

### **3. Knowledge & Thinking Tools**

| Tool | Description |
| :--- | :--- |
| **`sequential-thinking`** | Used for deep, multi-step planning and reflection. |
| **`memory`** | A suite of 9 tools to manage a Knowledge Graph of our shared context. |
| **`fetch`** | A specialized tool for retrieving raw or simplified web content. |

### **4. Meta Tools**

| Tool | Description |
| :--- | :--- |
| **`list_resources`** / **`read_resource`** | Access data streams provided by MCP servers. |

---

## **"Is this everything?"**

1. **Scratch Directory**: antigravity has a dedicated space to store temporary scripts or data files that persist across turns: `<appDataDir>/brain/<conversation-id>/scratch/`.
2. **Persistent Context**: antigravity "remember" things by creating Knowledge Items (KIs) or reading past conversation logs, which allows it to maintain context over very long projects.
