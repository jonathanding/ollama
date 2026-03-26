# Fork 分支迁移实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 myollama 从"main 混杂自定义改动"迁移到"main 同步上游 + dev 活跃开发"的分支结构。

**Architecture:** 添加 upstream remote，从当前 main 创建 dev 分支保存所有自定义改动，然后 reset main 到上游 HEAD。迁移后 main 只用于 sync，dev 用于日常开发，pr/* 用于向上游提交 PR。

**Tech Stack:** Git

**Spec:** [`docs/superpowers/specs/2026-03-26-fork-branch-strategy-design.md`](../specs/2026-03-26-fork-branch-strategy-design.md)

---

## 前置状态确认

当前 repo 状态：
- Remote: 只有 `origin` → `github.com/jonathanding/ollama.git`
- 分支: `main`（当前，5259 commits）、`feature/profiling-tracing`（5208 commits，已合并进 main）
- 远程分支: 只有 `origin/main`（`feature/profiling-tracing` 没有远程分支）
- 远程 tags: `trace-ok`, `v0.2.0-trace-viz`, `v0.3.0-trace-replay`, `v0.3.1-trace-ui-polish`, `v0.4.0-release-infra`, `v0.4.1-release-ps1`, `v0.6.6-tracing-release-packaging`
- 未跟踪文件: `.claude/`, `CLAUDE.md`, `docs/` 下多个文件, `ollama-llama-notes.md`
- Stash: 无
- GitHub default branch: `main`

---

### Task 1: 添加 upstream remote 并 fetch

**前置条件:** 无

- [ ] **Step 1: 添加 upstream remote**

```bash
git remote add upstream https://github.com/ollama/ollama.git
```

- [ ] **Step 2: Fetch upstream**

```bash
git fetch upstream
```

Expected: 下载上游所有分支和 tags，输出大量 `* [new branch]` 和 `* [new tag]` 行。

- [ ] **Step 3: 验证 remote 配置**

```bash
git remote -v
```

Expected output（4 行）:
```
origin    https://github.com/jonathanding/ollama.git (fetch)
origin    https://github.com/jonathanding/ollama.git (push)
upstream  https://github.com/ollama/ollama.git (fetch)
upstream  https://github.com/ollama/ollama.git (push)
```

---

### Task 2: 创建 dev 分支并推送

**前置条件:** Task 1 完成

- [ ] **Step 1: 确认当前在 main 分支**

```bash
git branch --show-current
```

Expected: `main`

- [ ] **Step 2: 从当前 main 创建 dev 分支**

```bash
git checkout -b dev
```

Expected: `Switched to a new branch 'dev'`

此时 dev 和 main 指向同一个 commit（`96af3562`），包含所有 5259 个 commit（上游 + 60 个自定义）。

- [ ] **Step 3: 推送 dev 到 origin**

```bash
git push -u origin dev
```

Expected: 推送成功，输出包含 `* [new branch] dev -> dev` 和 `Branch 'dev' set up to track remote branch 'dev' from 'origin'`。

- [ ] **Step 4: 验证 dev 在 GitHub 上存在**

```bash
git ls-remote --heads origin dev
```

Expected: 输出一行，显示 `96af3562...` 指向 `refs/heads/dev`。

---

### Task 3: 重置 main 到上游

**前置条件:** Task 2 完成（dev 已安全推送到 origin）

⚠️ **此 task 包含破坏性操作（reset --hard + force push）。执行前务必确认 Task 2 的 Step 4 验证通过，即 dev 已安全存在于 GitHub。**

- [ ] **Step 1: 切回 main**

```bash
git checkout main
```

Expected: `Switched to branch 'main'`

注意：工作目录中的未跟踪文件（`.claude/`, `CLAUDE.md`, `docs/` 等）不会受影响。

- [ ] **Step 2: 确认 upstream/main 的 HEAD**

```bash
git log --oneline upstream/main -3
```

Expected: 显示上游 ollama 最近 3 个 commit。记下最新的 commit hash，用于后续验证。

- [ ] **Step 3: Reset main 到 upstream/main**

```bash
git reset --hard upstream/main
```

Expected: `HEAD is now at <upstream-head-hash> <upstream-head-message>`

- [ ] **Step 4: 验证 main 现在等于 upstream/main**

```bash
git log --oneline -3
```

Expected: 输出与 Step 2 完全一致。自定义的 60 个 commit 不再出现在 main 上。

- [ ] **Step 5: 验证未跟踪文件仍然存在**

```bash
git status --short | head -20
```

Expected: 未跟踪文件（`??` 前缀）仍在，可能还会出现一些被 reset 影响的 tracked 文件变化（因为上游 main 和之前的 main 有文件差异）。这是预期的——这些差异属于你的自定义改动，已经保存在 dev 分支上。

- [ ] **Step 6: Force push main 到 origin**

```bash
git push --force origin main
```

Expected: 推送成功，输出包含 `+ 96af3562...<upstream-hash> main -> main (forced update)`。

- [ ] **Step 7: 验证 GitHub 上的 main 已更新**

```bash
git log --oneline origin/main -3
```

Expected: 与 upstream/main 一致。

---

### Task 4: 配置 main 跟踪 upstream

**前置条件:** Task 3 完成

- [ ] **Step 1: 设置 main 跟踪 upstream/main**

```bash
git branch --set-upstream-to=upstream/main main
```

Expected: `Branch 'main' set up to track remote branch 'main' from 'upstream'.`

- [ ] **Step 2: 验证跟踪关系**

```bash
git branch -vv
```

Expected output 类似:
```
  dev    96af356 [origin/dev] docs: point pre-built release link...
* main   <hash>  [upstream/main] <upstream latest commit message>
```

关键点：`main` 后面显示 `[upstream/main]`，`dev` 后面显示 `[origin/dev]`。

---

### Task 5: 清理旧分支

**前置条件:** Task 4 完成

- [ ] **Step 1: 删除本地 feature/profiling-tracing 分支**

```bash
git branch -d feature/profiling-tracing
```

Expected: `Deleted branch feature/profiling-tracing (was 36214182).`

如果报错说分支未完全合并，使用 `-D`（大写）强制删除，因为该分支的内容已保存在 dev 中。

- [ ] **Step 2: 确认远程没有该分支（无需删除）**

```bash
git ls-remote --heads origin feature/profiling-tracing
```

Expected: 无输出（该分支从未推送到远程）。

- [ ] **Step 3: 验证最终分支列表**

```bash
git branch -a
```

Expected:
```
  dev
* main
  remotes/origin/HEAD -> origin/main
  remotes/origin/dev
  remotes/origin/main
  remotes/upstream/main
  (可能还有 upstream 的其他分支)
```

---

### Task 6: 处理未跟踪文件

**前置条件:** Task 5 完成

当前在 main 分支上有一批未跟踪文件（`.claude/`, `CLAUDE.md`, `docs/`, `ollama-llama-notes.md`）。这些文件属于你的自定义工作，应该提交到 dev 分支。

- [ ] **Step 1: 确认未跟踪文件列表**

```bash
git status --short
```

Expected: 列出所有 `??` 开头的未跟踪文件，以及可能因 reset 产生的已跟踪文件差异。

- [ ] **Step 2: 切到 dev 分支**

```bash
git checkout dev
```

Expected: `Switched to branch 'dev'`。未跟踪文件会跟着过来。

注意：如果 main reset 后工作目录中出现了 tracked 文件的修改（因为上游和你的 dev 有差异），切到 dev 时这些差异会消失（因为 dev 有这些文件的正确版本）。

- [ ] **Step 3: 将未跟踪文件加入 staging**

```bash
git add .claude/ CLAUDE.md docs/TODO.md docs/debugging-and-profiling.md docs/qwen3-cpu-offload-lock-analysis.md docs/superpowers/ docs/threading-analysis-report.md ollama-llama-notes.md
```

- [ ] **Step 4: 确认 staging 内容**

```bash
git status
```

Expected: 上述文件显示为 `new file` 在 staging 区。

- [ ] **Step 5: 提交**

```bash
git commit -m "docs: add project documentation and claude config"
```

- [ ] **Step 6: 推送 dev**

```bash
git push origin dev
```

---

### Task 7: 最终验证

**前置条件:** Task 6 完成

- [ ] **Step 1: 验证 main 与上游一致**

```bash
git checkout main && git diff upstream/main
```

Expected: 无输出（diff 为空），表示 main 与上游完全一致。

- [ ] **Step 2: 验证 dev 包含所有自定义改动**

```bash
git checkout dev && git log --oneline main..dev | wc -l
```

Expected: 数字 >= 61（60 个原有自定义 commit + 1 个刚提交的 docs commit）。

- [ ] **Step 3: 验证 remote 配置**

```bash
git remote -v
```

Expected:
```
origin    https://github.com/jonathanding/ollama.git (fetch)
origin    https://github.com/jonathanding/ollama.git (push)
upstream  https://github.com/ollama/ollama.git (fetch)
upstream  https://github.com/ollama/ollama.git (push)
```

- [ ] **Step 4: 验证分支跟踪关系**

```bash
git branch -vv
```

Expected: `main` 跟踪 `upstream/main`，`dev` 跟踪 `origin/dev`。

- [ ] **Step 5: 验证 GitHub 上 default branch 仍为 main**

```bash
git ls-remote --symref origin HEAD
```

Expected: 输出包含 `ref: refs/heads/main`。

- [ ] **Step 6: 切回 dev 作为日常工作分支**

```bash
git checkout dev
```

Expected: `Switched to branch 'dev'`

---

## 迁移后注意事项

1. **日常开发在 dev 上进行**，不在 main 上提交任何代码。
2. **同步上游**：`git checkout main && git pull`，然后 `git checkout dev && git rebase main && git push --force origin dev`。
3. **提 PR**：从 main 开 `pr/<功能名>` 分支，cherry-pick dev 上的 commit，push 后在 GitHub 创建 PR。
4. **完整工作流文档**：[`docs/superpowers/specs/2026-03-26-fork-branch-strategy-design.md`](../specs/2026-03-26-fork-branch-strategy-design.md)
