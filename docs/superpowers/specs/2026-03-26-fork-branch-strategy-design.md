# Fork 分支策略与迁移设计

> 日期: 2026-03-26
> 状态: 已批准

## 背景

myollama 是从 ollama/ollama fork 而来。当前所有自定义改动（profiler、trace-analyzer 等约 60 个 commit）直接提交在 main 分支上，导致：

- 无法干净地同步上游更新（main 上混杂了自定义代码和上游代码）
- 无法向上游提交 PR（难以隔离出只包含目标功能的 diff）

## 目标状态

### 分支结构

| 分支 | 用途 | 跟踪 |
|------|------|------|
| `main` | 与 upstream ollama/ollama 保持同步，只做 pull，不放自定义代码 | `upstream/main` |
| `dev` | 增强版 ollama，包含所有自定义改动，活跃开发分支 | `origin/dev` |
| `pr/<功能名>` | 临时 PR 分支，从 main 开出，cherry-pick dev 的内容，PR 合并后删除 | `origin/pr/<功能名>` |

### Remote 配置

| Remote | URL | 用途 |
|--------|-----|------|
| `origin` | `github.com/jonathanding/ollama.git` | 你的 fork |
| `upstream` | `github.com/ollama/ollama.git` | 上游原始项目 |

### 清理

- 删除 `feature/profiling-tracing` 分支（已合并进 main/dev）

## 迁移方案

采用 **直接 reset main** 方案：

```bash
# 1. 添加 upstream remote
git remote add upstream https://github.com/ollama/ollama.git
git fetch upstream

# 2. 从当前 main 创建 dev（保留全部自定义 commit）
git checkout -b dev
git push -u origin dev

# 3. 重置 main 到上游
git checkout main
git reset --hard upstream/main
git push --force origin main

# 4. 设置 main 跟踪 upstream
git branch --set-upstream-to=upstream/main main

# 5. 清理旧分支
git branch -d feature/profiling-tracing
git push origin --delete feature/profiling-tracing

# 6. 未跟踪文件处理
# .claude/, CLAUDE.md, docs/, ollama-llama-notes.md 等未提交的文件
# reset main 后仍留在工作目录，切到 dev 后提交即可
```

**注意事项：**

- 步骤 3 的 force push 会改写 GitHub 上的 main 历史。此 fork 只有本人使用，无协作风险。
- 步骤 4 让 `git pull` 在 main 上默认拉取 upstream（而非 origin）。
- 未跟踪文件不受 `reset --hard` 影响，不会丢失。

---

## 日常工作流程

### 1. 日常开发（在 dev 上）

```bash
git checkout dev
# ... 写代码 ...
git commit
git push origin dev
```

### 2. 同步上游更新

```bash
# 更新 main
git checkout main
git pull                        # main 跟踪 upstream/main，直接 pull

# 把 dev rebase 到最新的 main
git checkout dev
git rebase main
git push --force origin dev     # rebase 后需要 force push
```

### 3. 提交 PR 到上游

```bash
# 1. 确保 main 是最新的
git checkout main
git pull

# 2. 从 main 开 PR 分支
git checkout -b pr/profiler-core

# 3. 从 dev 挑选需要的 commit
git cherry-pick <commit1> <commit2> ...
# 或者如果是连续的一段：
git cherry-pick <start>..<end>

# 4. 如果需要整理（合并多个 commit、修改 commit message 等）
git rebase -i main

# 5. 推送并创建 PR
git push -u origin pr/profiler-core
# 然后在 GitHub 上向 ollama/ollama 的 main 提 PR
```

### 4. PR 合并后的收尾

```bash
# 1. 同步上游（此时包含了刚被合并的 PR）
git checkout main
git pull

# 2. 删除 PR 分支
git branch -d pr/profiler-core
git push origin --delete pr/profiler-core

# 3. rebase dev — 已合并的 commit 会自动消失
git checkout dev
git rebase main
git push --force origin dev
```

### 5. 处理 PR review 修改

```bash
# 上游 reviewer 要求修改时
git checkout pr/profiler-core
# ... 修改代码 ...
git commit
git push origin pr/profiler-core
# PR 会自动更新
```

## 完整生命周期图

```
upstream/main ──pull──▶ main ──branch──▶ pr/profiler-core ──PR──▶ upstream
                          │                     ▲
                          │               cherry-pick
                          │                     │
                          └──branch──▶ dev ─────┘
                               ▲          (活跃开发)
                               │
                           rebase main
                          (PR 合并后)
```

随着 PR 逐步被上游合并并 sync 回 main，dev 与 main 的差距会逐步缩小。最终如果所有改动都被上游接受，dev 可以退役。
