# How to Contribute to This Repository

Step-by-step guide for working on this project. If you follow this exactly, you won't break anything that is already in `main`.

> **Note:** this guide applies to the entire team. The repository owner ([@mar7inlancheros-maker](https://github.com/mar7inlancheros-maker)) pushes their own changes directly to `main`, without a branch or pull request; see the **"Repository Owner Workflow"** section at the end.

## Before You Start (Once)

### 1. Request Access

You need to be a collaborator on the repo. If you haven't been added yet, ask to be invited using your GitHub username or email, then accept the invitation you receive by email or GitHub notification.

### 2. Configure Your Git Identity

Open a terminal (in VS Code: `Terminal → New Terminal`) and run this once on your computer:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

Use the same email address that is associated with your GitHub account.

### 3. Clone the Repository

Choose a folder on your computer (outside OneDrive, to avoid sync conflicts) and run:

```bash
git clone https://github.com/mar7inlancheros-maker/Asset-Management-.git
cd Asset-Management-
```

This downloads the entire project. Open it in VS Code (`File → Open Folder...` and select the `Asset-Management-` folder).

### 4. Create Your Local Configuration File

Make a copy of `.env.example` and rename it to `.env`. Fill in your own values (for example, your real email address for the User-Agent header required by the SEC). This file is **never uploaded to GitHub**; it stays only on your computer.

---

## The One Golden Rule

**If you are not the repository owner, never run `git push` while you are on `main`.** Always work from your own branch. This repository has no automatic branch protection to prevent it (it is private on a free plan), so this rule only works if everyone follows it.

---

## The Workflow, Step by Step

Repeat this cycle every time you are going to work on something new.

### Step 1 — Get the Latest Changes Before Starting

```bash
git checkout main
git pull
```

This ensures you start from the latest version of the project, not an outdated copy.

### Step 2 — Create Your Own Branch

```bash
git checkout -b feature/descriptive-name
```

Replace `descriptive-name` with something short and clear. Real examples for this project:

```bash
git checkout -b feature/factor-momentum
git checkout -b fix/fama-macbeth-division-by-zero
git checkout -b experiment/new-sector-universe
```

Use the appropriate prefix:

- `feature/` → something new
- `fix/` → bug fix
- `experiment/` → an experiment that may not end up being used

### Step 3 — Work Normally

Edit your `.py` files in VS Code, run your code, and run the tests. Nothing changes from how you already work.

### Step 4 — Review What You Changed

```bash
git status
```

This shows which files you modified. To see the changes line by line:

```bash
git diff
```

### Step 5 — Save Your Changes (Commit)

```bash
git add .
git commit -m "Short, clear description of the change"
```

Examples of good commit messages:

- `"Add liquidity filter to quality_factor"`
- `"Fix division by zero when sector has fewer than 3 companies"`
- `"Add tests for the new score calculation"`

If you worked on several unrelated things, split them into separate commits (run `add` and `commit` several times, with different files each time) instead of mixing everything into one.

**Before committing, always check** that `git status` does not list your `.env` file. If it does, stop and notify the team before continuing.

### Step 6 — Push Your Branch to GitHub

The first time you push that specific branch:

```bash
git push -u origin feature/descriptive-name
```

After that, if you keep working on the same branch, just run:

```bash
git push
```

### Step 7 — Open the Pull Request

After the `push`, the terminal will usually show you a link like:

```
https://github.com/mar7inlancheros-maker/Asset-Management-/pull/new/feature/descriptive-name
```

Copy and paste it into your browser. If the link does not appear, go to the repository page on GitHub, where a banner with a **"Compare & pull request"** button will be shown.

When you open it:

- Write a clear title.
- In the description, explain what you did and why (2–3 lines are enough).
- Click **"Create pull request"**.

### Step 8 — Notify the Team

Since there is no automatic protection, this step takes its place: post in the team chat that your pull request is ready for review, and include the link.

### Step 9 — Wait for Review

A teammate opens the PR on GitHub, reviews the code, and:

- If everything is fine: they approve it and click **"Merge pull request"**.
- If something needs adjustment: they comment directly on the relevant lines.

If you are asked to make a change, fix it on the same local branch and repeat Steps 5 and 6 (`add`, `commit`, `push`). The pull request updates automatically; you do not need to open a new one.

### Step 10 — Cleanup

Once your PR has been merged, GitHub shows a **"Delete branch"** button right there. Delete the branch; it has served its purpose. Then, on your computer:

```bash
git checkout main
git pull
git branch -d feature/descriptive-name
```

That's it. Go back to Step 1 for the next thing you work on.

---

## If You Get a Merge Conflict

Git will mark the conflicting section of the file like this:

```python
<<<<<<< HEAD
    return df["roic"] * 0.6 + df["margin"] * 0.4
=======
    return df["roic"] * 0.5 + df["margin"] * 0.3 + df["growth"] * 0.2
>>>>>>> feature/your-branch
```

Decide which version should stay (or combine both manually), delete the `<<<<<<<`, `=======`, and `>>>>>>>` lines, save the file, and continue:

```bash
git add file_name.py
git commit -m "Resolve merge conflict in file_name.py"
git push
```

If you are not sure how to resolve it, ask for help instead of guessing. A badly resolved conflict can accidentally delete someone else's work.

---

## Frequently Asked Questions

**Can I work directly on `main` for something quick?**
No. Even if it feels slower, creating a branch takes 10 seconds and keeps a mistake from immediately affecting the whole team.

**What if `git pull` says I have uncommitted changes?**
Commit or discard your local changes first (`git add` + `git commit`, or `git stash` to set them aside temporarily without committing), then try again.

**What if I pushed something I shouldn't have (such as `.env`)?**
Notify the repository owner immediately. Deleting the file and pushing a new commit is not enough: it remains visible in the commit history and must be removed separately (and any secrets in it should be rotated).

**How do I view the change history?**

```bash
git log --oneline
```

---

## Repository Owner Workflow

The repository owner writes most of the code and pushes their changes **directly to `main`, without a branch or pull request**. Their workflow is:

```bash
git checkout main
git pull
# edit whatever is needed...
git add .
git commit -m "Description of the change"
git push
```

The one step they never skip is `git pull` at the start, so their push does not conflict with anything the team merged in the meantime.

This exception applies only to the owner. The rest of the team follows the branch + pull request workflow described above.
