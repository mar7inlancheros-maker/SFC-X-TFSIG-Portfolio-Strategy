# How to Contribute to This Repository

Step-by-step guide for working on this project. If you follow this exactly, you won't break anything that is already in `main`.

> **Note:** this guide applies to the entire team. The repository owner ([@mar7inlancheros-maker](https://github.com/mar7inlancheros-maker)) pushes their own changes directly to `main`, without a branch or pull request — see the **"Repository Owner Workflow"** section at the end.

## Before You Start (Once)

### 1. Request Access

You need to be a collaborator on the repo. If you haven't been added yet, ask to be invited using your GitHub username or email, and accept the invitation you receive by email/notification.

### 2. Configure Your Git Identity

Open a terminal (in VS Code: `Terminal → New Terminal`) and run this once on your computer:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

Use the same email address associated with your GitHub account.

### 3. Clone the Repository

Choose a folder on your computer (outside OneDrive, to avoid sync conflicts) and run:

```bash
git clone https://github.com/mar7inlancheros-maker/Asset-Management-.git
cd Asset-Management-
```

This downloads the entire project. Open it in VS Code (`File → Open Folder...`, select the `Asset-Management-` folder).

### 4. Create Your Local Configuration File

Copy `.env.example` and rename it to `.env`. Fill in your own values (for example, your real email address for the User-Agent required by the SEC). This file is **never uploaded to GitHub** — it is only for you, on your computer.

---

## The One Golden Rule

**If you are not the repository owner, never run `git push` while you are on `main`.** Always work from your own branch. This repository does not have an automatic protection rule preventing this (because it is private on a free plan), so this rule depends on everyone following it by agreement.

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
git checkout -b feature/nombre-descriptivo
```

Replace `nombre-descriptivo` with something clear and short. Real examples for this project:

```bash
git checkout -b feature/factor-momentum
git checkout -b fix/division-cero-fama-macbeth
git checkout -b experiment/nuevo-universo-sectores
```

Use the appropriate prefix:
- `feature/` → something new
- `fix/` → bug fix
- `experiment/` → an experiment that may not end up being used

### Step 3 — Work Normally

Edit your `.py` files in VS Code, run your tests, and test your code. Nothing different from how you already work.

### Step 4 — Review What You Changed

```bash
git status
```

This shows which files you modified. If you want to see the line-by-line details:

```bash
git diff
```

### Step 5 — Save Your Changes (Commit)

```bash
git add .
git commit -m "Descripción corta y clara del cambio"
```

Examples of good commit messages:
- `"Add liquidity filter to quality_factor"`
- `"Fix division by zero when sector has fewer than 3 companies"`
- `"Add tests for the new score calculation"`

If you worked on several different things, separate them into different commits (run `add` and `commit` multiple times with different files) instead of mixing everything into one.

**Before this step, always check** that `git status` does not show your `.env` file in the list — if you see it there, stop and notify the team before continuing.

### Step 6 — Push Your Branch to GitHub

The first time you push that specific branch:

```bash
git push -u origin feature/nombre-descriptivo
```

The next times, if you keep working on the same branch, simply run:

```bash
git push
```

### Step 7 — Open the Pull Request

After the `push`, the terminal will usually show you a link like:

```
https://github.com/mar7inlancheros-maker/Asset-Management-/pull/new/feature/nombre-descriptivo
```

Copy and paste it into your browser. If the link did not appear, go to the repository page on GitHub — a notification with a **"Compare & pull request"** button will appear.

When you open it:
- Write a clear title.
- In the description, explain what you did and why (2–3 lines are enough).
- Click **"Create pull request"**.

### Step 8 — Notify the Team

Since there is no automatic protection, this step replaces that function: let the team know in the team chat that you left a pull request ready for review, with the link.

### Step 9 — Wait for Review

Someone on the team opens the PR on GitHub, reviews the code, and:
- If everything is fine: they approve it and click **"Merge pull request"**.
- If something needs adjustment: they comment directly on the relevant lines.

If you are asked to make a change, fix it in your same local branch, and repeat Steps 5 and 6 (`add`, `commit`, `push`) — the pull request updates automatically; you do not need to open a new one.

### Step 10 — Cleanup

Once your PR has been merged, GitHub offers a **"Delete branch"** button right there — delete it; it has served its purpose. On your computer:

```bash
git checkout main
git pull
git branch -d feature/nombre-descriptivo
```

And that's it — go back to Step 1 for the next thing you work on.

---

## If You Get a Merge Conflict

Git te va a marcar el archivo así:

```python
<<<<<<< HEAD
    return df["roic"] * 0.6 + df["margin"] * 0.4
=======
    return df["roic"] * 0.5 + df["margin"] * 0.3 + df["growth"] * 0.2
>>>>>>> feature/tu-rama
```

Decide which version should remain (or combine both manually), delete the `<<<<<<<`, `=======`, and `>>>>>>>` lines, save the file, and continue:

```bash
git add nombre_del_archivo.py
git commit -m "Resuelve conflicto en nombre_del_archivo.py"
git push
```

If you are not sure how to resolve it, ask for help before guessing — a poorly resolved conflict can accidentally delete someone else's work.

---

## Frequently Asked Questions

**Can I work directly on `main` for something quick?**
No. Even if it seems slower, creating a branch takes 10 seconds and prevents a mistake from immediately affecting the entire team.

**What if `git pull` says I have uncommitted changes?**
Save or discard your local changes first (`git add` + `git commit`, or `git stash` if you want to temporarily save them without committing), then try again.

**What if I pushed something I shouldn't have (such as `.env`)?**
Notify the repository owner immediately. Deleting it and pushing a new commit is not enough — the file remains visible in the previous history and must be handled separately.

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
# edita lo que sea...
git add .
git commit -m "descripción del cambio"
git push
```

The only step they do not skip is `git pull` at the beginning — this prevents their push from conflicting with something the team uploaded in the meantime.

This exception applies only to the owner. The rest of the team follows the branch + pull request workflow described above.
