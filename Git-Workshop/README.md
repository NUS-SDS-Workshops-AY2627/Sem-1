# Git Workshop — Add Your Own Introduction

Welcome! This exercise walks you through the basic Git/GitHub workflow: forking, cloning, making changes, and opening a pull request.

## What you'll do
Create your own introduction file, connect your local machine to this repo, and submit it via a pull request.

## Steps

### 1. Fork this repository
Click **Fork** (top right of the repo page) to create your own copy under your GitHub account.

### 2. Clone your fork locally
```bash
git clone https://github.com/<your-username>/Sem-1.git
cd Sem-1
```

### 3. Create a new branch
```bash
git checkout -b <your-name>-intro
```

### 4. Copy the template and create your file
```bash
cp Git-Workshop/template.md Git-Workshop/<your-name>.md
```
Open `Git-Workshop/<your-name>.md` in your editor and fill it in.

### 5. Stage and commit your changes
```bash
git add Git-Workshop/<your-name>.md
git commit -m "Add <your-name> introduction"
```

### 6. Push your branch to your fork
```bash
git push -u origin <your-name>-intro
```

### 7. Open a pull request
Go to your fork on GitHub — you'll see a banner prompting **"Compare & pull request."** Click it, make sure the base repository is set to `NUS-SDS-Workshops-AY2627/Sem-1` and base branch is `main`, then submit.

### 8. Wait for review
A workshop organiser will review and merge your PR. Once merged, your file will appear in the main repo for everyone to see!

## Notes
- Use your **name or handle** for the branch and filename (e.g. `adam-mikail.md`) so it's easy to identify.
- Keep your PR focused on just your own file, don't edit anyone else's.
- If you hit merge conflicts or get stuck, ask a facilitator.