# Git Workshop: Add Your Own Introduction

Welcome! This exercise walks you through the basic Git and GitHub workflow: cloning, making changes, and opening a pull request.

## What you'll do
Clone this repo, add your own introduction file, and submit it via a pull request.

## Steps

### 1. Clone this repository
```bash
git clone https://github.com/NUS-SDS-Workshops-AY2627/Sem-1.git
cd Sem-1
```

### 2. Create a new branch
```bash
git checkout -b <your-name>-intro
```

### 3. Copy the template and create your file
```bash
cp Git-Workshop/template.md Git-Workshop/<your-name>.md
```
Open `Git-Workshop/<your-name>.md` in your editor and fill it in.

### 4. Stage and commit your changes
```bash
git add Git-Workshop/<your-name>.md
git commit -m "Add <your-name> introduction"
```

### 5. Push your branch
```bash
git push -u origin <your-name>-intro
```

### 6. Open a pull request
GitHub will print a link after you push, or go to the repo and click **"Compare & pull request."** Make sure the base branch is `main`, set a reviewer (Adam210503 or JasonOng0109), then submit.

### 7. Wait for review
A workshop organiser will review and merge your PR. Once merged, your file will appear in the repo for everyone to see!

## Notes
- Use your **name or handle** for the branch and filename (e.g. `adam-mikail.md`) so it's easy to identify.
- Keep your PR focused on just your own file, don't edit anyone else's.
- If you hit merge conflicts or get stuck, ask a facilitator.