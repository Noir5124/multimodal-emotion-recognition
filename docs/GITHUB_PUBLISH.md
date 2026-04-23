# Publish to GitHub

`gh` CLI not installed in this environment, so easiest path:

## Option 1. GitHub Website + Git

1. Create new empty repo on GitHub.
2. Open terminal in:

```text
C:\Users\HP\Desktop\caveman\emotion_recognition_pipeline
```

3. Repo already initialized locally. Run:

```powershell
git add .
git commit -m "feat: add multimodal emotion recognition pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Option 2. GitHub Desktop

1. Open GitHub Desktop
2. Add local repository:

```text
C:\Users\HP\Desktop\caveman\emotion_recognition_pipeline
```

3. Publish repository
4. Choose public or private
5. Push

## Recommended Repo Name

```text
multimodal-emotion-recognition
```

## Before Pushing

Keep model weights out of repo unless using Git LFS or Releases.

Safe to push:

- code
- docs
- generated evaluation graphs
- notebook instructions

Avoid pushing:

- `.venv/`
- `artifacts/`
- external trained models
- local logs
