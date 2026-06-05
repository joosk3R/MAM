# GitHub Upload Guide

This guide assumes that the target repository is:

```text
https://github.com/joosk3R/MAM
```

## 1. Create an empty repository on GitHub

Create a public repository named `MAM` under the `joosk3R` account.

Do not initialize it with README, `.gitignore`, or LICENSE because these files are already included in this folder.

## 2. Move into this repository folder

```bash
cd MAM_upload_ready
```

If you rename this folder to `MAM`, use:

```bash
cd MAM
```

## 3. Initialize Git and commit

```bash
git init -b main
git add .
git commit -m "Initial public release"
```

## 4. Connect to GitHub

```bash
git remote add origin https://github.com/joosk3R/MAM.git
git push -u origin main
```

## 5. Check before public release

Confirm that the following files are not uploaded:

```text
*.dcm
*.nii
*.nii.gz
*.pth
*.pt
*.ckpt
```

Confirm that the repository URL matches the paper:

```text
https://github.com/joosk3R/MAM
```
