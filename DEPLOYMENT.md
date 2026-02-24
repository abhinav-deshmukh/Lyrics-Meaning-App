# Deployment: Staging → Production

We use a **staging branch** so you can validate on a live URL before merging to `main`, which deploys to production (active users).

## Branch strategy

| Branch   | Purpose        | Deploys to        |
|----------|----------------|-------------------|
| `staging`| Test changes   | Railway Staging   |
| `main`   | Production     | Railway Production (surasa.up.railway.app) |

## One-time setup

### 1. Create the `staging` branch

```bash
git checkout -b staging
git push -u origin staging
```

### 2. Add a Staging environment in Railway

- In [Railway](https://railway.app) → your project **remarkable-compassion**
- Create a new **Environment** (e.g. "Staging") — [Railway: Environments](https://docs.railway.app/guides/environments)
- In that environment, add a **Service** (or duplicate your existing one) and connect the same repo
- Set the service to deploy from branch **`staging`** (in Service → Settings → Source → Branch)
- Give the staging service a **domain** (e.g. `surasa-staging.up.railway.app`) so you have a stable URL

Your Production environment stays pointed at **`main`** (current behavior).

## Daily workflow

### 1. Develop and test locally

```bash
git checkout staging
# ... make changes ...
streamlit run app.py   # validate at http://localhost:8501
```

### 2. Push to staging (deploy to staging URL only)

```bash
git add .
git commit -m "Your message"
git push origin staging
```

- Railway will auto-deploy the **Staging** service from `staging`.
- Open your **staging URL** and validate.

### 3. Promote to production when ready

```bash
git checkout main
git merge staging
git push origin main
```

- Railway will auto-deploy **Production** from `main`.
- Production URL (e.g. surasa.up.railway.app) now has the new version.

### 4. Keep staging in sync (optional)

After merging to main, update staging so it doesn’t drift:

```bash
git checkout staging
git merge main
git push origin staging
```

## Quick reference

| Goal                    | Command |
|-------------------------|--------|
| Deploy to staging only  | `git push origin staging` |
| Deploy to production    | Merge staging → main, then `git push origin main` |
| Test locally            | `streamlit run app.py` |

## If you prefer not to use Railway environments yet

- You can still use the **branch workflow** without a second Railway service:
  - Do all development on `staging`, push to `staging`, and only merge to `main` when you’re ready.
  - Production (main) still auto-deploys from `main`; you just wouldn’t have a separate staging URL until you add a Staging environment and point it at `staging`.
