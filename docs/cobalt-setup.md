# Cobalt setup for YouTube audio (Railway)

Use your own Cobalt instance so the Lyrics Meaning App can download YouTube audio without hitting YouTube’s “bot” block.

## Step 1: Deploy Cobalt on Railway

1. Open **https://railway.com/new/template/cobalt** (or search “Cobalt” in Railway templates).
2. Click **Deploy** / **Deploy Now**.
3. Log in or create a Railway account if needed.
4. Railway will create a new **project** with a Cobalt service. Wait for the first build to finish.

## Step 2: Get Cobalt’s public URL

1. In the **Cobalt project**, open the Cobalt **service** (the single service in that project).
2. Go to **Settings** → **Networking** (or **Public Networking**).
3. Click **Generate domain** (or **Add domain**).
4. Copy the URL Railway gives you, e.g. `https://cobalt-production-xxxx.up.railway.app`.  
   This is your **Cobalt base URL** (no path, no trailing slash).

(If the template set a custom domain or variable like `RAILWAY_PUBLIC_DOMAIN`, use that as the base URL.)

## Step 3: (Optional) Note the API key

If the Cobalt template or docs asked you to set an **API key** when deploying, copy that value.  
You’ll add it to the Lyrics app in the next step. If no key was required, skip this.

## Step 4: Point the Lyrics app at Cobalt

1. Open your **Lyrics Meaning App** project on Railway (the one with your Streamlit app).
2. Open the **Streamlit service** (your app).
3. Go to **Variables**.
4. Add:
   - **Name:** `COBALT_API_URL`  
   - **Value:** the Cobalt URL from Step 2, e.g. `https://cobalt-production-xxxx.up.railway.app`  
     (no trailing slash)
5. If you use an API key:  
   - **Name:** `COBALT_API_KEY`  
   - **Value:** the key from Step 3
6. Save. Railway will redeploy your app.

## Step 5: Test

1. Wait for the Lyrics app to finish redeploying.
2. Open your app URL and try a YouTube link (search or paste).
3. If it works, Cobalt is handling the download. If you get an error, check:
   - `COBALT_API_URL` is correct and has no trailing slash.
   - Cobalt service is running and its URL opens in a browser (you may see a simple page or “Cannot GET /”).
   - If Cobalt requires an API key, `COBALT_API_KEY` is set in the Lyrics app variables.

## Summary

| Where        | What to set |
|-------------|-------------|
| Cobalt project (Railway) | Deploy from template, generate domain |
| Lyrics app (Railway)    | `COBALT_API_URL` = Cobalt URL; optionally `COBALT_API_KEY` |

No code changes are required; the app already uses Cobalt when `COBALT_API_URL` is set.
