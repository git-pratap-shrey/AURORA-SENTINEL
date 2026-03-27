# Aurora Sentinel: Cloudflare Tunnel Guide (Option B)

This guide explains how to expose the Aurora Sentinel application via a **single** Cloudflare Tunnel. By building the frontend first, the FastAPI backend will serve both the UI and the API from a single port (`8000`), making the setup simple and robust.

---

## 1. Prerequisites

### Build the Frontend
The backend only serves the frontend if it's already built. Run this once:
```bash
cd frontend
npm install
npm run build
cd ..
```
*Verify: Check if `frontend/build/index.html` exists.*

---

## 2. Install & Authenticate Cloudflare

### Install `cloudflared` (if not installed)
```bash
# On Ubuntu/Debian:
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
```

### Authenticate
Run this to log in to your Cloudflare account:
```bash
cloudflared tunnel login
```
*A browser window will open. Select your domain and authorize.*

---

## 3. Create the Tunnel

### Create a tunnel name (e.g., "aurora-sentinel")
```bash
cloudflared tunnel create aurora-sentinel
```
*This will output a **Tunnel ID**. Copy it.*

### Configure the Tunnel
The `cloudflared_config.yml` file is provided in the project root. Update it with your **Tunnel ID** and **Hostname**:

1.  Open [cloudflared_config.yml](file:///home/anon/PROJECTS/aurora/AURORA-SENTINEL-1/cloudflared_config.yml).
2.  Replace `<tunnel-id-or-name>` with your Tunnel ID.
3.  Replace `<tunnel-id>.json` with your Tunnel ID (usually in `~/.cloudflared/`).
4.  Replace `<your-subdomain.example.com>` with the public URL you want (e.g., `aurora.example.com`).

---

## 4. Final Routing (DNS)

Route your public hostname to the tunnel:
```bash
cloudflared tunnel route dns aurora-sentinel <your-subdomain.example.com>
```

---

## 5. Run the Application

Now, start both the app and the tunnel:

### Start the App
```bash
# In one terminal:
./start.sh
```

### Start the Tunnel
```bash
# In another terminal:
cloudflared tunnel --config cloudflared_config.yml run
```

---

## ☁️ Accessing the App
Open your browser to `https://<your-subdomain.example.com>`.

- **Frontend**: Served automatically by FastAPI.
- **Backend API**: Accessible at `/alerts`, `/analytics`, etc.
- **WebSockets**: Handled automatically by Cloudflare.

---

### Troubleshooting
- **CORS Errors**: The backend is already configured with `allow_origins=["*"]`, so you shouldn't see any.
- **Frontend not loading**: Ensure `frontend/build` exists and you've run `npm run build`.
- **WebSocket issues**: Cloudflare supports WebSockets by default, but ensure no proxy or firewall is blocking them.
