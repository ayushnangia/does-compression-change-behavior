Bootstrap: docker
From: {{IMAGE}}

# Bake everything harbor's singularity bootstrap.sh would otherwise install
# at container start (needs internet, which compute nodes don't have):
#   /usr/bin/python3, the /opt/harbor-server venv with pip+uvicorn+fastapi,
#   tmux, asciinema. bootstrap.sh short-circuits when these already exist.
# Reconstructed from harbor 0.20 bootstrap.sh (original template was never
# committed and died with Narval scratch).

%post
    export DEBIAN_FRONTEND=noninteractive
    # --- python3 + tmux (+ asciinema where available) ---
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq || true
        apt-get install -y -qq python3 python3-venv python3-pip tmux asciinema curl ca-certificates || \
        apt-get install -y -qq python3 python3-venv python3-pip tmux curl ca-certificates || true
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache python3 py3-pip tmux asciinema curl ca-certificates || \
        apk add --no-cache python3 py3-pip tmux curl ca-certificates || true
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y python3 python3-pip tmux curl ca-certificates || true
    elif command -v yum >/dev/null 2>&1; then
        yum install -y python3 python3-pip tmux curl ca-certificates || true
    fi
    test -x /usr/bin/python3 || { echo "FATAL: no python3 after install"; exit 1; }

    # --- /opt/harbor-server venv with pip + uvicorn + fastapi ---
    /usr/bin/python3 -m venv /opt/harbor-server 2>/dev/null || \
        /usr/bin/python3 -m venv --without-pip /opt/harbor-server
    if ! /opt/harbor-server/bin/python3 -m pip --version >/dev/null 2>&1; then
        /opt/harbor-server/bin/python3 -m ensurepip --default-pip 2>/dev/null || true
    fi
    if ! /opt/harbor-server/bin/python3 -m pip --version >/dev/null 2>&1; then
        curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        /opt/harbor-server/bin/python3 /tmp/get-pip.py --quiet
    fi
    /opt/harbor-server/bin/python3 -m pip install --quiet --upgrade pip
    /opt/harbor-server/bin/python3 -m pip install --quiet uvicorn fastapi
    /opt/harbor-server/bin/python3 -c "import uvicorn, fastapi" || { echo "FATAL: server venv incomplete"; exit 1; }

    # asciinema fallback via pip if apt/apk had none
    command -v asciinema >/dev/null 2>&1 || \
        /opt/harbor-server/bin/python3 -m pip install --quiet asciinema || true

    # --- TB2 VERIFIER deps (the all-zeros bug, 2026-08-06): every task's
    # tests/test.sh curl-installs uv from astral.sh and uvx-resolves
    # python3.13+pytest AT VERIFY TIME. Compute nodes are air-gapped, so
    # the verifier could never run and every reward was 0 regardless of
    # the agent. Pre-warm uv + the EXACT uvx environments (union over all
    # tasks sharing this image) at bake time; at runtime test.sh's curl
    # fails harmlessly, `source /root/.local/bin/env` finds the baked file,
    # and uvx hits the warm cache offline. ---
    curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh || true
    export PATH=/root/.local/bin:$PATH
{{TEST_PREWARM}}

%environment
    # uvx on PATH even though test.sh's curl-install fails offline; force
    # uv to its baked cache instead of a doomed 300s network timeout
    export PATH=/root/.local/bin:$PATH
    export UV_OFFLINE=1
