# Running PinPoint on Claude (most human mode)

This makes PinPoint sound dramatically more human, stops the made-up nonsense,
and runs faster than the local model. It needs internet + a few dollars of API
credit.

## One-time setup (5 minutes)

### 1. Get an API key
- Go to **https://console.anthropic.com/**
- Sign up / log in.
- Click **Settings → Billing** and add a little credit (even $5 lasts a long
  time for chatting).
- Click **API Keys → Create Key**, copy the key (it starts with `sk-ant-...`).

### 2. Install the Anthropic library
Open a terminal in the pinpoint folder and run:
```
pip install anthropic
```

### 3. Put your key in the launcher
- Open **`run_claude.bat`** (in this folder) in Notepad.
- Replace `PASTE_YOUR_KEY_HERE` with the key you copied.
- Save.

## Every time you want to run her on Claude
Just **double-click `run_claude.bat`** (or run it from the terminal).
That's it — she boots up on Claude.

To go back to the free local model, run `pinpoint` normally as before.

## Notes
- Default Claude model: **claude-sonnet-4-5** (fast + very human).
- Want the absolute smartest? Edit `run_claude.bat` and change the
  `ANTHROPIC_MODEL` line to `claude-opus-4-1` (smarter, costs more).
- Your key is private — don't share `run_claude.bat` with the key filled in.
