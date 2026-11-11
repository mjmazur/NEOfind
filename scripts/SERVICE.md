# Running the NEOfind campaign as a service

The campaign lives in `~/neofind/campaign_w3w4`. Two systemd user services
keep it going: `neofind-campaign` (the search, restarted after a crash) and
`neofind-reviewer` (the review page, http://127.0.0.1:8765). Both start at
boot without a login (linger is on) and resume from `campaign.sqlite`.

The service runs with `--automate-now`: it starts at once, then follows the
observatory's detection runs, which use the GPU. The controller (10.24.8.10)
sends SLAVEMSG datagrams to UDP port 17633; the campaign stops on
SLAVEMSG_DETBEG and starts again 30 minutes after SLAVEMSG_DETEND. It also
steps aside ("yielding" in the log) while any other process holds 300 MiB or
more of GPU memory (`--yield-min-mib`, `--yield-ignore`), so a boot in the
middle of a detection run is safe. `--automate` instead waits for a DETEND
on a fresh start. Without either, it works from 10:00 to 17:00
America/Toronto (`--hours`, `--tz`).

The controller must send its SLAVEMSGs to this machine (10.24.8.57).
`campaign.log` records every one received ("automate: SLAVEMSG_...").
To try it by hand: `.venv/bin/python -m neowise slavemsg detend` (or
`detbeg`) sends one from localhost, which is always accepted.

Run all commands from `~/source/NEOfind`.

## Install and start the services (once; already done on this machine)

```
mkdir -p ~/.config/systemd/user
cp scripts/neofind-campaign.service scripts/neofind-reviewer.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now neofind-campaign neofind-reviewer
loginctl enable-linger $USER   # start at boot and keep running after logout
```

Only one campaign can run on a folder at a time. If one was started by hand,
stop it first (`.venv/bin/python -m neowise campaign stop ~/neofind/campaign_w3w4`
and wait for `campaign status` to say "not running"), or the service will
refuse to start.

## Everyday use

| What | Command |
|---|---|
| Progress, rate, days left | `.venv/bin/python -m neowise campaign status ~/neofind/campaign_w3w4` |
| Pause (free the GPU now) | `.venv/bin/python -m neowise campaign pause ~/neofind/campaign_w3w4` |
| Resume | `.venv/bin/python -m neowise campaign resume ~/neofind/campaign_w3w4` |
| Blind review, sky map, plots | open http://127.0.0.1:8765 (from elsewhere: `ssh -L 8765:127.0.0.1:8765 mmazur@10.24.8.57`, then http://localhost:8765) |
| Requeue visits that failed (e.g. after a SkyBoT outage) | `.venv/bin/python -m neowise campaign retry-failed ~/neofind/campaign_w3w4` |
| Campaign log | `tail -f ~/neofind/campaign_w3w4/campaign.log` |
| Service output | `tail -f ~/neofind/campaign_w3w4.out` (reviewer: `~/neofind/reviewer.out`) |
| Service state | `systemctl --user status neofind-campaign` |
| Stop / start the campaign | `systemctl --user stop neofind-campaign` / `systemctl --user start neofind-campaign` |
| Restart the review page | `systemctl --user restart neofind-reviewer` |
| Turn it off for good | `systemctl --user disable --now neofind-campaign neofind-reviewer` |

Pause/resume is the way to borrow the GPU: the service keeps running but
does nothing until resumed. The review page has the same Pause button.

## Changing the campaign's options

Edit `ExecStart` in `~/.config/systemd/user/neofind-campaign.service` (and
the copy in `scripts/`), then:

```
systemctl --user daemon-reload
systemctl --user restart neofind-campaign
```

Keep `--wide-share 0.25`: without it the fields beyond ±10° stay queued
but are never searched.

Other options for `ExecStart` (see `python -m neowise campaign --help`):

- `--lat-max 20` (or `90` for the whole sky): widen the area; new fields are
  queued after the existing ones.
- `--hours 9-18`, `--tz America/Toronto`: active hours and their time zone.
- `--gpu-limit 0.8`: share of the GPU searches may use.
- `--jobs 2`: searches at a time.
- `--keep-fits`: keep every visit's images (about 40 MB each).
