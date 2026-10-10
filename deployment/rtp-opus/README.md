# RTP/Opus receiver

The receiver validates RTP version 2 and Opus payload type 111, then writes a local spool with
`Audio.opus`, packet timing, and a loss-visible metadata receipt. Recorder source addresses,
listener bindings, ports and spool roots belong to private site configuration.
It does not modify capture archives or treat receiver arrival time as the recorder's capture clock.

Provision and verify the site's network identity before starting the receiver. A generic checkout
must not listen for an internal recorder or reuse another site's allowlist.

The service file is a template. Render it with the current checkout, interpreter and private
receiver configuration before installation:

```sh
python deployment/rtx3090ti-ubuntu/render_service.py render-rtp \
  "$PWD" "$PWD/.venv/bin/python" "/absolute/private/receiver-config.json" \
  deployment/rtp-opus/visioncortex-rtp-opus.service /tmp/visioncortex-rtp-opus.service
systemd-analyze --user verify /tmp/visioncortex-rtp-opus.service
```

Install the verified rendered file using the site's service procedure; copying the template directly
leaves unresolved paths and cannot create a working receiver service.
