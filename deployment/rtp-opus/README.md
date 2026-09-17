# RTP/Opus receiver

The receiver accepts only the recorder source `192.168.1.156` on `192.168.1.145:50030/UDP`, validates RTP version 2 and Opus payload type 111, and writes a local spool with `Audio.opus`, packet timing, and a loss-visible metadata receipt. It does not alter the NAS capture or claim receiver arrival time is the recorder's capture clock.

The systemd unit is installed per-user. The runtime configuration and spool stay outside Git because they are machine-specific runtime state.
