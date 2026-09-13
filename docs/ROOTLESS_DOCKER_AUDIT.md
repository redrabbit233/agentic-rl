# Rootless Docker prerequisite audit

Date: 2026-09-12

Status: **Rootless Docker installer blocked on an additional, unapproved system prerequisite.**

## Stage 0 — read-only administrator audit

- Administrator identity: `chenkegen`
- Hostname: `zkti`
- Kernel: `Linux zkti 5.15.0-94-generic #104-Ubuntu SMP Tue Jan 9 15:25:40 UTC 2024 x86_64`
- `docker`: absent
- `dockerd`: absent
- `newuidmap`: absent
- `newgidmap`: absent
- `slirp4netns`: absent
- `fuse-overlayfs`: absent
- Installed package query for `uidmap`, `slirp4netns`, and `fuse-overlayfs`: no matching installed packages.
- `kernel.unprivileged_userns_clone = 1`
- `/etc/subuid`: `chenduo:427680:65536`
- `/etc/subgid`: `chenduo:427680:65536`

The existing subordinate-ID allocation and unprivileged-user-namespace setting satisfy the relevant rootless prerequisite. No rootful Docker daemon or system Docker socket was found.

## Stage 1 — simulated package operation

Command simulated:

```text
apt-get -s install --no-install-recommends uidmap slirp4netns fuse-overlayfs
```

Simulation result:

```text
0 upgraded, 4 newly installed, 0 to remove and 222 not upgraded.
NEW: fuse-overlayfs 1.7.1-1
NEW: libslirp0 4.6.1-1build1 (dependency)
NEW: slirp4netns 1.0.1-2
NEW: uidmap 1:4.8.1-2ubuntu2.2
```

The simulation has no package removals, no upgrades, and no kernel, NVIDIA, CUDA, Docker-daemon, firewall, or network changes. It is safe to proceed with the requested minimal installation.

## Stage 2 result

Following the safe simulation result, the administrator installed exactly:

```text
uidmap 1:4.8.1-2ubuntu2.2
slirp4netns 1.0.1-2
fuse-overlayfs 1.7.1-1
libslirp0 4.6.1-1build1 (required library dependency)
```

The operation had 0 package upgrades and 0 removals. It did not install `docker.io`, start a daemon, enable a service, modify `/etc/docker`, or modify network/firewall rules. `newuidmap`, `newgidmap`, `slirp4netns`, and `fuse-overlayfs` are now present. The `chenduo` subuid/subgid entries remain intact.

## Stage 4 result and blocker

The Docker official rootless installer was invoked as `chenduo` only. It stopped before installing Docker because its prerequisite check requires the system `iptables` executable:

```text
Missing system requirements ... apt-get -y install iptables
```

`iptables` was not in the approved minimal package set. The shared-server rules prohibit modifying iptables/nftables and require a stop before expanding system scope. `SKIP_IPTABLES=1` was not used, so no prerequisite check was bypassed. No rootless daemon, rootful daemon, system Docker socket, or Docker network configuration was created.

Explicit approval is required before simulating or installing the additional `iptables` package. The next safe action, if approved, is `apt-get -s install --no-install-recommends iptables`, followed by review of its exact package plan.
