#!/usr/bin/env bash
# Thin SSH wrapper for the AutoDL box, so every caller uses the same options.
#
# Why this exists (Round 151 pitfall): the instance regenerates its host key on
# restart, and a stale entry in the user's known_hosts turns every connection
# into "Host key verification failed" (with BatchMode it fails outright; the
# write path for known_hosts also is not writable in every sandbox).  Pinning
# the key to /dev/null with StrictHostKeyChecking=no keeps the workflow moving.
# This is acceptable only because the host is reached over AutoDL's tunnel; it
# is NOT a general-purpose default.
#
# Usage:
#   scripts/ssh_autodl.sh                 # interactive shell on the remote
#   scripts/ssh_autodl.sh 'uptime; df -h' # run one command: ssh [opts] host [cmd]
#   QWEN35_EXTRA_OPTS='-T -v' scripts/ssh_autodl.sh 'cat f'   # extra ssh flags
#   QWEN35_RSYNC=1 scripts/ssh_autodl.sh  # exec rsync with remote shell wired up
#
# Environment:
#   QWEN35_HOST        default connect.nmb1.seetacloud.com
#   QWEN35_PORT        default 19236
#   QWEN35_USER        default root
#   QWEN35_EXTRA_OPTS  extra flags inserted before the destination (word-split)
set -euo pipefail

HOST="${QWEN35_HOST:-connect.nmb1.seetacloud.com}"
PORT="${QWEN35_PORT:-19236}"
USER_NAME="${QWEN35_USER:-root}"

SSH_OPTS=(
  -o ConnectTimeout=15
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
  -o LogLevel=ERROR
  -p "$PORT"
)
# shellcheck disable=SC2206  # intentional word-splitting of an explicit opt list
if [[ -n "${QWEN35_EXTRA_OPTS:-}" ]]; then
  SSH_OPTS+=(${QWEN35_EXTRA_OPTS})
fi

if [[ "${QWEN35_RSYNC:-0}" == "1" ]]; then
  exec rsync -e "ssh ${SSH_OPTS[*]}" "$@"
fi

# ssh requires the destination BEFORE the remote command:
#   ssh [options] destination [command]
exec ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "$@"
