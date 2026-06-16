#!/usr/bin/env bash
#
# Build + push the synthea-neo4j agent image to Docker Hub.
#
# Version is read from the VERSION file in this directory. Bump it
# manually before running (e.g. v1.0.0 -> v1.0.1) and commit the change
# so the repo records every published tag.
#
# Usage:
#   ./build-and-push.sh                 # uses VERSION file
#   ./build-and-push.sh v1.2.3          # one-off override (does not edit VERSION)
#   ./build-and-push.sh --dry-run       # print what would happen, don't build/push
#   ./build-and-push.sh -h | --help
#
# Requires: a prior `docker login -u baiondata`.

set -euo pipefail

IMAGE="baiondata/ci-rp-app-agent"
HERE="$(cd "$(dirname "$0")" && pwd)"

DRY_RUN=0
TAG_OVERRIDE=""

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      sed -n '2,17p' "$0"
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -*)
      echo "Unknown flag: $arg" >&2
      exit 1
      ;;
    *)
      TAG_OVERRIDE="$arg"
      ;;
  esac
done

if [[ -n "$TAG_OVERRIDE" ]]; then
  TAG="$TAG_OVERRIDE"
else
  if [[ ! -f "$HERE/VERSION" ]]; then
    echo "No VERSION file at $HERE/VERSION. Create one (e.g. 'v1.0.1') or pass a tag." >&2
    exit 1
  fi
  TAG="$(tr -d '[:space:]' < "$HERE/VERSION")"
fi

if [[ -z "$TAG" ]]; then
  echo "Empty tag." >&2
  exit 1
fi

FULL="${IMAGE}:${TAG}"
echo "Image:  ${FULL}"
echo "Source: ${HERE}"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "(dry-run) docker build -t ${FULL} ${HERE}"
  echo "(dry-run) docker push ${FULL}"
  exit 0
fi

# Refuse to clobber a tag that's already published, except for the
# floating 'latest' / 'dev' aliases.
if [[ "$TAG" != "latest" && "$TAG" != "dev" ]]; then
  if docker manifest inspect "${FULL}" >/dev/null 2>&1; then
    echo "ERROR: ${FULL} already exists on Docker Hub." >&2
    echo "       Bump VERSION (or pass a new tag) before re-running." >&2
    exit 2
  fi
fi

echo
echo "==> Building ${FULL} ..."
docker build -t "${FULL}" "${HERE}"

echo
echo "==> Pushing ${FULL} ..."
docker push "${FULL}"

DIGEST="$(docker inspect --format='{{index .RepoDigests 0}}' "${FULL}" 2>/dev/null || true)"
SIZE="$(docker image inspect "${FULL}" --format '{{.Size}}' 2>/dev/null || echo 0)"
SIZE_MB=$(( SIZE / 1024 / 1024 ))

echo
echo "Done."
echo "  Tag:    ${FULL}"
echo "  Size:   ${SIZE_MB} MB"
echo "  Digest: ${DIGEST:-(unknown)}"
echo
echo "  Pull elsewhere with:"
echo "    docker pull ${FULL}"
