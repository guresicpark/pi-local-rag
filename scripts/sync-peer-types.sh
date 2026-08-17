#!/usr/bin/env bash
#
# Rebuild vendor/pi-coding-agent-types from the published
# @earendil-works/pi-coding-agent npm package.
#
# Why this exists: the published package ships an npm-shrinkwrap.json, which
# freezes its whole dependency subtree and makes it immune to the root
# package's dependency resolution (including overrides and hoisting-based
# stub dedupe). For local development we only need its TypeScript
# declarations (our imports from it are type-only), so we vendor a
# types-only copy — same package identity and dependency list, but without
# the shrinkwrap — which keeps the dev tree normally resolvable and free of
# install-time deprecation warnings.
#
# Usage:
#   scripts/sync-peer-types.sh [version]     # default: the version already
#                                            # vendored, or latest
set -euo pipefail

VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor/pi-coding-agent-types"
UPSTREAM_NAME="@earendil-works/pi-coding-agent"

current_version() {
  node -e "console.log(require('$VENDOR_DIR/package.json').version)" 2>/dev/null || true
}

# Target version: explicit argument > currently vendored > latest.
if [[ $# -ge 1 ]]; then
  VERSION="$1"
else
  VERSION="$(current_version)"
  [[ -n "$VERSION" ]] || VERSION="$(npm view "$UPSTREAM_NAME" version)"
fi
echo "Vendoring $UPSTREAM_NAME@$VERSION type declarations…"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

npm pack "$UPSTREAM_NAME@$VERSION" --pack-destination "$WORK_DIR" --silent
TARBALL="$WORK_DIR/$(ls "$WORK_DIR")"

# Extract only the .d.ts files, preserving the dist/ directory structure.
TAR_ROOT="$(tar tzf "$TARBALL" | head -1 | cut -d/ -f1)"
mkdir -p "$WORK_DIR/extract"
tar xzf "$TARBALL" -C "$WORK_DIR/extract"
rm -rf "$VENDOR_DIR"
mkdir -p "$VENDOR_DIR"
(cd "$WORK_DIR/extract/$TAR_ROOT" && find dist -name '*.d.ts' -print0 | xargs -0 tar cf -) \
  | (cd "$VENDOR_DIR" && tar xf -)

# Rewrite the manifest: same identity + dependencies (so transitive type
# imports resolve), types-only exports, no shrinkwrap side-effects. The
# version is nudged to the next patch so it satisfies the peer range
# declared in the root package.json.
node - "$WORK_DIR/extract/$TAR_ROOT/package.json" "$VENDOR_DIR/package.json" "$VERSION" <<'EOF'
const fs = require("fs");
const upstreamPath = process.argv[2];
const outputPath = process.argv[3];
const upstreamVersion = process.argv[4];
const upstream = JSON.parse(fs.readFileSync(upstreamPath, "utf8"));
const bumpedVersion = upstreamVersion.replace(/(\d+)$/, m => String(Number(m) + 1));
const shim = {
  name: "@earendil-works/pi-coding-agent",
  version: bumpedVersion,
  description:
    "TYPE-ONLY vendored copy of @earendil-works/pi-coding-agent@" + upstreamVersion +
    " (dist/*.d.ts only, npm-shrinkwrap.json removed so the dependency tree resolves" +
    " normally and stays subject to the root package overrides). Rebuilt by" +
    " scripts/sync-peer-types.sh.",
  type: "module",
  main: "./dist/index.js",
  types: "./dist/index.d.ts",
  exports: {
    ".": { types: "./dist/index.d.ts", import: "./dist/index.js" },
    "./rpc-entry": { types: "./dist/rpc-entry.d.ts", import: "./dist/rpc-entry.js" },
    "./client": { types: "./dist/client/index.d.ts", import: "./dist/client/index.js" },
  },
  dependencies: upstream.dependencies,
  peerDependencies: upstream.peerDependencies,
};
fs.writeFileSync(outputPath, JSON.stringify(shim, null, 2) + "\n");
console.log("Wrote " + outputPath + " (version " + bumpedVersion + ")");
EOF

echo "Done. Reinstall dependencies to pick up the refreshed types:"
echo "  rm -rf node_modules package-lock.json && npm install"
