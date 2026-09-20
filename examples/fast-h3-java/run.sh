#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mvn -q package
exec java --enable-native-access=ALL-UNNAMED -cp 'target/classes:target/dependency/*' inc.reactor.cookbook.FastH3Demo "$@"
