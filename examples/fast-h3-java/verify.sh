#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mvn -q package
java --enable-native-access=ALL-UNNAMED -cp 'target/test-classes:target/classes:target/dependency/*' inc.reactor.cookbook.DemoChecks
