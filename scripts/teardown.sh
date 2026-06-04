#!/usr/bin/env bash
# Destroy all AWS resources to stop incurring charges.
set -euo pipefail
cd "$(dirname "$0")/../infra"
cdk destroy --force
echo "All resources destroyed."
