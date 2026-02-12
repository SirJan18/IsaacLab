#!/bin/bash
# Build with error checking using Python script
set -e
set -o pipefail

# Use container.py to start (which triggers build if needed)
./container.py start 2>&1 | tee build.log
BUILD_EXIT_CODE=$?

# Unset fail-fast flags
set +e
set +o pipefail

# Only print success message if build succeeded
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "✅ BUILD SUCCESSFUL - Isaac Lab container ready"
fi

exit $BUILD_EXIT_CODE

