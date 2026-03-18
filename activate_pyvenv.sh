#! /usr/bin/env bash

function usage() {
    echo "Run with \`source activate_pyvenv.sh\`"
    echo "  -h, --help    Show this help message and exit"
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    return 0
fi

source /home/juan95/pyvenv/pose_est/bin/activate 
