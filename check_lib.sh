#!/bin/bash

# This script checks if required library exists and compiles it if needed

# check_lib.sh $(NX) $(NY) $(NZ) $(PROG_DESC) $(PROG_TYPE)

# echo ""
echo ""
echo "# looking for library /lib/libcugpe_$4_$1x$2x$3"
CUGPE_HOME="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "# cugpe home: $CUGPE_HOME"

cd "$CUGPE_HOME/dev"
make -e NX=$1 NY=$2 NZ=$3 PROG_DESC=$4 INTERACTIONS=$5
cd "$CUGPE_HOME"

# if [ -f "$CUGPE_HOME/lib/libcugpe_$4_$1x$2x$3" ];
# then
#     echo '# library found'
# else
#     echo '# building required library'
#     
#     if [ -f "$CUGPE_HOME/lib/libcugpe_$4_$1x$2x$3" ];
#     then
#         echo '# library built'
#     else
#         echo '# building library failed'
#     fi
# fi

echo ""
# echo ""