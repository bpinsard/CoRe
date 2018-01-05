#!/bin/bash
for f in `ls vmrk/` ; do
    sed -E "s/Mk([[:digit:]]+)=([[:alpha:]]+),(.*),([[:digit:]]+),([[:digit:]]+),([[:digit:]]+)/Mk\1=\2,\3,\4,0,\6/g" vmrk/$f > vmrk_sp0dur/$f ;
done
