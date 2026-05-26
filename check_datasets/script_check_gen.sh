# Check number of each
for group in mida:3 vasc:4 skull:5; do
    name=${group%%:*}
    expect=${group##*:}
    for d in *_${name}/gen*/; do
        out="$d"
        bad=0
        for p in L S 4 C; do
            c=$(ls "$d"/${p}* 2>/dev/null | wc -l)
            out="$out $p=$c"
            [ "$c" -ne "$expect" ] && bad=1
        done
        [ $bad -eq 1 ] && echo "$out (≠$expect)"
    done
done




# Remove gen folder with the bad number of ele
for group in mida:3 vasc:4 skull:5; do
    name=${group%%:*}
    expect=${group##*:}
    for d in *_${name}/gen*/; do
        bad=0
        for p in L S 4 C; do
            c=$(ls "$d"/${p}* 2>/dev/null | wc -l)
            if [ "$c" -ne "$expect" ]; then
                echo "ERREUR: $d $p=$c (≠$expect)"
                bad=1
            fi
        done
        if [ "$bad" -eq 1 ]; then
            echo "Suppression $d"
            rm -rf "$d"
        fi
    done
done



# Compute the expected number of job to run

#!/bin/bash

declare -A expect_per_gen=( [mida]=3 [vasc]=4 [skull]=5 )
target=660

for group in mida vasc skull; do
    count=$(ls *_${group}/gen*/C* 2>/dev/null | wc -l)
    expect=${expect_per_gen[$group]}
    diff=$(( count - target ))

    if [ "$diff" -eq 0 ]; then
        echo "$group: OK ($count fichiers, aucun gen* à relancer)"
    elif [ "$diff" -lt 0 ]; then
        missing=$(( -diff ))
        need=$(( (missing + expect - 1) / expect ))  # arrondi vers le haut
        echo "$group: $count fichiers → manque $missing → relancer $need gen*"
    else
        echo "$group: $count fichiers → $diff fichiers en trop"
    fi
done