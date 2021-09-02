echo "Evaluation function (trec nist): "$1
echo "Trec file referebce: " $2
echo "Trec file to-be evaluated"  $3
echo -e

for metric in "P" "ndcg_cut"; do
    $1 -m $metric.1 $2 $3
    $1 -m $metric.3 $2 $3
    $1 -m $metric.5 $2 $3
    echo -e
done

$1 -m recall.1000 -m map_cut.1000 -m ndcg_cut.1000 $2 $3
