for net in deephic_small hicsr_small hicplus_small deephic_big hicsr_big hicplus_big vehicle down
do
	for chro in 4 14 16 20
	do
		echo $chro
		3DChromatin_ReplicateQC run_all --metadata_samples hicqc_inputs/metric_${net}_${chro}.samples --metadata_pairs hicqc_inputs/metric_${net}_${chro}.pairs --bins hicqc_inputs/bins_${chro}.bed.gz --outdir hicqc_results/${net}_${chro}  --concise_analysis
	done
done
