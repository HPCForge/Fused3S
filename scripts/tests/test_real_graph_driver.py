import os

hidden = [64]

for hid in hidden:
	for data in ['reddit']:
		print("=> {}, hiddn: {}".format(data, hid))
		command = f"/pub/zitongl5/NCU-2024.3/ncu --set full -f --import-source yes --source-folders /pub/zitongl5/FTC-MM/DTC-SpMM --export FTC_MM_{data}_hidden{hid}_A30_imp.ncu-rep --kernel-name regex:TC_* python run_fusedMM.py --dataset {data}"
		os.system(command)
		command = f"/pub/zitongl5/NCU-2024.3/ncu --set full -f --import-source yes --source-folders /pub/zitongl5/FTC-MM/DTC-SpMM --export DTC_SpMM_{data}_hidden{hid}_A30.ncu-rep --kernel-name regex:spmm_forward* python run_DTC_SpMM.py --dataset {data}"
		os.system(command)
	print("----------------------------")
print("===========================")
