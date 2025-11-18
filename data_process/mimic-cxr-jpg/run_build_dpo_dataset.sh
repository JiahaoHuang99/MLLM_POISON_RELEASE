rm log_build_dpo_dataset.txt

# set env 
export API_PROVIDER="VAPI"
export API_KEY="XXXXXX"

# run build_dpo_dataset.py
nohup python build_dpo_dataset.py > log_build_dpo_dataset.txt &



