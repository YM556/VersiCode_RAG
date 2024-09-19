#python test_block_generate_chunk_org.py --loc local --with_rag True
#path=$(python test_block_generate_chunk_org.py --source library_source_code --loc local  | tail -n 1)
#cd ../../Evaluate
#echo $path
python clear_ans.py './tmp/meta-llama/Llama-3-70b-chat-hf/library_source_code_block.json'
python evaluate_block.py './tmp/meta-llama/Llama-3-70b-chat-hf/library_source_code_block.json'
#cd ../Generation/together_ai_scripts