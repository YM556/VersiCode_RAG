#python test_token_lifespan.py --sample_num 200 --loc remote --type all
python test_token_lifespan.py --sample_num 200 --loc remote --with_rag True --type all
#python test_token_lifespan.py --sample_num 200 --loc remote --type add
python test_token_lifespan.py --sample_num 200 --loc remote --with_rag True --type add
#python test_token_lifespan.py --sample_num 200 --loc remote --type deprecation
python test_token_lifespan.py --sample_num 200 --loc remote --type deprecation --with_rag True
#export HF_ENDPOINT=https://hf-mirror.com
#cd ../../Reterival
#python FAISS.py
#python FAISS.py --task block --queries /root/autodl-tmp/VersiCode-RAG/Reterival/datasets/library_source_code/queries_block.jsonl


cd ../../
python -m Generation.together_ai.test_block_generate_chunk --with_rag True --loc local
python -m Generation.together_ai.test_block_generate_chunk  --loc local
cd Generation/together_ai