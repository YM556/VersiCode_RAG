#python test_line_generate_chunk.py --loc local --with_rag True
path=$(python test_line_generate_chunk.py --loc local --with_rag True | tail -n 1)
echo $path
python clear_ans.py "$path"
python evaluate_line.py "$path"