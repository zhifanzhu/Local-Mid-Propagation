command! MakeTags !ctags -f .tags -R --exclude=data --exclude=workvids --exclude=work_dirs --exclude=configs --exclude=zoo

set tags+=../mmcv/.tags
set tags+=~/.conda/envs/mmlab/lib/python3.7/site-packages/mmcv-0.2.7-py3.7.egg/mmcv/.tags
set tags+=~/.conda/envs/mmlab/lib/python3.7/site-packages/torch/.tags

set wildignore+=data/**
set wildignore+=*.so

let mmcvpath='~/.conda/envs/mmlab/lib/python3.7/site-packages/mmcv-0.2.7-py3.7.egg/mmcv/**'
let dirs='.,'.system("find . -maxdepth 1 -type d | cut -d/ -f2 | grep '^[^.]' | grep -v 'data' | awk '{print}' ORS='/**,'")
let &path=dirs.mmcvpath
" Equivalatent
" exe "set path=".dirs
