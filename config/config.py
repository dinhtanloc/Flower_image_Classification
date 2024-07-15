# config root dir, library path
import os, sys
__script_path=os.path.abspath(globals().get('__file__','.'))
__script_dir = os.path.dirname(__script_path)
root_dir = os.path.abspath(os.path.dirname(f'{__script_dir}')).replace("\\", "/")
print(root_dir)
source_dir    = os.path.join(root_dir, "prj").replace("\\", "/")
libraries_dir = os.path.join(root_dir, "libraries").replace("\\", "/")
include_dirs  = [source_dir]
for lib in include_dirs:
    if lib not in sys.path: sys.path.insert(0, lib)
# np.set_printoptions(precision=2, suppress=True, formatter={'float': '{: 0.4f}'.format}, linewidth=1000)

# common info of project
data_dir    = os.path.join(root_dir, "data").replace("\\", "/")
# dataset_dir = os.path.join(data_dir, "datasets").replace("\\", "/")
exps_dir     = os.path.join(root_dir, "exps").replace("\\", "/")
prj_dir     = os.path.join(root_dir, "prj").replace("\\", "/")

# path of module
models_dir     = os.path.join(root_dir, "models").replace("\\", "/")
weights_models_dir=os.path.join(models_dir, "weights").replace("\\", "/")
weights_prj_dir=os.path.join(models_dir, '4.build_and_improve', "weights").replace("\\", "/")
