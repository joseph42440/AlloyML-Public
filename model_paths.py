import pickle

if 'google.colab' in str(get_ipython()):
    model_dir = "AlloyML_Public/models"
else:
    model_dir = "models"

models = {"DoS": pickle.load(open(f"{model_dir}/DoS_Model_1619609637.sav", "rb")),
          "elongation": pickle.load(open(f"{model_dir}/Elongation_Model_1619609639.sav", "rb")),
          "tensile": pickle.load(open(f"{model_dir}/Tensile_Strength_Model_1619609643.sav", "rb")),
          "yield": pickle.load(open(f"{model_dir}/Yield_Strength_Model_1619609646.sav", "rb")),
          }

